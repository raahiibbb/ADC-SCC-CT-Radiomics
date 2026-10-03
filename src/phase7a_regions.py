"""Phase 7A - deterministic derivation of the two GLOBAL regions.

    GTV  = ROI_A_GTV.nii.gz
    Rim  = ROI_E_GTV_Rim.nii.gz  MINUS  ROI_A_GTV.nii.gz

Both native masks are sampled onto ONE shared 2 mm reference grid derived from
the patient's own CT, with nearest-neighbour interpolation.  Because the two
masks share the native geometry and the target grid, every output voxel takes
its value from the SAME native voxel in both masks - so the subtraction is
exact, not approximate, and the three protocol properties

    GTV n Rim = 0
    GTV u Rim = GTV+Rim
    GTV subset of GTV+Rim            (native geometry as well)

hold by construction.  They are nevertheless VERIFIED for every patient and
recorded, because "holds by construction" is an argument, not evidence.

The Gradient ROI is not read here.  No "cleaned" ROI product is read here.
Nothing under Vuong5ROI/ or Dataset/ is ever written.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import SimpleITK as sitk

from resample import reference_grid, resample_to_grid


@dataclass
class PatientRegions:
    """The 2 mm CT plus the two disjoint global masks, on one shared grid."""
    patient_id: str
    ct: sitk.Image                    # float32, 2 mm isotropic
    masks: Dict[str, sitk.Image]      # "gtv" / "rim", uint8 0/1, same grid
    grid: dict
    geometry: dict                    # every verified quantity, for the QA record


def _as_bool_array(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayViewFromImage(img) > 0


def _same_geometry(a: sitk.Image, b: sitk.Image) -> bool:
    return (a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing())
            and np.allclose(a.GetOrigin(), b.GetOrigin())
            and np.allclose(a.GetDirection(), b.GetDirection()))


def build_regions(cfg: dict, patient_id: str) -> PatientRegions:
    paths = cfg["paths"]
    pre = cfg["preprocessing"]
    ver = cfg["regions"]["verification"]

    pdir = os.path.join(paths["vuong_root"], patient_id)
    ct = sitk.ReadImage(os.path.join(pdir, paths["ct_filename"]))
    gtv_n = sitk.ReadImage(os.path.join(pdir, paths["gtv_filename"]))
    rim_src_n = sitk.ReadImage(os.path.join(pdir, paths["gtv_rim_filename"]))

    for name, m in (("gtv", gtv_n), ("gtv_rim", rim_src_n)):
        if not _same_geometry(m, ct):
            raise RuntimeError("%s: %s mask geometry does not match the CT" % (patient_id, name))

    # ---- native-space check: GTV must be a subset of GTV+Rim ---------------
    g_nat = _as_bool_array(gtv_n)
    e_nat = _as_bool_array(rim_src_n)
    n_gtv_nat = int(g_nat.sum())
    n_gtvrim_nat = int(e_nat.sum())
    n_gtv_outside_nat = int((g_nat & ~e_nat).sum())
    if bool(ver["require_gtv_subset_of_gtv_rim_native"]) and n_gtv_outside_nat != 0:
        raise RuntimeError("%s: %d native GTV voxels lie outside GTV+Rim"
                           % (patient_id, n_gtv_outside_nat))

    # ---- one shared 2 mm grid ---------------------------------------------
    grid = reference_grid(ct, pre["resample_spacing_mm"])
    ct_r = resample_to_grid(sitk.Cast(ct, sitk.sitkFloat32), grid,
                            pre["ct_interpolator"], pre["ct_default_value"])
    gtv_r = resample_to_grid(sitk.Cast(gtv_n, sitk.sitkUInt8), grid,
                             pre["mask_interpolator"], 0)
    gtvrim_r = resample_to_grid(sitk.Cast(rim_src_n, sitk.sitkUInt8), grid,
                                pre["mask_interpolator"], 0)

    g = sitk.GetArrayFromImage(gtv_r) > 0
    e = sitk.GetArrayFromImage(gtvrim_r) > 0
    r = e & ~g                                   # THE rim definition

    # ---- the three protocol properties, verified ---------------------------
    n_gtv, n_rim, n_union_src = int(g.sum()), int(r.sum()), int(e.sum())
    n_intersect = int((g & r).sum())
    union = g | r
    n_union = int(union.sum())
    n_union_mismatch = int((union ^ e).sum())

    if bool(ver["require_disjoint"]) and n_intersect != 0:
        raise RuntimeError("%s: GTV and rim share %d voxels" % (patient_id, n_intersect))
    if bool(ver["require_union_equals_gtv_rim"]) and n_union_mismatch != 0:
        raise RuntimeError("%s: GTV u rim differs from GTV+Rim in %d voxels"
                           % (patient_id, n_union_mismatch))
    minv = int(ver["min_region_voxels"])
    for nm, n in (("gtv", n_gtv), ("rim", n_rim)):
        if n < minv:
            raise RuntimeError("%s: region %s has %d voxels (< %d)" % (patient_id, nm, n, minv))

    voxel_mm3 = float(np.prod(grid["spacing"]))
    geometry = {
        "PatientID": patient_id,
        "native_ct_size": list(ct.GetSize()),
        "native_ct_spacing": [float(v) for v in ct.GetSpacing()],
        "native_gtv_voxels": n_gtv_nat,
        "native_gtv_rim_voxels": n_gtvrim_nat,
        "native_gtv_outside_gtv_rim_voxels": n_gtv_outside_nat,
        "resampled_size": [int(v) for v in grid["size"]],
        "resampled_spacing": [float(v) for v in grid["spacing"]],
        "gtv_voxels": n_gtv,
        "rim_voxels": n_rim,
        "gtv_rim_voxels": n_union_src,
        "gtv_rim_intersection_voxels": n_intersect,
        "union_voxels": n_union,
        "union_vs_gtv_rim_mismatch_voxels": n_union_mismatch,
        "gtv_volume_mm3": n_gtv * voxel_mm3,
        "rim_volume_mm3": n_rim * voxel_mm3,
        "gtv_rim_volume_mm3": n_union_src * voxel_mm3,
        "rim_fraction_of_gtv_rim": (n_rim / n_union_src) if n_union_src else float("nan"),
        "disjoint_ok": n_intersect == 0,
        "union_ok": n_union_mismatch == 0,
        "native_subset_ok": n_gtv_outside_nat == 0,
    }

    rim_img = sitk.GetImageFromArray(r.astype(np.uint8))
    rim_img.CopyInformation(gtv_r)
    return PatientRegions(patient_id=patient_id, ct=ct_r,
                          masks={"gtv": gtv_r, "rim": rim_img},
                          grid=grid, geometry=geometry)
