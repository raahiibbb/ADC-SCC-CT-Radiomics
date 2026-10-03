"""Phase 8A - GTV, 8 mm rim and GTV+Rim for the external cohort.

This reproduces, exactly, the region definitions the LUNG1 arm of the project
has used since Phase 4:

    GTV      = the primary-tumour mask on the native CT grid
    GTV+Rim  = { phi <= 8 mm }  OR  GTV          (Vuong et al. 2020 ROI E)
    Rim      = GTV+Rim  MINUS  GTV               (Vuong et al. 2020 ROI C)

`phi` is the signed distance to the GTV surface in physical millimetres,
computed with the `fine_levelset_edt` method that
`Vuong5ROI/src/vuong5roi/geometry.py` validated against analytically known
spheres (8 mm expansion: -1.3 % volume error, Dice 0.985; the best of the four
benchmarked implementations).  The binary mask is linearly interpolated onto a
1 mm isotropic grid, its 0.5 iso-level is taken as the sub-voxel surface, the
Euclidean distance is computed there and sampled back at the native voxel
centres.  Fully 3D, spacing / origin / direction aware, never voxel-count
based and never per-slice.

The union is anchored on the GTV mask itself (`| gtv`), exactly as the LUNG1
builder does, which makes `GTV subset of GTV+Rim` exact by construction rather
than dependent on where the sub-voxel iso-surface falls.

Nothing under `Vuong5ROI/` is imported, executed or modified; it is a read-only
reference implementation.  The LUNG1 "cleaned" ROI variants are NOT reproduced,
because Phases 4-7A used the RAW masks.
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

_FAR = 1.0e6


def zyx_spacing(img: sitk.Image):
    """SimpleITK spacing is (x, y, z); numpy arrays are (z, y, x)."""
    sx, sy, sz = img.GetSpacing()
    return (float(sz), float(sy), float(sx))


def copy_geometry(arr: np.ndarray, ref: sitk.Image) -> sitk.Image:
    out = sitk.GetImageFromArray(np.ascontiguousarray(arr.astype(np.uint8)))
    out.CopyInformation(ref)
    return sitk.Cast(out, sitk.sitkUInt8)


def same_geometry(a: sitk.Image, b: sitk.Image, tol: float = 1e-3) -> bool:
    if tuple(a.GetSize()) != tuple(b.GetSize()):
        return False
    for x, y in ((a.GetSpacing(), b.GetSpacing()),
                 (a.GetOrigin(), b.GetOrigin()),
                 (a.GetDirection(), b.GetDirection())):
        if np.max(np.abs(np.asarray(x) - np.asarray(y))) > tol:
            return False
    return True


def _bbox(mask: np.ndarray):
    idx = np.argwhere(mask)
    if idx.size == 0:
        return None
    return idx.min(axis=0), idx.max(axis=0) + 1


def _fine_levelset_distance(m: np.ndarray, samp, fine_mm: float) -> np.ndarray:
    shp = np.asarray(m.shape)
    samp = np.asarray(samp, dtype=float)
    if not m.any():
        return np.full(m.shape, _FAR, dtype=np.float32)

    n_f = np.maximum(np.ceil((shp - 1) * samp / fine_mm).astype(int) + 1, 2)
    axes = [np.linspace(0.0, float(shp[i] - 1), n_f[i]) for i in range(3)]
    coords = np.stack(np.meshgrid(*axes, indexing="ij"), axis=0)
    mf = ndimage.map_coordinates(m.astype(np.float32), coords, order=1,
                                 mode="constant", cval=0.0) >= 0.5

    if mf.all() or not mf.any():
        return (ndimage.distance_transform_edt(~m, sampling=samp)
                - ndimage.distance_transform_edt(m, sampling=samp)).astype(np.float32)

    fs = (fine_mm, fine_mm, fine_mm)
    phi_f = (ndimage.distance_transform_edt(~mf, sampling=fs)
             - ndimage.distance_transform_edt(mf, sampling=fs)).astype(np.float32)

    back = np.stack(np.meshgrid(
        *[np.arange(shp[i]) * (n_f[i] - 1) / max(shp[i] - 1, 1)
          for i in range(3)], indexing="ij"), axis=0)
    return ndimage.map_coordinates(phi_f, back, order=1,
                                   mode="nearest").astype(np.float32)


def signed_distance_field(mask_arr: np.ndarray, samp, method="fine_levelset_edt",
                          crop_margin_mm=None, fine_mm=1.0) -> np.ndarray:
    """Signed distance in mm: negative inside `mask_arr`, positive outside."""
    if method != "fine_levelset_edt":
        raise ValueError("Phase 8A freezes the validated method "
                         "'fine_levelset_edt'; got %r" % method)
    mask = mask_arr > 0
    phi = np.full(mask.shape, _FAR, dtype=np.float32)
    bb = _bbox(mask)
    if bb is None:
        return phi
    lo, hi = bb
    samp = np.asarray(samp, dtype=float)

    if crop_margin_mm is None:
        sl = tuple(slice(0, s) for s in mask.shape)
    else:
        pad = np.ceil(crop_margin_mm / samp).astype(int) + 2
        lo2 = np.maximum(lo - pad, 0)
        hi2 = np.minimum(hi + pad, np.asarray(mask.shape))
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo2, hi2))

    phi[sl] = _fine_levelset_distance(mask[sl], samp, fine_mm)
    return phi


def touches_volume_boundary(mask_arr: np.ndarray) -> bool:
    m = mask_arr > 0
    return bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()
                or m[:, :, 0].any() or m[:, :, -1].any())


def build_regions(gtv: sitk.Image, cfg_regions: dict):
    """Return (masks, info) with masks = {gtv, rim, gtv_rim} on the GTV grid."""
    expansion_mm = float(cfg_regions["expansion_mm"])
    fine_mm = float(cfg_regions["fine_grid_mm"])
    method = str(cfg_regions["distance_method"])
    ver = cfg_regions["verification"]

    gtv_arr = sitk.GetArrayViewFromImage(gtv) > 0
    samp = zyx_spacing(gtv)
    phi = signed_distance_field(gtv_arr, samp, method=method,
                                crop_margin_mm=expansion_mm + 4.0 * max(samp),
                                fine_mm=fine_mm)

    dil = (phi <= expansion_mm) | gtv_arr        # ROI E: GTV+Rim
    rim = dil & ~gtv_arr                         # ROI C: the 8 mm rim

    n_gtv = int(gtv_arr.sum())
    n_rim = int(rim.sum())
    n_union = int(dil.sum())
    n_int = int((gtv_arr & rim).sum())
    n_union_mismatch = int(((gtv_arr | rim) ^ dil).sum())
    n_gtv_outside = int((gtv_arr & ~dil).sum())

    if bool(ver["require_disjoint"]) and n_int != 0:
        raise RuntimeError("GTV and rim share %d voxels" % n_int)
    if bool(ver["require_union_equals_gtv_rim"]) and n_union_mismatch != 0:
        raise RuntimeError("GTV u rim differs from GTV+Rim in %d voxels" % n_union_mismatch)
    if bool(ver["require_gtv_subset_of_gtv_rim"]) and n_gtv_outside != 0:
        raise RuntimeError("%d GTV voxels lie outside GTV+Rim" % n_gtv_outside)
    minv = int(ver["min_region_voxels"])
    for nm, n in (("gtv", n_gtv), ("rim", n_rim), ("gtv_rim", n_union)):
        if n < minv:
            raise RuntimeError("region %s has %d voxels (< %d)" % (nm, n, minv))

    voxel_mm3 = float(np.prod(np.asarray(gtv.GetSpacing(), dtype=float)))
    info = {
        "distance_method": method,
        "fine_grid_mm": fine_mm,
        "expansion_mm": expansion_mm,
        "native_spacing_mm": [float(v) for v in gtv.GetSpacing()],
        "gtv_touches_volume_boundary": touches_volume_boundary(gtv_arr),
        "sdf_zero_level_voxel_diff": int(np.count_nonzero((phi <= 0.0) != gtv_arr)),
        "native_gtv_voxels": n_gtv,
        "native_rim_voxels": n_rim,
        "native_gtv_rim_voxels": n_union,
        "native_gtv_rim_intersection_voxels": n_int,
        "native_union_vs_gtv_rim_mismatch_voxels": n_union_mismatch,
        "native_gtv_outside_gtv_rim_voxels": n_gtv_outside,
        "native_gtv_volume_mm3": n_gtv * voxel_mm3,
        "native_rim_volume_mm3": n_rim * voxel_mm3,
        "native_gtv_rim_volume_mm3": n_union * voxel_mm3,
        "native_rim_fraction_of_gtv_rim": (n_rim / n_union) if n_union else float("nan"),
        "max_distance_in_rim_mm": float(phi[rim].max()) if n_rim else None,
        "disjoint_ok": n_int == 0,
        "union_ok": n_union_mismatch == 0,
        "subset_ok": n_gtv_outside == 0,
    }
    masks = {"gtv": copy_geometry(gtv_arr, gtv),
             "rim": copy_geometry(rim, gtv),
             "gtv_rim": copy_geometry(dil, gtv)}
    return masks, info
