"""Per-patient Gradient local-radiomics bag extraction.

One patient -> one .npz bag (the MIL bag) + one .json QA sidecar.

Pipeline for one patient:
    CT.nii.gz + ROI_D_gradient.nii.gz
      -> 2 mm isotropic resampling (CT linear, mask nearest)
      -> non-overlapping 5x5x5 lattice anchored at voxel (0,0,0)
      -> retention rule (center in ROI, >= 27 ROI voxels, full in bounds)
      -> PyRadiomics original-image firstorder/GLCM/GLRLM/GLSZM per patch
      -> [N_patches, N_features] + coordinates
"""
from __future__ import annotations

import json
import logging
import os
import time

import numpy as np
import SimpleITK as sitk

import radiomics
from radiomics import featureextractor

from config_io import extraction_signature, load_radiomics_params, project_path
from patches import crop_block, enumerate_patches, patch_roi_stats
from resample import resample_ct_and_mask

radiomics.setVerbosity(logging.ERROR)
logging.getLogger("radiomics").setLevel(logging.ERROR)

NPZ_VERSION = 1

# rejection reason codes
R_CENTER_OUTSIDE = "center_outside_roi"
R_TOO_FEW_VOXELS = "roi_voxels_below_min"
R_EXTRACTION_ERROR = "pyradiomics_error"
R_NONFINITE = "nonfinite_features"
R_FEATURE_COUNT = "feature_count_mismatch"


def build_extractor(cfg):
    return featureextractor.RadiomicsFeatureExtractor(load_radiomics_params(cfg))


def _feature_keys(result):
    return sorted(k for k in result if not k.startswith("diagnostics_"))


def extract_patient(cfg, patient, extractor=None, out_dir=None):
    """Extract one patient's bag. `patient` needs PatientID / Histology / label."""
    t0 = time.time()
    pid = patient["PatientID"]
    extractor = extractor or build_extractor(cfg)
    out_dir = out_dir or project_path(cfg, *cfg["paths"]["features_dir"].split("/"))
    os.makedirs(out_dir, exist_ok=True)

    pdir = os.path.join(cfg["paths"]["vuong_root"], pid)
    ct = sitk.ReadImage(os.path.join(pdir, cfg["paths"]["ct_filename"]))
    mask = sitk.ReadImage(os.path.join(pdir, cfg["experiment"]["roi_filename"]))

    geom = {
        "native_ct_size": list(ct.GetSize()),
        "native_ct_spacing": [float(v) for v in ct.GetSpacing()],
        "native_ct_origin": [float(v) for v in ct.GetOrigin()],
        "native_ct_direction": [float(v) for v in ct.GetDirection()],
        "mask_matches_ct": bool(
            mask.GetSize() == ct.GetSize()
            and np.allclose(mask.GetSpacing(), ct.GetSpacing())
            and np.allclose(mask.GetOrigin(), ct.GetOrigin())
            and np.allclose(mask.GetDirection(), ct.GetDirection())),
    }
    if not geom["mask_matches_ct"]:
        raise RuntimeError("%s: mask geometry does not match CT" % pid)

    t_res = time.time()
    ct_r, mask_r, grid = resample_ct_and_mask(ct, mask, cfg)
    t_res = time.time() - t_res

    mask_arr = sitk.GetArrayFromImage(mask_r)
    n_roi_voxels_total = int((mask_arr > 0).sum())
    voxel_vol = float(np.prod(grid["spacing"]))

    pc = cfg["patches"]
    size = pc["size_voxels"]
    stride = pc["stride_voxels"]
    min_roi = int(pc["min_roi_voxels"])

    candidates, bbox = enumerate_patches(
        mask_arr, size, stride, bool(pc["require_full_in_bounds"]))

    rejected = {R_CENTER_OUTSIDE: 0, R_TOO_FEW_VOXELS: 0,
                R_EXTRACTION_ERROR: 0, R_NONFINITE: 0, R_FEATURE_COUNT: 0}
    errors = []
    feats = []
    names = None
    coords_index, coords_start, coords_world = [], [], []
    bbox_world_min, bbox_world_max = [], []
    roi_voxels, roi_fractions = [], []

    n_patch_voxels = int(np.prod(size))
    t_feat = time.time()
    for cand in candidates:
        start, center = cand["start"], cand["center"]

        # rule 1: patch centre must lie inside the ROI
        ci, cj, ck = (int(v) for v in center)
        if not (0 <= ck < mask_arr.shape[0] and 0 <= cj < mask_arr.shape[1]
                and 0 <= ci < mask_arr.shape[2]) or mask_arr[ck, cj, ci] == 0:
            rejected[R_CENTER_OUTSIDE] += 1
            continue

        # rule 2: at least `min_roi_voxels` ROI voxels in the block
        n_roi, _ = patch_roi_stats(mask_arr, start, size)
        if n_roi < min_roi:
            rejected[R_TOO_FEW_VOXELS] += 1
            continue

        # rule 3: PyRadiomics must return a complete, finite feature vector
        try:
            sub_ct = crop_block(ct_r, start, size)
            sub_mk = crop_block(mask_r, start, size)
            res = extractor.execute(sub_ct, sub_mk, label=1)
        except Exception as exc:                       # noqa: BLE001
            rejected[R_EXTRACTION_ERROR] += 1
            if len(errors) < 25:
                errors.append({"start": [int(v) for v in start],
                               "reason": R_EXTRACTION_ERROR,
                               "detail": str(exc)[:300]})
            continue

        keys = _feature_keys(res)
        if names is None:
            names = keys
        elif keys != names:
            rejected[R_FEATURE_COUNT] += 1
            if len(errors) < 25:
                errors.append({"start": [int(v) for v in start],
                               "reason": R_FEATURE_COUNT,
                               "detail": "got %d keys, expected %d"
                                         % (len(keys), len(names))})
            continue

        vec = np.array([float(res[k]) for k in names], dtype=np.float64)
        if not np.all(np.isfinite(vec)):
            rejected[R_NONFINITE] += 1
            if len(errors) < 25:
                bad = [names[i] for i in np.where(~np.isfinite(vec))[0][:5]]
                errors.append({"start": [int(v) for v in start],
                               "reason": R_NONFINITE, "detail": ",".join(bad)})
            continue

        feats.append(vec)
        coords_start.append([int(v) for v in start])
        coords_index.append([ci, cj, ck])
        coords_world.append(list(ct_r.TransformContinuousIndexToPhysicalPoint(
            [float(ci), float(cj), float(ck)])))
        bbox_world_min.append(list(ct_r.TransformContinuousIndexToPhysicalPoint(
            [float(start[0]) - 0.5, float(start[1]) - 0.5, float(start[2]) - 0.5])))
        bbox_world_max.append(list(ct_r.TransformContinuousIndexToPhysicalPoint(
            [float(start[0] + size[0]) - 0.5, float(start[1] + size[1]) - 0.5,
             float(start[2] + size[2]) - 0.5])))
        roi_voxels.append(n_roi)
        roi_fractions.append(n_roi / float(n_patch_voxels))
    t_feat = time.time() - t_feat

    n_kept = len(feats)
    names = names or []
    features = (np.vstack(feats) if n_kept
                else np.zeros((0, len(names)), dtype=np.float64))

    qa = {
        "patient_id": pid,
        "histology": patient["Histology"],
        "label": int(patient["label"]),
        "npz_version": NPZ_VERSION,
        "config_signature": extraction_signature(cfg),
        "roi": cfg["experiment"]["roi"],
        "geometry": geom,
        "resampled_grid": {
            "size": [int(v) for v in grid["size"]],
            "spacing": [float(v) for v in grid["spacing"]],
            "origin": [float(v) for v in grid["origin"]],
            "direction": [float(v) for v in grid["direction"]],
        },
        "roi_voxels_resampled": n_roi_voxels_total,
        "roi_volume_mm3_resampled": n_roi_voxels_total * voxel_vol,
        "roi_bbox_ijk": ({"min": [int(v) for v in bbox["min_ijk"]],
                          "max": [int(v) for v in bbox["max_ijk"]]} if bbox else None),
        "n_candidate_patches": len(candidates),
        "n_retained_patches": n_kept,
        "n_rejected_patches": len(candidates) - n_kept,
        "rejected_by_reason": rejected,
        "rejection_examples": errors,
        "n_features": int(features.shape[1]),
        "feature_matrix_shape": list(features.shape),
        "nonfinite_in_saved_matrix": int((~np.isfinite(features)).sum()) if n_kept else 0,
        "roi_voxel_count_min_max": ([int(min(roi_voxels)), int(max(roi_voxels))]
                                    if roi_voxels else None),
        "timing_seconds": {
            "resample": round(t_res, 3),
            "features": round(t_feat, 3),
            "total": round(time.time() - t0, 3),
            "per_candidate_ms": (round(1000.0 * t_feat / len(candidates), 3)
                                 if candidates else None),
        },
    }

    npz_path = os.path.join(out_dir, "%s.npz" % pid)
    np.savez_compressed(
        npz_path,
        features=features.astype(np.float32),
        feature_names=np.array(names, dtype=object),
        coords_index=np.array(coords_index, dtype=np.int32).reshape(n_kept, 3),
        coords_start_index=np.array(coords_start, dtype=np.int32).reshape(n_kept, 3),
        coords_world=np.array(coords_world, dtype=np.float64).reshape(n_kept, 3),
        bbox_world_min=np.array(bbox_world_min, dtype=np.float64).reshape(n_kept, 3),
        bbox_world_max=np.array(bbox_world_max, dtype=np.float64).reshape(n_kept, 3),
        roi_voxels=np.array(roi_voxels, dtype=np.int32),
        roi_fraction=np.array(roi_fractions, dtype=np.float32),
        patient_id=np.array(pid),
        histology=np.array(patient["Histology"]),
        label=np.array(int(patient["label"]), dtype=np.int8),
        roi_name=np.array(cfg["experiment"]["roi"]),
        config_signature=np.array(qa["config_signature"]),
        npz_version=np.array(NPZ_VERSION, dtype=np.int32),
        grid_size=np.array(grid["size"], dtype=np.int32),
        grid_spacing=np.array(grid["spacing"], dtype=np.float64),
        grid_origin=np.array(grid["origin"], dtype=np.float64),
        grid_direction=np.array(grid["direction"], dtype=np.float64),
        patch_size=np.array(size, dtype=np.int32),
        patch_stride=np.array(stride, dtype=np.int32),
    )
    with open(os.path.join(out_dir, "%s.qa.json" % pid), "w", encoding="utf-8") as fh:
        json.dump(qa, fh, indent=2)
    qa["_npz_path"] = npz_path
    return qa


REQUIRED_KEYS = ["features", "feature_names", "coords_index", "coords_start_index",
                 "coords_world", "bbox_world_min", "bbox_world_max", "roi_voxels",
                 "roi_fraction", "patient_id", "label", "config_signature",
                 "grid_size", "grid_spacing", "grid_origin", "grid_direction"]


def validate_existing(path, pid, signature, expect_label=None):
    """True if `path` is a complete bag produced by the current configuration."""
    try:
        with np.load(path, allow_pickle=True) as z:
            for k in REQUIRED_KEYS:
                if k not in z:
                    return False, "missing key %s" % k
            if str(z["patient_id"]) != pid:
                return False, "patient_id mismatch"
            if str(z["config_signature"]) != signature:
                return False, "config signature changed"
            f = z["features"]
            if f.ndim != 2:
                return False, "features not 2-D"
            n = f.shape[0]
            if n and not np.all(np.isfinite(f)):
                return False, "nonfinite features stored"
            if len(z["feature_names"]) != f.shape[1]:
                return False, "feature_names length mismatch"
            for k in ["coords_index", "coords_start_index", "coords_world",
                      "bbox_world_min", "bbox_world_max"]:
                if z[k].shape != (n, 3):
                    return False, "%s shape mismatch" % k
            if z["roi_voxels"].shape != (n,) or z["roi_fraction"].shape != (n,):
                return False, "roi_voxels/roi_fraction shape mismatch"
            if expect_label is not None and int(z["label"]) != int(expect_label):
                return False, "label mismatch"
    except Exception as exc:                           # noqa: BLE001
        return False, "unreadable: %s" % str(exc)[:200]
    return True, "ok"
