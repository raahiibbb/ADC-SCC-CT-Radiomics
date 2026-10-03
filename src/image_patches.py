"""Phase 6A - 2.5D CT image crops on the FROZEN GTV+Rim patch centres.

One patient -> one .npz of [N, 3, 32, 32] int16 CT crops, where row i of the
image file corresponds to row i of the frozen radiomics bag
``patch_features/gtv_rim/<PID>.npz``.

Nothing here samples a new lattice, moves a patch centre, drops an instance or
reads a label into the geometry.  The only inputs are:

  * the frozen bag        - patch centres, world coordinates, grid, label, fold
  * the read-only CT      - Vuong5ROI/outputs/full_cohort/<PID>/CT.nii.gz

Index convention (identical to src/patches.py): index tuples are (i, j, k) in
SimpleITK order; the numpy volume is indexed [k, j, i].  With identity direction
cosines and an LPS frame, +i is patient-Left, +j is Posterior, +k is Superior.

Crop geometry, for a frozen patch centre c = (ci, cj, ck):

    window start along each axis = c - 16, 32 samples, so the centre voxel c
    lands at OUTPUT INDEX 16 of both in-plane axes of all three views.

    channel 0  axial    fixed k = ck   rows = j ascending   cols = i ascending
    channel 1  coronal  fixed j = cj   rows = k ascending   cols = i ascending
    channel 2  sagittal fixed i = ci   rows = k ascending   cols = j ascending

The three planes therefore intersect at exactly the voxel c, and
``img[0, 16, 16] == img[1, 16, 16] == img[2, 16, 16] == volume[ck, cj, ci]``.
The 32-pixel window is even, so its geometric midpoint sits at index c - 0.5
(1.0 mm from the centre) on every axis of every view - a constant, deterministic
offset, never a per-instance decision.

The stored arrays carry NO cosmetic flips.  Display flips (superior at the top
of the coronal/sagittal panels) are applied only in the QA figures.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

NPZ_VERSION = 1

REQUIRED_KEYS = [
    "images", "patch_index", "coords_index", "coords_world", "coords_start_index",
    "crop_window_start_index", "n_padded_pixels", "padded", "patient_id",
    "histology", "label", "outer_fold", "crop_signature",
    "source_extraction_signature", "source_bag_sha256", "source_coords_sha256",
    "grid_size", "grid_spacing", "grid_origin", "grid_direction",
    "hu_clip", "fov_mm", "image_size", "views", "pad_value_hu", "npz_version",
]


# --------------------------------------------------------------------- config
def load_config(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = os.path.abspath(path)
    cfg["_project_root"] = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
    return cfg


def ppath(cfg: dict, rel: str) -> str:
    return os.path.join(cfg["_project_root"], *rel.split("/"))


def crop_signature(cfg: dict) -> str:
    """Stable hash over everything that changes the numeric content of a crop."""
    relevant = {
        "roi": cfg["experiment"]["roi"],
        "roi_filename": cfg["experiment"]["roi_filename"],
        "preprocessing": cfg["preprocessing"],
        "crop": cfg["crop"],
        "source_extraction_signature": cfg["source"]["expected_extraction_signature"],
        "ct_filename": cfg["paths"]["ct_filename"],
    }
    blob = json.dumps(relevant, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def transform_signature(cfg: dict) -> str:
    blob = json.dumps(cfg["transform"], sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def coords_fingerprint(coords_index: np.ndarray, coords_world: np.ndarray) -> str:
    """Canonical hash of the patch ordering + geometry of one bag.

    This is the identity manifest entry: it is computed the same way from the
    frozen radiomics bag and from the Phase-6A image file, so equality proves the
    two files describe the same instances in the same order.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(coords_index, dtype=np.int32).tobytes())
    h.update(np.ascontiguousarray(coords_world, dtype=np.float64).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- frozen split
def load_outer_folds(cfg: dict) -> Dict[str, int]:
    """PatientID -> the outer fold in which the patient is the held-out test case."""
    import pandas as pd
    df = pd.read_csv(ppath(cfg, cfg["source"]["splits_file"]))
    test = df[df["split"] == "test"]
    if test["PatientID"].duplicated().any():
        raise RuntimeError("frozen split assigns a patient to more than one test fold")
    return {str(r.PatientID): int(r.fold) for r in test.itertuples()}


def load_frozen_bag(cfg: dict, pid: str) -> dict:
    """Read the frozen GTV+Rim bag.  Read-only; nothing is written back."""
    path = os.path.join(ppath(cfg, cfg["source"]["features_dir"]), "%s.npz" % pid)
    with np.load(path, allow_pickle=True) as d:
        bag = {
            "path": path,
            "patient_id": str(d["patient_id"]),
            "histology": str(d["histology"]),
            "label": int(d["label"]),
            "config_signature": str(d["config_signature"]),
            "coords_index": np.asarray(d["coords_index"], dtype=np.int32),
            "coords_start_index": np.asarray(d["coords_start_index"], dtype=np.int32),
            "coords_world": np.asarray(d["coords_world"], dtype=np.float64),
            "grid_size": np.asarray(d["grid_size"], dtype=np.int32),
            "grid_spacing": np.asarray(d["grid_spacing"], dtype=np.float64),
            "grid_origin": np.asarray(d["grid_origin"], dtype=np.float64),
            "grid_direction": np.asarray(d["grid_direction"], dtype=np.float64),
            "n_patches": int(np.asarray(d["features"]).shape[0]),
        }
    if bag["patient_id"] != pid:
        raise RuntimeError("%s: bag patient_id is %s" % (pid, bag["patient_id"]))
    expected_sig = cfg["source"]["expected_extraction_signature"]
    if bag["config_signature"] != expected_sig:
        raise RuntimeError("%s: bag signature %s != expected %s"
                           % (pid, bag["config_signature"], expected_sig))
    return bag


# ------------------------------------------------------------------- CT volume
def resampled_ct(cfg: dict, pid: str, with_mask: bool = False):
    """Resample the read-only CT onto the SAME 2 mm grid the frozen bags use."""
    import SimpleITK as sitk
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from resample import reference_grid, resample_to_grid   # noqa: WPS433

    pdir = os.path.join(cfg["paths"]["vuong_root"], pid)
    ct = sitk.ReadImage(os.path.join(pdir, cfg["paths"]["ct_filename"]))
    pre = cfg["preprocessing"]
    grid = reference_grid(ct, pre["resample_spacing_mm"])
    ct_r = resample_to_grid(sitk.Cast(ct, sitk.sitkFloat32), grid,
                            pre["ct_interpolator"], pre["ct_default_value"])
    mask_r = None
    if with_mask:
        mask = sitk.ReadImage(os.path.join(pdir, cfg["experiment"]["roi_filename"]))
        mask_r = resample_to_grid(sitk.Cast(mask, sitk.sitkUInt8), grid,
                                  pre["mask_interpolator"], 0)
        mask_r = sitk.Cast(mask_r > 0, sitk.sitkUInt8)
    return ct_r, mask_r, grid


def verify_grid(bag: dict, grid: dict, ct_r) -> None:
    """The Phase-6A grid must be bit-for-bit the grid the frozen centres index."""
    if list(bag["grid_size"]) != [int(v) for v in grid["size"]]:
        raise RuntimeError("grid size mismatch: bag %s, phase6a %s"
                           % (list(bag["grid_size"]), list(grid["size"])))
    for key, arr in (("spacing", bag["grid_spacing"]), ("origin", bag["grid_origin"]),
                     ("direction", bag["grid_direction"])):
        if not np.allclose(np.asarray(grid[key], dtype=np.float64), arr, atol=1e-9):
            raise RuntimeError("grid %s mismatch" % key)
    # every frozen world coordinate must reproduce from the Phase-6A image geometry
    ci = bag["coords_index"]
    if ci.shape[0]:
        got = np.array([ct_r.TransformContinuousIndexToPhysicalPoint(
            [float(a), float(b), float(c)]) for a, b, c in ci], dtype=np.float64)
        err = float(np.abs(got - bag["coords_world"]).max())
        if err > 1e-6:
            raise RuntimeError("world-coordinate round-trip error %.3e mm" % err)


# ------------------------------------------------------------------- the crops
def clip_to_int16(ct_arr: np.ndarray, hu_clip: Sequence[float]) -> np.ndarray:
    """Clip HU, round to nearest, cast to int16.  Order is fixed and documented."""
    lo, hi = float(hu_clip[0]), float(hu_clip[1])
    return np.rint(np.clip(ct_arr, lo, hi)).astype(np.int16)


def crop_stack(vol16: np.ndarray, coords_index: np.ndarray, cfg: dict):
    """[N, 3, 32, 32] int16 crops + per-instance padded-pixel counts.

    `vol16` is the clipped int16 volume indexed [k, j, i].  Out-of-volume samples
    are filled with the configured pad value; the instance is never dropped.
    """
    ccfg = cfg["crop"]
    size = int(ccfg["image_size"][0])
    if list(ccfg["image_size"]) != [size, size]:
        raise RuntimeError("only square crops are supported")
    off = int(ccfg["centre_offset_voxels"])
    if off * 2 != size:
        raise RuntimeError("centre_offset_voxels must be image_size // 2")
    pad_val = int(ccfg["pad_value_hu"])
    n = int(coords_index.shape[0])

    # pad by `off` on every side so a window start of (c - off) becomes exactly c
    padded = np.pad(vol16, off, mode="constant", constant_values=pad_val)
    valid = np.pad(np.ones(vol16.shape, dtype=np.uint8), off,
                   mode="constant", constant_values=0)

    images = np.empty((n, 3, size, size), dtype=np.int16)
    n_pad = np.zeros(n, dtype=np.int32)
    for t in range(n):
        ci, cj, ck = (int(v) for v in coords_index[t])
        # channel 0 axial: fixed k, rows j, cols i
        images[t, 0] = padded[ck + off, cj:cj + size, ci:ci + size]
        # channel 1 coronal: fixed j, rows k, cols i
        images[t, 1] = padded[ck:ck + size, cj + off, ci:ci + size]
        # channel 2 sagittal: fixed i, rows k, cols j
        images[t, 2] = padded[ck:ck + size, cj:cj + size, ci + off]
        n_pad[t] = int(3 * size * size
                       - int(valid[ck + off, cj:cj + size, ci:ci + size].sum())
                       - int(valid[ck:ck + size, cj + off, ci:ci + size].sum())
                       - int(valid[ck:ck + size, cj:cj + size, ci + off].sum()))
    return images, n_pad


# ------------------------------------------------------------------ per patient
def extract_patient_images(cfg: dict, pid: str, out_dir: Optional[str] = None,
                           outer_fold: Optional[int] = None) -> dict:
    t0 = time.time()
    out_dir = out_dir or ppath(cfg, cfg["outputs"]["image_dir"])
    os.makedirs(out_dir, exist_ok=True)

    bag = load_frozen_bag(cfg, pid)
    if outer_fold is None:
        outer_fold = load_outer_folds(cfg)[pid]

    ct_r, _, grid = resampled_ct(cfg, pid)
    verify_grid(bag, grid, ct_r)

    import SimpleITK as sitk
    ct_arr = sitk.GetArrayFromImage(ct_r)                 # [k, j, i] float32
    hu_clip = cfg["crop"]["hu_clip"]
    vol16 = clip_to_int16(ct_arr, hu_clip)

    t_crop = time.time()
    images, n_pad = crop_stack(vol16, bag["coords_index"], cfg)
    t_crop = time.time() - t_crop

    n = int(images.shape[0])
    if n != bag["n_patches"]:
        raise RuntimeError("%s: produced %d images for %d frozen patches"
                           % (pid, n, bag["n_patches"]))
    lo, hi = int(hu_clip[0]), int(hu_clip[1])
    if n and (int(images.min()) < lo or int(images.max()) > hi):
        raise RuntimeError("%s: HU outside the clip window" % pid)

    off = int(cfg["crop"]["centre_offset_voxels"])
    sig = crop_signature(cfg)
    fp = coords_fingerprint(bag["coords_index"], bag["coords_world"])

    npz_path = os.path.join(out_dir, "%s.npz" % pid)
    np.savez_compressed(
        npz_path,
        images=images,
        patch_index=np.arange(n, dtype=np.int32),
        coords_index=bag["coords_index"],
        coords_world=bag["coords_world"],
        coords_start_index=bag["coords_start_index"],
        crop_window_start_index=(bag["coords_index"].astype(np.int32) - off),
        n_padded_pixels=n_pad,
        padded=(n_pad > 0),
        patient_id=np.array(pid),
        histology=np.array(bag["histology"]),
        label=np.array(bag["label"], dtype=np.int8),
        outer_fold=np.array(int(outer_fold), dtype=np.int8),
        crop_signature=np.array(sig),
        source_extraction_signature=np.array(bag["config_signature"]),
        source_bag_sha256=np.array(sha256_file(bag["path"])),
        source_coords_sha256=np.array(fp),
        grid_size=bag["grid_size"],
        grid_spacing=bag["grid_spacing"],
        grid_origin=bag["grid_origin"],
        grid_direction=bag["grid_direction"],
        hu_clip=np.array([lo, hi], dtype=np.int32),
        fov_mm=np.array(cfg["crop"]["fov_mm"], dtype=np.float64),
        image_size=np.array(cfg["crop"]["image_size"], dtype=np.int32),
        views=np.array(cfg["crop"]["views"], dtype=object),
        pad_value_hu=np.array(int(cfg["crop"]["pad_value_hu"]), dtype=np.int32),
        resample_spacing_mm=np.array(cfg["preprocessing"]["resample_spacing_mm"],
                                     dtype=np.float64),
        ct_interpolator=np.array(cfg["preprocessing"]["ct_interpolator"]),
        pad_rule=np.array(cfg["crop"]["pad_rule"]),
        npz_version=np.array(NPZ_VERSION, dtype=np.int32),
    )

    return {
        "patient_id": pid,
        "histology": bag["histology"],
        "label": bag["label"],
        "outer_fold": int(outer_fold),
        "n_images": n,
        "n_frozen_patches": bag["n_patches"],
        "n_padded_instances": int((n_pad > 0).sum()),
        "n_padded_pixels": int(n_pad.sum()),
        "hu_min": int(images.min()) if n else None,
        "hu_max": int(images.max()) if n else None,
        "crop_signature": sig,
        "source_extraction_signature": bag["config_signature"],
        "source_bag_sha256": sha256_file(bag["path"]),
        "source_coords_sha256": fp,
        "grid_size": [int(v) for v in bag["grid_size"]],
        "bytes": os.path.getsize(npz_path),
        "seconds": round(time.time() - t0, 3),
        "seconds_crop": round(t_crop, 3),
        "_npz_path": npz_path,
    }


def validate_existing(path: str, pid: str, signature: str,
                      n_expected: Optional[int] = None,
                      coords_fp: Optional[str] = None):
    """True if `path` is a complete image file written by the current crop config."""
    try:
        with np.load(path, allow_pickle=True) as z:
            for k in REQUIRED_KEYS:
                if k not in z:
                    return False, "missing key %s" % k
            if str(z["patient_id"]) != pid:
                return False, "patient_id mismatch"
            if str(z["crop_signature"]) != signature:
                return False, "crop signature changed"
            im = z["images"]
            if im.dtype != np.int16:
                return False, "images dtype is %s" % im.dtype
            if im.ndim != 4 or im.shape[1:] != (3, 32, 32):
                return False, "images shape %s" % (im.shape,)
            n = int(im.shape[0])
            if n_expected is not None and n != n_expected:
                return False, "n_images %d != frozen %d" % (n, n_expected)
            if coords_fp is not None and str(z["source_coords_sha256"]) != coords_fp:
                return False, "coords fingerprint mismatch"
            for k in ("coords_index", "coords_world", "coords_start_index",
                      "crop_window_start_index"):
                if z[k].shape != (n, 3):
                    return False, "%s shape mismatch" % k
            if z["patch_index"].shape != (n,) or z["n_padded_pixels"].shape != (n,):
                return False, "per-instance array shape mismatch"
            if not np.array_equal(z["patch_index"], np.arange(n, dtype=np.int32)):
                return False, "patch_index is not 0..N-1 in order"
            lo, hi = (int(v) for v in z["hu_clip"])
            if n and (int(im.min()) < lo or int(im.max()) > hi):
                return False, "HU outside the clip window"
    except Exception as exc:                                     # noqa: BLE001
        return False, "unreadable: %s" % str(exc)[:200]
    return True, "ok"


def eligible_patients(cfg: dict) -> List[str]:
    feat_dir = ppath(cfg, cfg["source"]["features_dir"])
    return sorted(f[:-4] for f in os.listdir(feat_dir) if f.endswith(".npz"))
