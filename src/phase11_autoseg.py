"""Phase 11 - ONE box-seeded tumour segmentation for all three cohorts.

Per axial slice that carries a box: voxels inside the box with HU above a
threshold -> morphological opening -> hole filling; then in 3-D keep the
connected components that hold >= 10 % of the largest one.

  * Lung-PET-CT-Dx: boxes = the collection's PASCAL-VOC annotations.
  * LUNG1 / Radiogenomics: boxes = the tight per-slice bounding boxes of the
    expert GTV (what a box annotator would draw).  Their expert masks are used
    ONLY to measure this method (Dice) - every downstream feature, for every
    cohort, is computed from the automatic mask, so the mask source is the
    same everywhere and cannot act as a site fingerprint.

Usage:
  python src/phase11_autoseg.py validate [--thresholds -500 -650]   # Dice on 343 experts
  python src/phase11_autoseg.py run --threshold T                  # write AUTO_GTV for all
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk
from joblib import Parallel, delayed
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase10_data import cohort, load_cfg, p  # noqa: E402

LPCD_ROOT = "C:/LUNG_phase11/lung_pet_ct_dx"
AUTO_ROOT = "C:/LUNG_phase11/auto_masks"


def dev_paths(cfg, key, pid):
    root = p(cfg, "lung1_image_root") if key.startswith("LUNG1") else p(cfg, "external_image_root")
    return os.path.join(root, pid, "CT.nii.gz"), os.path.join(root, pid, "ROI_A_GTV.nii.gz")


def boxes_from_mask(m):
    """(z, y, x) bool -> per-slice tight bounding boxes, same shape."""
    b = np.zeros_like(m, dtype=bool)
    for z in np.where(m.reshape(m.shape[0], -1).any(1))[0]:
        ys, xs = np.nonzero(m[z])
        b[z, ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    return b


def segment(ct, box, thr):
    out = np.zeros_like(box, dtype=bool)
    st = ndi.generate_binary_structure(2, 1)
    for z in np.where(box.reshape(box.shape[0], -1).any(1))[0]:
        cand = (ct[z] > thr) & box[z]
        cand = ndi.binary_opening(cand, structure=st, iterations=1)
        lab, n = ndi.label(cand)
        if n == 0:
            continue
        vol = np.bincount(lab.ravel())[1:]
        cand = lab == (np.argmax(vol) + 1)
        out[z] = ndi.binary_fill_holes(cand)
    lab, n = ndi.label(out)
    if n > 1:
        vol = np.bincount(lab.ravel())[1:]
        out = np.isin(lab, np.where(vol >= 0.1 * vol.max())[0] + 1)
    return out


def load_dev(cfg, key, pid):
    ct_p, g_p = dev_paths(cfg, key, pid)
    ct = sitk.ReadImage(ct_p, sitk.sitkFloat32)
    g = sitk.Resample(sitk.ReadImage(g_p, sitk.sitkUInt8), ct, sitk.Transform(),
                      sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return ct, sitk.GetArrayFromImage(ct), sitk.GetArrayFromImage(g) > 0


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / s if s else 1.0


def validate_one(cfg, key, pid, thrs):
    try:
        _, ct, g = load_dev(cfg, key, pid)
        box = boxes_from_mask(g)
        r = {"PatientKey": key, "box_dice": dice(box, g)}
        for t in thrs:
            r["dice_%d" % t] = dice(segment(ct, box, t), g)
        return r
    except Exception as e:
        return {"PatientKey": key, "error": str(e)[:200]}


def run_one(cfg, key, pid, thr):
    dest = os.path.join(AUTO_ROOT, key.replace("::", "__"))
    os.makedirs(dest, exist_ok=True)
    if key.startswith("LPCD"):
        d = os.path.join(LPCD_ROOT, "nifti", pid)
        ct = sitk.ReadImage(os.path.join(d, "CT.nii.gz"), sitk.sitkFloat32)
        box = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(d, "BOX.nii.gz"))) > 0
        ct_path = os.path.join(d, "CT.nii.gz")
    else:
        ct, arr, g = load_dev(cfg, key, pid)
        box = boxes_from_mask(g)
        ct_path = dev_paths(cfg, key, pid)[0]
    arr = sitk.GetArrayFromImage(ct)
    m = segment(arr, box, thr)
    if m.sum() == 0:
        return {"PatientKey": key, "status": "failed: empty mask"}
    for name, a in (("AUTO_GTV", m), ("BOX", box)):
        im = sitk.GetImageFromArray(a.astype(np.uint8))
        im.CopyInformation(ct)
        sitk.WriteImage(im, os.path.join(dest, name + ".nii.gz"))
    vox = float(np.prod(ct.GetSpacing()))
    return {"PatientKey": key, "status": "ok", "ct_path": ct_path,
            "auto_volume_ml": m.sum() * vox / 1000.0, "box_volume_ml": box.sum() * vox / 1000.0}


def all_patients(cfg):
    c = cohort(cfg)[["PatientKey", "PatientID"]]
    sel = pd.read_csv(os.path.join(LPCD_ROOT, "series_selection.csv"))
    sel = sel[sel.selectable]
    ok = [pid for pid in sel.PatientID
          if os.path.isfile(os.path.join(LPCD_ROOT, "nifti", pid, "record.json"))
          and json.load(open(os.path.join(LPCD_ROOT, "nifti", pid, "record.json"))).get("status") == "ok"]
    l = pd.DataFrame({"PatientKey": ["LPCD::" + x for x in ok], "PatientID": ok})
    return pd.concat([c, l], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["validate", "run"])
    ap.add_argument("--thresholds", type=int, nargs="*", default=[-400, -500, -600, -700])
    ap.add_argument("--threshold", type=int)
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args()
    cfg = load_cfg()
    out_dir = os.path.join(p(cfg, "results_dir").replace("phase10", "phase11"))
    os.makedirs(out_dir, exist_ok=True)
    if a.step == "validate":
        c = cohort(cfg)
        rows = Parallel(n_jobs=a.jobs, verbose=2)(delayed(validate_one)(cfg, k, i, a.thresholds)
                                                  for k, i in zip(c.PatientKey, c.PatientID))
        df = pd.DataFrame(rows)
        df["site"] = df.PatientKey.str.split("::").str[0]
        df.to_csv(os.path.join(out_dir, "autoseg_validation.csv"), index=False)
        cols = [x for x in df.columns if x.endswith("dice") or x.startswith("dice_")]
        print(df.groupby("site")[cols].median().round(3).to_string())
        print(df.groupby("site")[cols].mean().round(3).to_string())
    else:
        pts = all_patients(cfg)
        rows = Parallel(n_jobs=a.jobs, verbose=2)(delayed(run_one)(cfg, k, i, a.threshold)
                                                  for k, i in zip(pts.PatientKey, pts.PatientID))
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(AUTO_ROOT, "autoseg_manifest.csv"), index=False)
        print(df.status.value_counts())


if __name__ == "__main__":
    main()
