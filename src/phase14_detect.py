"""Phase 14 - automatic tumour localisation for the locked TCGA test.

Pre-registered in docs/PHASE14_PREREGISTRATION.md.
TotalSegmentator `lung_nodules` (Apache-2.0) -> largest 3-D component ->
per-slice boxes (BOX.nii.gz, input to the frozen MedSAM2 pipeline).
Also runs the detector QA on 60 development patients with expert GTVs
(no histology read).  No TCGA label is read.  Resumable.

Usage (.venv-totalseg):
  python src/phase14_detect.py detect      # TCGA 85 + QA 60
  python src/phase14_detect.py boxes       # largest component -> BOX, QA metrics
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = "C:/LUNG_phase14/detect"


def cases():
    t = pd.read_csv("C:/LUNG_phase12/tcga_selection.csv")          # no label column
    rows = [("TCGA::" + p, os.path.join("C:/LUNG_phase12/tcga/nifti", p, "CT.nii.gz"), None) for p in t.PatientID]
    c = pd.read_csv(os.path.join(ROOT, "cohort", "phase8b_multicenter_cohort.csv"))
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "phase10.yaml"), encoding="utf-8"))["paths"]
    rng = np.random.RandomState(0)
    for pre in ("LUNG1", "RADIOGENOMICS"):
        sub = c[c.PatientKey.str.startswith(pre)]
        sub = sub.iloc[rng.choice(len(sub), 30, replace=False)]
        root = cfg["lung1_image_root"] if pre == "LUNG1" else cfg["external_image_root"]
        root = root if os.path.isabs(root) else os.path.join(ROOT, *root.split("/"))
        for k, pid in zip(sub.PatientKey, sub.PatientID):
            rows.append((k, os.path.join(root, pid, "CT.nii.gz"), os.path.join(root, pid, "ROI_A_GTV.nii.gz")))
    return rows


def detect():
    from totalsegmentator.python_api import totalsegmentator
    t0 = time.time()
    for i, (k, ctp, _) in enumerate(cases()):
        d = os.path.join(OUT, k.replace("::", "__"))
        if os.path.isfile(os.path.join(d, "lung_nodules.nii.gz")):
            continue
        try:
            totalsegmentator(ctp, d, task="lung_nodules", quiet=True)
        except Exception as e:
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "ERROR.txt"), "w").write(str(e)[:1000])
        print(f"{i + 1} {k} {time.time() - t0:.0f}s", flush=True)


def boxes():
    rows = []
    for k, ctp, gtv in cases():
        d = os.path.join(OUT, k.replace("::", "__"))
        r = {"PatientKey": k}
        try:
            n = sitk.ReadImage(os.path.join(d, "lung_nodules.nii.gz"))
            ct = sitk.ReadImage(ctp)
            n = sitk.Resample(n, ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
            a = sitk.GetArrayFromImage(n) > 0
            lab, cnt = ndi.label(a)
            r["n_components"] = int(cnt)
            if cnt == 0:
                r["status"] = "no nodule detected"
                rows.append(r)
                continue
            vol = np.bincount(lab.ravel())[1:]
            big = lab == (np.argmax(vol) + 1)
            box = np.zeros_like(big)
            for z in np.where(big.reshape(big.shape[0], -1).any(1))[0]:
                ys, xs = np.nonzero(big[z])
                box[z, ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
            bi = sitk.GetImageFromArray(box.astype(np.uint8))
            bi.CopyInformation(ct)
            sitk.WriteImage(bi, os.path.join(d, "BOX.nii.gz"))
            r.update(status="ok", largest_ml=float(vol.max() * np.prod(ct.GetSpacing()) / 1000.0))
            if gtv:
                g = sitk.Resample(sitk.ReadImage(gtv), ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
                g = sitk.GetArrayFromImage(g) > 0
                r.update(hit=bool((big & g).any()), dice_largest=float(2 * (big & g).sum() / (big.sum() + g.sum())),
                         any_hit=bool((a & g).any()))
        except Exception as e:
            r["status"] = "failed: %s" % str(e)[:200]
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "detect_manifest.csv"), index=False)
    qa = df[~df.PatientKey.str.startswith("TCGA")]
    qa = qa.assign(site=qa.PatientKey.str.split("::").str[0])
    print("QA (60 dev, expert GTV):")
    print(qa.groupby("site")[["hit", "any_hit", "dice_largest"]].mean().round(3).to_string())
    tc = df[df.PatientKey.str.startswith("TCGA")]
    print("TCGA:", tc.status.value_counts().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["detect", "boxes"])
    a = ap.parse_args()
    detect() if a.step == "detect" else boxes()
