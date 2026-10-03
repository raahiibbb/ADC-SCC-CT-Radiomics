"""Phase 14 - frozen MedSAM2 pipeline on the detector boxes (TCGA + detector QA).

Settings are imported unchanged from phase13_medsam2 (checkpoint latest, full
slice, key-slice box, bidirectional propagation).  Also writes the TCGA
cohort file (NO label) used by the feature extractors.
Usage (.venv-medsam2): python src/phase14_segment.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import phase13_medsam2 as M  # noqa: E402

DET = "C:/LUNG_phase14/detect"
OUT = "C:/LUNG_phase14/medsam2_masks"


def main():
    settings = json.load(open(os.path.join(M.OUT_ROOT, "settings.json")))
    assert settings["ckpt"] == M.CKPTS["latest"] and settings["prompt"] == "mid" and settings["crop"] == 0
    pred = M.build("latest")
    man = pd.read_csv(os.path.join(DET, "detect_manifest.csv"))
    rows, qa = [], []
    gt = {}
    for k, ctp, g in __import__("phase14_detect").cases():
        gt[k] = (ctp, g)
    for k, status in zip(man.PatientKey, man.status):
        ctp, g = gt[k]
        r = {"PatientKey": k, "ct_path": ctp}
        if status != "ok":
            r["status"] = "excluded: " + str(status)
            rows.append(r)
            continue
        try:
            img = sitk.ReadImage(ctp, sitk.sitkFloat32)
            box = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(DET, k.replace("::", "__"), "BOX.nii.gz"))) > 0
            m = M.segment(pred, sitk.GetArrayFromImage(img), box, img.GetSpacing(), False, "mid")
            if m.sum() == 0:
                r["status"] = "excluded: empty MedSAM2 mask"
                rows.append(r)
                continue
            d = os.path.join(OUT, k.replace("::", "__"))
            os.makedirs(d, exist_ok=True)
            mi = sitk.GetImageFromArray(m.astype(np.uint8))
            mi.CopyInformation(img)
            sitk.WriteImage(mi, os.path.join(d, "AUTO_GTV.nii.gz"))
            r.update(status="ok", auto_volume_ml=float(m.sum() * np.prod(img.GetSpacing()) / 1000.0))
            if g:
                gg = sitk.Resample(sitk.ReadImage(g), img, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
                gg = sitk.GetArrayFromImage(gg) > 0
                r["dice_vs_gtv"] = M.dice(m, gg)
        except Exception as e:
            r["status"] = "excluded: %s" % str(e)[:200]
        rows.append(r)
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "autoseg_manifest.csv"), index=False)
    q = df[~df.PatientKey.str.startswith("TCGA") & df.dice_vs_gtv.notna()] if "dice_vs_gtv" in df else df.iloc[:0]
    print("QA MedSAM2-on-detector Dice vs expert GTV:",
          q.assign(site=q.PatientKey.str.split("::").str[0]).groupby("site").dice_vs_gtv.agg(["mean", "median", "count"]).round(3).to_dict())
    t = df[df.PatientKey.str.startswith("TCGA")]
    print("TCGA:", t.status.value_counts().to_dict())
    ok = t[t.status == "ok"]
    pd.DataFrame({"PatientKey": ok.PatientKey, "PatientID": ok.PatientKey.str[6:], "site": "TCGA",
                  "label": -1, "ct_path": ok.ct_path}).to_csv(
        os.path.join(ROOT, "cohort", "phase14_tcga_cohort.csv"), index=False)      # label NOT included (-1)


if __name__ == "__main__":
    main()
