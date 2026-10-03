"""Phase 14B - second locked external test on Lung3 (NSCLC-Radiomics-Genomics).

Pre-registered in docs/PHASE14B_LUNG3_PREREGISTRATION.md.  Frozen components
reused unchanged: phase12_build.read_series (DICOM -> NIfTI), TotalSegmentator
lung_nodules (largest component -> boxes), phase13 MedSAM2 settings, phase12
extractors, phase14_test model fitting.  Labels are joined ONLY in `score`,
after the predictions are written and hashed.

Steps / environments:
  convert  (.venv-phase10)   DICOM -> C:/LUNG_phase15/nifti/<pid>/CT.nii.gz
  detect   (.venv-totalseg)  TotalSegmentator + BOX
  segment  (.venv-medsam2)   MedSAM2 -> C:/LUNG_phase15/medsam2_masks + cohort file (no label)
  clin | predict | score (.venv-phase10)
Features in between:  ADC_COHORT/ADC_MASK_ROOT/ADC_FEAT_OUT + phase12_features.py radiomics|loc
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

SRC = "C:/LUNG_phase15/lung3/nsclc_radiomics_genomics"
BASE = "C:/LUNG_phase15"
NIFTI, DET, MASKS = BASE + "/nifti", BASE + "/detect", BASE + "/medsam2_masks"
COHORT = os.path.join(ROOT, "cohort", "phase14b_lung3_cohort.csv")
FEAT = os.path.join(ROOT, "global_features", "phase14b")
RES = os.path.join(ROOT, "results", "phase14b")
CLIN_XLS = BASE + "/meta/Lung3.metadata.xls"


def convert():
    import SimpleITK as sitk
    from phase12_build import read_series
    rows = []
    for pid in sorted(os.listdir(SRC)):
        d = os.path.join(NIFTI, pid)
        rec_p = os.path.join(d, "record.json")
        if os.path.isfile(rec_p) and json.load(open(rec_p)).get("status") == "ok":
            rows.append(json.load(open(rec_p)))
            continue
        os.makedirs(d, exist_ok=True)
        series = [s for s in glob.glob(os.path.join(SRC, pid, "*", "*")) if os.path.isdir(s)]
        series.sort(key=lambda s: -len(os.listdir(s)))            # most images
        try:
            if len(os.listdir(series[0])) < 40:
                raise RuntimeError("fewer than 40 images")
            img, _, info = read_series(series[0])
            sitk.WriteImage(img, os.path.join(d, "CT.nii.gz"))
            rec = dict(status="ok", PatientID=pid, series_dir=series[0], **info)
        except Exception as e:
            rec = dict(status="failed", PatientID=pid, error=str(e)[:300])
        json.dump(rec, open(rec_p, "w"), indent=1)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(BASE, "convert_manifest.csv"), index=False)
    print(df.status.value_counts().to_dict(), "| slice spacing:", df.slice_spacing_median.round(1).value_counts().to_dict())


def detect():
    import SimpleITK as sitk
    from scipy import ndimage as ndi
    from totalsegmentator.python_api import totalsegmentator
    man = pd.read_csv(os.path.join(BASE, "convert_manifest.csv"))
    rows, t0 = [], time.time()
    for i, pid in enumerate(man.loc[man.status == "ok", "PatientID"]):
        ctp = os.path.join(NIFTI, pid, "CT.nii.gz")
        d = os.path.join(DET, pid)
        r = {"PatientID": pid}
        try:
            if not os.path.isfile(os.path.join(d, "lung_nodules.nii.gz")):
                totalsegmentator(ctp, d, task="lung_nodules", quiet=True)
            ct = sitk.ReadImage(ctp)
            n = sitk.Resample(sitk.ReadImage(os.path.join(d, "lung_nodules.nii.gz")), ct, sitk.Transform(),
                              sitk.sitkNearestNeighbor, 0)
            a = sitk.GetArrayFromImage(n) > 0
            lab, cnt = ndi.label(a)
            r["n_components"] = int(cnt)
            if cnt == 0:
                r["status"] = "no nodule detected"
            else:
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
        except Exception as e:
            r["status"] = "failed: %s" % str(e)[:200]
        rows.append(r)
        print(f"{i + 1} {pid} {r['status']} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DET, "detect_manifest.csv"), index=False)
    print(df.status.value_counts().to_dict())


def segment():
    import SimpleITK as sitk
    import phase13_medsam2 as M
    st = json.load(open(os.path.join(M.OUT_ROOT, "settings.json")))
    assert st["ckpt"] == M.CKPTS["latest"] and st["prompt"] == "mid" and st["crop"] == 0
    pred = M.build("latest")
    det = pd.read_csv(os.path.join(DET, "detect_manifest.csv"))
    rows = []
    for pid, status in zip(det.PatientID, det.status):
        k, ctp = "LUNG3::" + pid, os.path.join(NIFTI, pid, "CT.nii.gz")
        r = {"PatientKey": k, "PatientID": pid, "ct_path": ctp}
        if status != "ok":
            r["status"] = "excluded: " + str(status)
            rows.append(r)
            continue
        try:
            img = sitk.ReadImage(ctp, sitk.sitkFloat32)
            box = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(DET, pid, "BOX.nii.gz"))) > 0
            m = M.segment(pred, sitk.GetArrayFromImage(img), box, img.GetSpacing(), False, "mid")
            if m.sum() == 0:
                raise RuntimeError("empty MedSAM2 mask")
            d = os.path.join(MASKS, k.replace("::", "__"))
            os.makedirs(d, exist_ok=True)
            mi = sitk.GetImageFromArray(m.astype(np.uint8))
            mi.CopyInformation(img)
            sitk.WriteImage(mi, os.path.join(d, "AUTO_GTV.nii.gz"))
            r.update(status="ok", auto_volume_ml=float(m.sum() * np.prod(img.GetSpacing()) / 1000.0))
        except Exception as e:
            r["status"] = "excluded: %s" % str(e)[:200]
        rows.append(r)
    df = pd.DataFrame(rows)
    os.makedirs(MASKS, exist_ok=True)
    df.to_csv(os.path.join(MASKS, "autoseg_manifest.csv"), index=False)
    ok = df[df.status == "ok"]
    pd.DataFrame({"PatientKey": ok.PatientKey, "PatientID": ok.PatientID, "site": "LUNG3", "label": -1,
                  "ct_path": ok.ct_path}).to_csv(COHORT, index=False)
    print(df.status.value_counts().to_dict())


def clin():
    t = pd.read_csv(COHORT)
    rows = []
    for k, pid in zip(t.PatientKey, t.PatientID):
        r = json.load(open(os.path.join(NIFTI, pid, "record.json")))
        age = r.get("PatientAge", "")
        rows.append({"PatientKey": k, "clin__age": float(age[:3]) if age[:3].isdigit() else np.nan,
                     "clin__male": float(r.get("PatientSex") == "M"), "status": "ok"})
    os.makedirs(FEAT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(FEAT, "clin.csv"), index=False)
    print("clin rows", len(rows), "| age missing", int(pd.DataFrame(rows).clin__age.isna().sum()))


def predict():
    import phase14_test as T
    T.TST_FEAT, T.RES = FEAT, RES
    # reuse the frozen fitting code with the Lung3 cohort file
    orig = pd.read_csv

    def rc(path, *a, **k):
        if str(path).endswith("phase14_tcga_cohort.csv"):
            path = COHORT
        return orig(path, *a, **k)
    pd.read_csv = rc
    try:
        T.predict()
    finally:
        pd.read_csv = orig


def label_map():
    l3 = pd.read_excel(CLIN_XLS)
    h = l3["characteristics.tag.histology"].astype(str).str.replace("\xa0", " ")
    y = pd.Series(np.nan, index=l3.index)
    adc = h.str.contains("Adenocarcinoma") | (h == "Solid Type And Acinar")
    scc = h.str.startswith("Squamous Cell Carcinoma") & ~h.str.contains("with adeno features")
    y[adc & ~scc] = 1
    y[scc] = 0
    return pd.DataFrame({"PatientKey": "LUNG3::" + l3["sample.name"], "y": y, "histology": h})


def score():
    from phase14_test import boot_ci
    from sklearn.metrics import roc_auc_score
    fn = os.path.join(RES, "tcga_predictions.csv")
    fz = json.load(open(os.path.join(RES, "tcga_predictions_frozen.json")))
    assert hashlib.sha256(open(fn, "rb").read()).hexdigest() == fz["sha256"], "predictions changed after freezing"
    out_fn = os.path.join(RES, "lung3_result.json")
    if os.path.isfile(out_fn):
        raise SystemExit("already scored once - refusing to rescore")
    p = pd.read_csv(fn)
    lab = label_map()                                               # first and only link of labels
    d = p.merge(lab, on="PatientKey", how="left")
    n_pred = len(d)
    d = d[d.y.notna()].copy()
    man = pd.read_csv(os.path.join(BASE, "convert_manifest.csv")).set_index("PatientID")
    d["spacing"] = d.PatientKey.str[7:].map(man.slice_spacing_median)
    res = dict(n_predicted=int(n_pred), n_scored=int(len(d)), n_ADC=int(d.y.sum()), n_SCC=int((d.y == 0).sum()),
               n_label_excluded=int(n_pred - len(d)), sha256_predictions=fz["sha256"])
    for m in ("primary", "secondary"):
        s, y = d[m].values, d.y.values.astype(int)
        yh = (s > 0).astype(int)
        se, sp = float((yh[y == 1] == 1).mean()), float((yh[y == 0] == 0).mean())
        r = dict(auc=float(roc_auc_score(y, s)), ci95=boot_ci(y, s), sensitivity=se, specificity=sp,
                 balanced_accuracy=(se + sp) / 2)
        for nm, msk in (("thin_le_2p5mm", d.spacing <= 2.5), ("thick_gt_2p5mm", d.spacing > 2.5)):
            g = d[msk]
            r[nm] = dict(n=int(len(g)), n_ADC=int(g.y.sum()),
                         auc=float(roc_auc_score(g.y, g[m])) if g.y.nunique() == 2 else None)
        res[m] = r
    json.dump(res, open(out_fn, "w"), indent=1)
    d.to_csv(os.path.join(RES, "lung3_scored.csv"), index=False)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["convert", "detect", "segment", "clin", "predict", "score"])
    a = ap.parse_args()
    globals()[a.step]()
