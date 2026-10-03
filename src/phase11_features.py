"""Phase 11 - features for the three-hospital cohort, ALL from AUTO_GTV.

  cohort   : write cohort/phase11_three_site_cohort.csv (+ clinical age/sex)
  radiomics: PyRadiomics on 2 mm grid - regions gtv, inner ring, 3 outer
             shells; ORIGINAL image only.            (run in .venv)
  loc      : tumour-position features (Phase-10 definitions)   (.venv-phase10)
  fmcib    : FMCIB 4096-d at the AUTO_GTV centroid             (.venv-phase10)

Resumable per block.  No label is read by any extractor.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml
from joblib import Parallel, delayed
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

CFG = yaml.safe_load(open(os.path.join(ROOT, "config", "phase11.yaml"), encoding="utf-8"))
AUTO = CFG["autoseg"]["auto_root"]
OUT = os.path.join(ROOT, *CFG["features"]["out_dir"].split("/"))
SP = CFG["features"]["spacing_mm"]


def manifest():
    m = pd.read_csv(os.path.join(AUTO, "autoseg_manifest.csv"))
    return m[m.status == "ok"].reset_index(drop=True)


def mask_path(key, name="AUTO_GTV"):
    return os.path.join(AUTO, key.replace("::", "__"), name + ".nii.gz")


def on_grid(ct_path, key):
    """2 mm isotropic CT (linear) + AUTO_GTV (nearest) on the same grid."""
    from phase10_extract_loc import resample
    ct = resample(sitk.ReadImage(ct_path, sitk.sitkFloat32), sitk.sitkLinear, -1024.0)
    g = sitk.Resample(sitk.ReadImage(mask_path(key), sitk.sitkUInt8), ct, sitk.Transform(),
                      sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return ct, g


def resumable(name, keys):
    f = os.path.join(OUT, name + ".csv")
    if not os.path.isfile(f):
        return None, list(keys)
    prev = pd.read_csv(f)
    done = set(prev.loc[prev.status == "ok", "PatientKey"])
    return prev, [k for k in keys if k not in done]


def save(name, prev, rows):
    new = pd.DataFrame(rows)
    if prev is not None and len(new):
        new = pd.concat([prev[~prev.PatientKey.isin(new.PatientKey)], new], ignore_index=True)
    elif prev is not None:
        new = prev
    os.makedirs(OUT, exist_ok=True)
    new.to_csv(os.path.join(OUT, name + ".csv"), index=False)
    print("%s: %d rows, failures %d" % (name, len(new), int((new.status != "ok").sum())))


# ------------------------------------------------------------------ cohort
def build_cohort():
    m = manifest()
    c10 = pd.read_csv(os.path.join(ROOT, "cohort", "phase8b_multicenter_cohort.csv"))
    dev = c10[["PatientKey", "PatientID", "Cohort", "label"]].rename(columns={"Cohort": "site"})
    sel = pd.read_csv("C:/LUNG_phase11/lung_pet_ct_dx/series_selection.csv")
    lp = sel[sel.selectable][["PatientID", "short_id", "label"]].copy()
    lp["PatientKey"] = "LPCD::" + lp.PatientID
    lp["site"] = "LPCD"
    coh = pd.concat([dev, lp[["PatientKey", "PatientID", "site", "label"]]], ignore_index=True)
    coh = coh[coh.PatientKey.isin(m.PatientKey)].reset_index(drop=True)
    # clinical: age + sex, all three sites
    l1 = pd.read_csv("E:/Thesis/Datasets/L.U.N.G/1/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv")
    rg = pd.read_csv("E:/Thesis/Datasets/L.U.N.G/2/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv")
    lx = pd.read_excel(CFG["lpcd"]["clinical_xlsx"])
    clin = pd.concat([
        pd.DataFrame({"PatientKey": "LUNG1::" + l1.PatientID, "clin__age": l1.age,
                      "clin__male": (l1.gender.str.lower() == "male").astype(float)}),
        pd.DataFrame({"PatientKey": "RADIOGENOMICS::" + rg["Case ID"],
                      "clin__age": pd.to_numeric(rg["Age at Histological Diagnosis"], errors="coerce"),
                      "clin__male": (rg.Gender.str.lower() == "male").astype(float)}),
        pd.DataFrame({"PatientKey": "LPCD::Lung_Dx-" + lx.NewPatientID.astype(str),
                      "clin__age": pd.to_numeric(lx.Age, errors="coerce"),
                      "clin__male": (lx.Sex.astype(str).str.upper() == "M").astype(float)}),
    ]).drop_duplicates("PatientKey")
    os.makedirs(OUT, exist_ok=True)
    clin[clin.PatientKey.isin(coh.PatientKey)].assign(status="ok").to_csv(os.path.join(OUT, "clin.csv"), index=False)
    coh = coh.merge(m[["PatientKey", "ct_path"]], on="PatientKey")
    coh.to_csv(os.path.join(ROOT, "cohort", "phase11_three_site_cohort.csv"), index=False)
    print(coh.groupby(["site", "label"]).size())


# --------------------------------------------------------------- radiomics
def regions(g):
    ga = sitk.GetArrayFromImage(g) > 0
    inside = ndi.distance_transform_edt(ga) * SP
    outside = ndi.distance_transform_edt(~ga) * SP
    r = {"gtv": ga, "ring_in": ga & (inside <= CFG["features"]["inner_ring_mm"])}
    for lo, hi in CFG["features"]["outer_shells_mm"]:
        r["shell%02d_%02d" % (lo, hi)] = (outside > lo) & (outside <= hi)
    return r


def radiomics_one(key, ct_path):
    from radiomics import featureextractor
    from phase10_extract_loc import segment_lungs
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    fc = {"firstorder": [], "glcm": [], "glrlm": [], "glszm": [], "gldm": [], "ngtdm": []}
    base = {"resegmentRange": CFG["features"]["resegment_range"], "resegmentMode": "absolute",
            "binWidth": CFG["features"]["bin_width"], "label": 1, "preCrop": True, "padDistance": 2,
            "geometryTolerance": 1e-4, "correctMask": False, "normalize": False,
            "minimumROIDimensions": 2, "distances": [1], "symmetricalGLCM": True}
    try:
        ct, g = on_grid(ct_path, key)
        _, body = segment_lungs(sitk.GetArrayFromImage(ct))
        row = {"PatientKey": key, "status": "ok"}
        for name, m in regions(g).items():
            if name.startswith("shell"):
                m = m & body
            classes = dict(fc, shape=[]) if name == "gtv" else fc
            ext = featureextractor.RadiomicsFeatureExtractor(
                {"imageType": {"Original": {}}, "featureClass": classes, "setting": base})
            mi = sitk.GetImageFromArray(m.astype(np.uint8))
            mi.CopyInformation(ct)
            res = ext.execute(ct, mi, label=1)
            for k, v in res.items():
                if not k.startswith("diagnostics"):
                    row["rad__%s__%s" % (name, k)] = float(v)
        return row
    except Exception as e:
        return {"PatientKey": key, "status": "failed: %s" % str(e)[:200]}


# -------------------------------------------------------------------- loc
def loc_one(key, ct_path):
    import phase10_extract_loc as L
    try:
        orig = L.paths
        L.paths = lambda cfg, k, pid: (ct_path, mask_path(key))
        r = L.features(None, key, None)
        L.paths = orig
        return r
    except Exception as e:
        return {"PatientKey": key, "status": "failed: %s" % str(e)[:200]}


# ------------------------------------------------------------------ fmcib
def fmcib_all(keys, paths):
    import torch
    import phase10_extract_fmcib as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = F.load_model(dev)
    rows = []
    for k, ctp in zip(keys, paths):
        try:
            ct = sitk.ReadImage(ctp, sitk.sitkFloat32)
            st = sitk.LabelShapeStatisticsImageFilter()
            st.Execute(sitk.ReadImage(mask_path(k), sitk.sitkUInt8) > 0)
            seed = np.array(st.GetCentroid(1))
            ref = sitk.Image([F.SIZE] * 3, sitk.sitkFloat32)
            ref.SetSpacing((1.0, 1.0, 1.0))
            ref.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
            ref.SetOrigin(tuple(seed - F.SIZE // 2))
            x = (sitk.GetArrayFromImage(sitk.Resample(ct, ref, sitk.Transform(), sitk.sitkLinear,
                                                      -1024.0, sitk.sitkFloat32)) + 1024.0) / 3072.0
            with torch.no_grad():
                f = model(torch.from_numpy(x[None, None]).float().to(dev)).cpu().numpy()[0]
            row = {"PatientKey": k, "status": "ok"}
            row.update({"fmcib__f%04d" % j: float(v) for j, v in enumerate(f)})
        except Exception as e:
            row = {"PatientKey": k, "status": "failed: %s" % str(e)[:200]}
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("block", choices=["cohort", "radiomics", "loc", "fmcib"])
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args()
    if a.block == "cohort":
        return build_cohort()
    coh = pd.read_csv(os.path.join(ROOT, "cohort", "phase11_three_site_cohort.csv"))
    prev, todo = resumable(a.block, coh.PatientKey)
    paths = dict(zip(coh.PatientKey, coh.ct_path))
    t0 = time.time()
    if a.block == "fmcib":
        rows = fmcib_all(todo, [paths[k] for k in todo])
    else:
        fn = radiomics_one if a.block == "radiomics" else loc_one
        rows = Parallel(n_jobs=a.jobs, verbose=2)(delayed(fn)(k, paths[k]) for k in todo)
    print("%s: %d patients in %.0fs" % (a.block, len(rows), time.time() - t0))
    save(a.block, prev, rows)


if __name__ == "__main__":
    main()
