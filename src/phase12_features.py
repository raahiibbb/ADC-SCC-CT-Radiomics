"""Phase 12 - four-site cohort (LUNG1, Radiogenomics, LPCD, NLST) and features.

The Phase-11 extractors are reused UNCHANGED (same settings), but ALL 1041
patients are re-extracted from the v5 cavity-fenced AUTO_GTV
(C:/LUNG_phase12/auto_masks_v2), so the mask source is identical at every site.

  cohort    : cohort/phase12_four_site_cohort.csv + clin.csv   (.venv-phase10)
  radiomics : PyRadiomics, all sites                           (.venv)
  loc       : tumour position, all sites                       (.venv-phase10)
  fmcib     : FMCIB centroid embedding, all sites              (.venv-phase10)

Resumable.  No label is read by any extractor.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import phase11_features as P  # noqa: E402
from phase12_select import cfg_load  # noqa: E402

CFG = cfg_load()
NIFTI = os.path.join(CFG["root"], "nlst", "nifti")
# env overrides (inherited by joblib workers) let Phase 13 reuse this pipeline
OUT12 = os.environ.get("ADC_FEAT_OUT", os.path.join(ROOT, "global_features", "phase12"))
AUTO2 = os.environ.get("ADC_MASK_ROOT", "C:/LUNG_phase12/auto_masks_v2")
COH = os.environ.get("ADC_COHORT", os.path.join(ROOT, "cohort", "phase12_four_site_cohort.csv"))


def v5_mask_path(key, name="AUTO_GTV"):
    return os.path.join(AUTO2, key.replace("::", "__"), name + ".nii.gz")


def _patch():
    P.mask_path = v5_mask_path
    P.OUT = OUT12


def radiomics_one(key, ct_path):
    _patch()
    return P.radiomics_one(key, ct_path)


def loc_one(key, ct_path):
    _patch()
    return P.loc_one(key, ct_path)


def build_cohort():
    man = pd.read_csv(os.path.join(CFG["root"], "nlst_build_manifest.csv"), dtype={"PatientID": str})
    man = man[man.status == "ok"]
    sel = pd.read_csv(os.path.join(CFG["root"], "nlst_selection.csv"), dtype={"PatientID": str})
    n = sel[sel.PatientID.isin(man.PatientID)]
    nl = pd.DataFrame({"PatientKey": "NLST::" + n.PatientID, "PatientID": n.PatientID.values, "site": "NLST",
                       "label": n.y.astype(int).values,
                       "ct_path": [os.path.join(NIFTI, p, "CT.nii.gz") for p in n.PatientID]})
    c11 = pd.read_csv(os.path.join(ROOT, "cohort", "phase11_three_site_cohort.csv"))
    coh = pd.concat([c11, nl], ignore_index=True)
    coh.to_csv(COH, index=False)
    # clinical: age at the scan (age at randomisation + screening year) and sex
    pr = pd.read_parquet(CFG["nlst"]["clinical_prsn"])
    pr["pid"] = pr["pid"].astype(str)
    pr = pr.set_index("pid").reindex(n.PatientID)
    clin_n = pd.DataFrame({"PatientKey": "NLST::" + n.PatientID.values,
                           "clin__age": pr.age.values + n.tp_num.values,
                           "clin__male": (pr.gender.values == 1).astype(float), "status": "ok"})
    os.makedirs(OUT12, exist_ok=True)
    c11c = pd.read_csv(os.path.join(P.OUT, "clin.csv"))
    pd.concat([c11c, clin_n], ignore_index=True).to_csv(os.path.join(OUT12, "clin.csv"), index=False)
    lesions = man.set_index("PatientID").reindex(n.PatientID)
    pd.DataFrame({"PatientKey": "NLST::" + n.PatientID.values, "n_lesions": lesions.n_lesions.values,
                  "auto_volume_ml": lesions.auto_volume_ml.values,
                  "slice_spacing": lesions.slice_spacing_median.values,
                  "kernel": lesions.ConvolutionKernel.values, "manufacturer": lesions.Manufacturer.values}
                 ).to_csv(os.path.join(OUT12, "nlst_qc.csv"), index=False)
    print(coh.groupby(["site", "label"]).size())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("block", choices=["cohort", "radiomics", "loc", "fmcib"])
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args()
    if a.block == "cohort":
        return build_cohort()
    coh = pd.read_csv(COH)
    man = pd.read_csv(os.path.join(AUTO2, "autoseg_manifest.csv"))
    coh = coh[coh.PatientKey.isin(man.loc[man.status == "ok", "PatientKey"])]
    _patch()
    os.makedirs(OUT12, exist_ok=True)
    prev, todo = P.resumable(a.block, coh.PatientKey)
    paths = dict(zip(coh.PatientKey, coh.ct_path))
    t0 = time.time()
    if a.block == "fmcib":
        rows = P.fmcib_all(todo, [paths[k] for k in todo])
    else:
        fn = radiomics_one if a.block == "radiomics" else loc_one
        rows = Parallel(n_jobs=a.jobs, verbose=2)(delayed(fn)(k, paths[k]) for k in todo)
    print("%s: %d patients in %.0fs" % (a.block, len(rows), time.time() - t0))
    P.save(a.block, prev, rows)


if __name__ == "__main__":
    main()
