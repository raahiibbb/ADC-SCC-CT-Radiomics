"""Phase 12 - four-site evaluation (LUNG1, Radiogenomics, LPCD, NLST).

Identical machinery to phase11_cv (pooled 5x5 CV with site-balanced AUC,
leave-one-site-out, block fusion); only the cohort, feature directory,
result directory and site list change.  The PRIMARY model is frozen from
Phase 11 (shells + loc_clin, ComBat) and was fixed before NLST was seen.

Usage:
  python src/phase12_cv.py pooled --blocks gtv ring_in shells loc_clin fmcib
  python src/phase12_cv.py loso   --blocks ...
  python src/phase12_cv.py fusion --blocks ...
  add --subset single_lesion to restrict NLST to patients with one annotated lesion
  add --subset three_site for the ablation "Phase-11 sites with v5 masks"
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import phase11_cv as C  # noqa: E402
from phase10_data import load_cfg  # noqa: E402

C.FEAT = os.environ.get("ADC_FEAT_OUT", os.path.join(ROOT, "global_features", "phase12"))
C.RES = os.environ.get("ADC_RES_OUT", os.path.join(ROOT, "results", "phase12"))
C.SITES = ("LUNG1", "RADIOGENOMICS", "LPCD", "NLST")
BLOCKS = ["gtv", "ring_in", "shells", "loc_clin", "fmcib"]
# Phase 13: CT-FM blocks (global crop average; tumour-weighted pooling)
C.BLOCKS.update({"ctfm": ("ctfm", "ctfm__"), "ctfmt": ("ctfm", "ctfmt__")})
PRIMARY = "shells+loc_clin"


def cohort(subset=None):
    c = pd.read_csv(os.path.join(ROOT, "cohort", "phase12_four_site_cohort.csv"))
    c["y"] = c["label"].astype(int)
    if subset == "single_lesion":
        q = pd.read_csv(os.path.join(C.FEAT, "nlst_qc.csv"))
        multi = set(q.loc[q.n_lesions > 1, "PatientKey"])
        c = c[~c.PatientKey.isin(multi)].reset_index(drop=True)
    elif subset == "three_site":           # ablation: Phase-11 sites, v5 masks
        c = c[c.site != "NLST"].reset_index(drop=True)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["pooled", "loso", "fusion"])
    ap.add_argument("--blocks", nargs="*", default=BLOCKS)
    ap.add_argument("--harm", nargs="*", default=["combat"])
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--subset", default=None, choices=[None, "single_lesion", "three_site"])
    a = ap.parse_args()
    if a.subset:
        C.RES = os.path.join(C.RES, "subset_" + a.subset)
    if a.subset == "three_site":
        C.SITES = ("LUNG1", "RADIOGENOMICS", "LPCD")
    cfg = load_cfg()
    coh = cohort(a.subset)
    print(coh.groupby(["site", "y"]).size().unstack().to_string(), flush=True)
    if a.step == "pooled":
        C.run_pooled(cfg, coh, a.blocks, a.harm, a.jobs)
    elif a.step == "loso":
        C.run_loso(cfg, coh, a.blocks, a.harm, a.jobs)
    else:
        for h in a.harm:
            C.run_fusion(cfg, a.blocks, h)


if __name__ == "__main__":
    main()
