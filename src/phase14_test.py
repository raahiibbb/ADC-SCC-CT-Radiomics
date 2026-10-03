"""Phase 14 - frozen model -> locked TCGA test (pre-registered).

  clin    : TCGA age/sex from the DICOM headers -> global_features/phase14/clin.csv
  predict : fit the PRIMARY (shells+loc_clin) and SECONDARY (gtv+shells+loc_clin)
            ONCE on all 1041 development patients (Phase-13 MedSAM2 features,
            ComBat, inner-CV hyper-parameters, seed 42); map TCGA as one new
            site (label-free location/scale); write predictions + sha256.
            NO TCGA label is read.
  score   : the ONLY step that opens tcga_LOCKED_labels.csv.  AUC + bootstrap
            CI, balanced accuracy at 0, TSS-stratified AUC.  Run once.

Usage (.venv-phase10):  python src/phase14_test.py clin|predict|score
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import phase11_cv as C  # noqa: E402
from phase10_data import load_cfg  # noqa: E402

DEV_FEAT = os.path.join(ROOT, "global_features", "phase13")
TST_FEAT = os.path.join(ROOT, "global_features", "phase14")
RES = os.path.join(ROOT, "results", "phase14")
MODELS = {"primary": ("shells", "loc_clin"), "secondary": ("gtv", "shells", "loc_clin")}
SEED = 42


def clin():
    t = pd.read_csv(os.path.join(ROOT, "cohort", "phase14_tcga_cohort.csv"))
    rows = []
    for k, pid in zip(t.PatientKey, t.PatientID):
        r = json.load(open(os.path.join("C:/LUNG_phase12/tcga/nifti", pid, "record.json")))
        age = r.get("PatientAge", "")
        rows.append({"PatientKey": k, "clin__age": float(age[:3]) if age[:3].isdigit() else np.nan,
                     "clin__male": float(r.get("PatientSex") == "M"), "status": "ok"})
    os.makedirs(TST_FEAT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(TST_FEAT, "clin.csv"), index=False)
    print("clin rows", len(rows))


def block(feat_dir, keys, name):
    C.FEAT = feat_dir
    X, cols = C.block_matrix(pd.DataFrame({"PatientKey": keys}), name)
    return X, cols


def predict():
    cfg = load_cfg()
    dev = pd.read_csv(os.path.join(ROOT, "cohort", "phase12_four_site_cohort.csv"))
    tst = pd.read_csv(os.path.join(ROOT, "cohort", "phase14_tcga_cohort.csv"))
    assert (tst.label == -1).all(), "test cohort must not carry labels"
    y, site = dev.label.values.astype(int), dev.site.values
    out = pd.DataFrame({"PatientKey": tst.PatientKey})
    chosen = {}
    for b in sorted({b for v in MODELS.values() for b in v}):
        Xd, cd = block(DEV_FEAT, dev.PatientKey, b)
        Xt, ct = block(TST_FEAT, tst.PatientKey, b)
        assert cd == ct, f"feature mismatch in {b}"
        pr, fitted = C.select_and_fit(cfg, Xd, y, site, SEED, "combat")
        Xr = Xt[:, pr.keep_]
        Xr = np.where(np.isnan(Xr), pr.med_, Xr)[:, pr.keep2_]
        pr.h_.fit_new_site(Xr, "TCGA", balanced=False)
        Z = (pr.h_.transform(Xr, np.array(["TCGA"] * len(Xr))) - pr.mu_) / pr.sd_
        s = C.ens(fitted, Z)
        out["score_" + b] = s
        out["z_" + b] = (s - s.mean()) / s.std()
        chosen[b] = {k: v[4] for k, v in fitted.items()}
        print(b, "fitted; n features", Xd.shape[1], flush=True)
    for m, bl in MODELS.items():
        out[m] = np.mean([out["z_" + b] for b in bl], axis=0)
    os.makedirs(RES, exist_ok=True)
    fn = os.path.join(RES, "tcga_predictions.csv")
    out.to_csv(fn, index=False)
    h = hashlib.sha256(open(fn, "rb").read()).hexdigest()
    json.dump(dict(sha256=h, n=len(out), models=MODELS, seed=SEED, chosen_hyperparameters=chosen),
              open(os.path.join(RES, "tcga_predictions_frozen.json"), "w"), indent=1, default=str)
    print("predictions frozen:", fn, h)


def boot_ci(y, s, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    a = [roc_auc_score(np.r_[np.ones(len(i1)), np.zeros(len(i0))],
                       np.r_[s[rng.choice(i1, len(i1))], s[rng.choice(i0, len(i0))]]) for _ in range(n)]
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def score():
    fn = os.path.join(RES, "tcga_predictions.csv")
    fz = json.load(open(os.path.join(RES, "tcga_predictions_frozen.json")))
    assert hashlib.sha256(open(fn, "rb").read()).hexdigest() == fz["sha256"], "predictions changed after freezing"
    if os.path.isfile(os.path.join(RES, "tcga_result.json")):
        raise SystemExit("already scored once - refusing to rescore (pre-registered single test)")
    p = pd.read_csv(fn)
    lab = pd.read_csv("C:/LUNG_phase12/tcga_LOCKED_labels.csv")          # first and only read
    lab["y"] = (lab.collection_id == "tcga_luad").astype(int)
    lab["PatientKey"] = "TCGA::" + lab.PatientID
    d = p.merge(lab[["PatientKey", "y"]], on="PatientKey")
    d["tss"] = d.PatientKey.str.split("-").str[1]
    res = {"n_scored": int(len(d)), "n_ADC": int(d.y.sum()), "n_SCC": int((d.y == 0).sum()),
           "n_selected_phase12": int(len(lab)), "sha256_predictions": fz["sha256"]}
    for m in ("primary", "secondary"):
        s, y = d[m].values, d.y.values
        ci = boot_ci(y, s)
        yh = (s > 0).astype(int)
        se, sp = float((yh[y == 1] == 1).mean()), float((yh[y == 0] == 0).mean())
        tss = [(t, len(g), roc_auc_score(g.y, g[m])) for t, g in d.groupby("tss") if g.y.nunique() == 2]
        res[m] = dict(auc=float(roc_auc_score(y, s)), ci95=ci, sensitivity=se, specificity=sp,
                      balanced_accuracy=(se + sp) / 2,
                      tss_stratified_auc=float(np.mean([a for _, _, a in tss])) if tss else None,
                      tss_used=[dict(tss=t, n=n, auc=a) for t, n, a in tss],
                      n_tss_total=int(d.tss.nunique()))
    json.dump(res, open(os.path.join(RES, "tcga_result.json"), "w"), indent=1)
    d.to_csv(os.path.join(RES, "tcga_scored.csv"), index=False)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["clin", "predict", "score"])
    a = ap.parse_args()
    {"clin": clin, "predict": predict, "score": score}[a.step]()
