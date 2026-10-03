"""Phase 7A - the nested cross-validation driver for A0 / A1 / A2 / A3.

    outer: the FROZEN 5-fold patient-level split (never regenerated)
    inner: stratified 3-fold patient-level CV inside each outer-training fold,
           seed 3000 + outer_fold  (the established project convention)

Everything below is fitted on training patients only, at whatever level it is
used: imputation, categorical encoding, variance filtering, scaling, feature
selection, elastic-net hyperparameters, mRMR, fusion fitting, calibration and
threshold selection.  No outer-test patient influences any choice, and every
outer-test fold is scored exactly once.

A3 is LOW-CAPACITY LATE FUSION with exactly two inputs.  Its stacker is fitted
only on INNER-OOF component predictions, so no component model ever contributes
an in-sample prediction about a patient it was trained on.

    python src/phase7a_cv.py --config config/phase7a.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7a_extract import extraction_signature, load_cfg, load_cohort, ppath   # noqa: E402
from phase7a_models import (CappedElasticNet, ClinicalEncoder, PlattCalibrator,  # noqa: E402
                            TrainingOnlyPreprocessor, fit_logistic, mrmr_select,
                            select_threshold)

PRIMARY = ("A0", "A1", "A2", "A3")
PRED_FILE = {"A0": "A0_oof.csv", "A1": "A1_gtv_oof.csv",
             "A2": "A2_rim_oof.csv", "A3": "A3_fusion_oof.csv"}


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def log(msg, fh=None):
    print(msg, flush=True)
    if fh:
        fh.write(msg + "\n")
        fh.flush()


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def load_feature_matrix(cfg, region):
    path = ppath(cfg, "gtv_csv" if region == "gtv" else "rim_csv")
    df = pd.read_csv(path)
    meta = ["PatientID", "Histology", "label"]
    names = [c for c in df.columns if c not in meta
             and not any(c.startswith(p) for p in cfg["radiomics"]["drop_column_prefixes"])]
    x = df[names].to_numpy(dtype=np.float64)
    return {"pids": list(df.PatientID), "names": names, "x": x,
            "y": df.label.to_numpy(dtype=int),
            "index": {p: i for i, p in enumerate(df.PatientID)},
            "sha256": sha256(path), "path": path}


def load_clinical(cfg, pids):
    df = pd.read_csv(cfg["paths"]["dataset_metadata"], low_memory=False)
    src = cfg["models"]["A0"]["source_columns"]
    df = df.set_index("PatientID")
    rows = {}
    for p in pids:
        r = df.loc[p]
        rows[p] = {"age": r[src["age"]], "sex": r[src["sex"]], "stage": r[src["stage"]]}
    return rows


def verify_frozen_inputs(cfg, cohort, gtv, rim, fh):
    """The driver refuses to train unless every frozen input checks out."""
    rec = {}
    sp = ppath(cfg, "splits_file")
    rec["splits_file_sha256"] = sha256(sp)
    if rec["splits_file_sha256"] != cfg["paths"]["splits_file_sha256"]:
        raise RuntimeError("the frozen outer split has changed")
    with open(ppath(cfg, "splits_meta_file"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta["splits_file_sha256"] != rec["splits_file_sha256"]:
        raise RuntimeError("split file does not match its own Phase-4 metadata")
    rec["split_meta_n_patients"] = meta["n_patients"]
    rec["cohort"] = {"n": len(cohort), "adc": int((cohort.label == 1).sum()),
                     "scc": int((cohort.label == 0).sum())}
    if rec["cohort"] != {"n": 202, "adc": 51, "scc": 151}:
        raise RuntimeError("cohort is not 202 / 51 ADC / 151 SCC")
    for tag, m in (("gtv", gtv), ("rim", rim)):
        if m["pids"] != list(cohort.PatientID):
            raise RuntimeError("%s feature rows do not match the cohort order" % tag)
        if len(m["pids"]) != len(set(m["pids"])):
            raise RuntimeError("%s has duplicate patient rows" % tag)
        rec["%s_matrix" % tag] = {"rows": len(m["pids"]), "features": len(m["names"]),
                                  "sha256": m["sha256"]}
    with open(ppath(cfg, "feature_names_json"), "r", encoding="utf-8") as f:
        fn = json.load(f)
    if fn["extraction_signature"] != extraction_signature(cfg):
        raise RuntimeError("cached global features carry a stale extraction signature")
    if fn["gtv"] != gtv["names"] or fn["rim"] != rim["names"]:
        raise RuntimeError("feature-name manifest disagrees with the matrices")
    rec["extraction_signature"] = fn["extraction_signature"]
    rec["rim_has_no_shape"] = not any("_shape_" in n for n in rim["names"])
    if not rec["rim_has_no_shape"]:
        raise RuntimeError("rim bank contains shape features, which the protocol forbids")
    rec["gtv_has_shape"] = any("_shape_" in n for n in gtv["names"])
    rec["protocol_sha256"] = sha256(os.path.join(cfg["_project_root"], "docs", "PHASE7_PROTOCOL.md"))
    rec["config_sha256"] = sha256(cfg["_config_path"])
    rec["pyradiomics_params_sha256"] = sha256(ppath(cfg, "pyradiomics_params"))
    log("pre-flight: split %s | gtv %d x %d | rim %d x %d | protocol %s"
        % (rec["splits_file_sha256"][:16], rec["gtv_matrix"]["rows"], rec["gtv_matrix"]["features"],
           rec["rim_matrix"]["rows"], rec["rim_matrix"]["features"], rec["protocol_sha256"][:16]), fh)
    return rec


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------

def outer_folds(cfg, cohort):
    sp = pd.read_csv(ppath(cfg, "splits_file"))
    folds = {}
    for k in sorted(sp.fold.unique()):
        f = sp[sp.fold == k]
        folds[int(k)] = {"train": list(f[f.split == "train"].PatientID),
                         "test": list(f[f.split == "test"].PatientID)}
    allp = set(cohort.PatientID)
    for k, d in folds.items():
        if set(d["train"]) | set(d["test"]) != allp or set(d["train"]) & set(d["test"]):
            raise RuntimeError("outer fold %d is not a clean partition of the cohort" % k)
    return folds


def inner_folds(cfg, train_pids, y_train, outer_fold):
    seed = int(cfg["validation"]["inner"]["base_seed"]) + int(outer_fold)
    skf = StratifiedKFold(n_splits=int(cfg["validation"]["inner"]["n_splits"]),
                          shuffle=bool(cfg["validation"]["inner"]["shuffle"]),
                          random_state=seed)
    idx = np.arange(len(train_pids))
    return [(tr, va) for tr, va in skf.split(idx, y_train)], seed


# ---------------------------------------------------------------------------
# the A1 / A2 procedure (identical; only the region differs)
# ---------------------------------------------------------------------------

def fit_radiomics_model(cfg, x_tr, y_tr, C, l1_ratio):
    pre = TrainingOnlyPreprocessor(
        abs_tol=float(cfg["pipeline"]["variance_filter"]["abs_tol"]),
        rel_tol=float(cfg["pipeline"]["variance_filter"]["rel_tol"]),
        all_nan_fill=float(cfg["pipeline"]["imputer"]["all_nan_fill"])).fit(x_tr)
    mcfg = cfg["models"]["A1"]
    est = CappedElasticNet(C=C, l1_ratio=l1_ratio,
                           max_features=int(mcfg["feature_cap"]["max_features"]),
                           est_cfg=mcfg).fit(pre.transform(x_tr), y_tr)
    return pre, est


def radiomics_fold(cfg, region, mat, fold, tr_pids, te_pids, inner, fh, tag):
    """One outer fold of A1 (region='gtv') or A2 (region='rim')."""
    mcfg = cfg["models"]["A1"]
    grid = [(l1, c) for l1 in mcfg["grid"]["l1_ratio"] for c in mcfg["grid"]["C"]]
    itr = mat["index"]
    x_all, y_all = mat["x"], mat["y"]
    tr_idx = np.asarray([itr[p] for p in tr_pids])
    te_idx = np.asarray([itr[p] for p in te_pids])
    y_tr = y_all[tr_idx]

    surface, inner_oof_cache = [], {}
    t0 = time.time()
    for l1, C in grid:
        oof = np.full(len(tr_pids), np.nan)
        aucs = []
        for j, (a, b) in enumerate(inner):
            pre, est = fit_radiomics_model(cfg, x_all[tr_idx[a]], y_tr[a], C, l1)
            p = est.predict_proba(pre.transform(x_all[tr_idx[b]]))
            oof[b] = p
            aucs.append(_auc(y_tr[b], p))
        surface.append({"l1_ratio": float(l1), "C": float(C),
                        "mean_inner_roc_auc": float(np.mean(aucs)),
                        "sd_inner_roc_auc": float(np.std(aucs, ddof=1)),
                        "inner_roc_auc": [float(v) for v in aucs]})
        inner_oof_cache[(l1, C)] = oof
    # tie-break: smaller C, then smaller l1_ratio, then config order
    best = sorted(range(len(surface)),
                  key=lambda i: (-surface[i]["mean_inner_roc_auc"],
                                 surface[i]["C"], surface[i]["l1_ratio"], i))[0]
    sel = surface[best]
    inner_oof = inner_oof_cache[(sel["l1_ratio"], sel["C"])]

    # refit on ALL outer-training patients with the selected hyperparameters
    pre, est = fit_radiomics_model(cfg, x_all[tr_idx], y_tr, sel["C"], sel["l1_ratio"])
    p_test = est.predict_proba(pre.transform(x_all[te_idx]))

    kept = pre.kept_indices
    chosen_global = [int(kept[j]) for j in est.selected]
    chosen_names = [mat["names"][j] for j in chosen_global]
    log("  %s fold %d: l1=%.2f C=%g  inner AUC %.4f  kept %d/%d dims  selected %d feats  (%.1fs)"
        % (tag, fold, sel["l1_ratio"], sel["C"], sel["mean_inner_roc_auc"],
           int(pre.keep.sum()), pre.n_input, len(chosen_names), time.time() - t0), fh)

    return {"surface": surface, "selected": sel, "inner_oof": inner_oof,
            "p_test": p_test, "pre": pre, "est": est,
            "selected_feature_indices": chosen_global,
            "selected_feature_names": chosen_names,
            "coefficients": [float(v) for v in est.coefficients],
            "intercept": float(est.intercept),
            "n_nonzero_stage1": int(est.n_nonzero_stage1),
            "converged": bool(est.converged_stage1 and est.converged_stage2),
            "seconds": round(time.time() - t0, 1)}


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


# ---------------------------------------------------------------------------
# A0
# ---------------------------------------------------------------------------

def clinical_fold(cfg, clinical, pids_order, fold, tr_pids, te_pids, inner, y_map, fh):
    a0 = cfg["models"]["A0"]
    y_tr = np.asarray([y_map[p] for p in tr_pids], dtype=int)
    surface, cache = [], {}
    for C in a0["C_grid"]:
        oof = np.full(len(tr_pids), np.nan)
        aucs = []
        for a, b in inner:
            enc = ClinicalEncoder().fit([clinical[tr_pids[i]] for i in a])
            xa = enc.transform([clinical[tr_pids[i]] for i in a])
            xb = enc.transform([clinical[tr_pids[i]] for i in b])
            pre = TrainingOnlyPreprocessor(
                abs_tol=float(cfg["pipeline"]["variance_filter"]["abs_tol"]),
                rel_tol=float(cfg["pipeline"]["variance_filter"]["rel_tol"])).fit(xa)
            est = fit_logistic(pre.transform(xa), y_tr[a], a0, C=C)
            p = est.predict_proba(pre.transform(xb))[:, 1]
            oof[b] = p
            aucs.append(_auc(y_tr[b], p))
        surface.append({"C": float(C), "mean_inner_roc_auc": float(np.mean(aucs)),
                        "inner_roc_auc": [float(v) for v in aucs]})
        cache[C] = oof
    best = sorted(range(len(surface)),
                  key=lambda i: (-surface[i]["mean_inner_roc_auc"], surface[i]["C"], i))[0]
    sel = surface[best]
    inner_oof = cache[sel["C"]]

    enc = ClinicalEncoder().fit([clinical[p] for p in tr_pids])
    xa = enc.transform([clinical[p] for p in tr_pids])
    xb = enc.transform([clinical[p] for p in te_pids])
    pre = TrainingOnlyPreprocessor(
        abs_tol=float(cfg["pipeline"]["variance_filter"]["abs_tol"]),
        rel_tol=float(cfg["pipeline"]["variance_filter"]["rel_tol"])).fit(xa)
    est = fit_logistic(pre.transform(xa), y_tr, a0, C=sel["C"])
    p_test = est.predict_proba(pre.transform(xb))[:, 1]
    log("  A0 fold %d: C=%g  inner AUC %.4f  design %d cols (%d kept)"
        % (fold, sel["C"], sel["mean_inner_roc_auc"], len(enc.columns), int(pre.keep.sum())), fh)
    return {"surface": surface, "selected": sel, "inner_oof": inner_oof, "p_test": p_test,
            "encoder_columns": enc.columns, "categories": enc.categories,
            "medians": enc.medians, "modes": enc.modes,
            "coefficients": [float(v) for v in est.coef_[0]],
            "intercept": float(est.intercept_[0]),
            "kept_columns": [enc.columns[i] for i in pre.kept_indices]}


# ---------------------------------------------------------------------------
# A3 - late fusion on inner-OOF component predictions only
# ---------------------------------------------------------------------------

def fusion_fold(cfg, a1, a2, y_tr, inner, fh, fold):
    fcfg = cfg["models"]["A3"]
    z_tr = np.column_stack([a1["inner_oof"], a2["inner_oof"]])
    if not np.all(np.isfinite(z_tr)):
        raise RuntimeError("fold %d: a component inner-OOF prediction is missing" % fold)

    # A3's own inner-OOF, from the SAME inner assignment over the fusion inputs
    oof = np.full(len(y_tr), np.nan)
    for a, b in inner:
        st = fit_logistic(z_tr[a], y_tr[a], fcfg)
        oof[b] = st.predict_proba(z_tr[b])[:, 1]

    stacker = fit_logistic(z_tr, y_tr, fcfg)
    z_te = np.column_stack([a1["p_test"], a2["p_test"]])
    p_test = stacker.predict_proba(z_te)[:, 1]
    log("  A3 fold %d: stacker coef [%.4f, %.4f] intercept %.4f (fitted on %d inner-OOF rows)"
        % (fold, stacker.coef_[0][0], stacker.coef_[0][1], stacker.intercept_[0], len(y_tr)), fh)
    return {"inner_oof": oof, "p_test": p_test,
            "coefficients": [float(v) for v in stacker.coef_[0]],
            "intercept": float(stacker.intercept_[0]),
            "fusion_inputs": ["P_ADC_GTV", "P_ADC_RIM"],
            "n_fusion_training_rows": int(len(y_tr)),
            "z_train": z_tr, "z_test": z_te}


# ---------------------------------------------------------------------------
# SECONDARY - mRMR
# ---------------------------------------------------------------------------

def mrmr_fold(cfg, mat, tr_pids, te_pids, inner, k):
    scfg = cfg["secondary"]["mrmr"]
    itr = mat["index"]
    tr_idx = np.asarray([itr[p] for p in tr_pids])
    te_idx = np.asarray([itr[p] for p in te_pids])
    y_tr = mat["y"][tr_idx]

    oof = np.full(len(tr_pids), np.nan)
    for a, b in inner:
        pre = TrainingOnlyPreprocessor(
            abs_tol=float(cfg["pipeline"]["variance_filter"]["abs_tol"]),
            rel_tol=float(cfg["pipeline"]["variance_filter"]["rel_tol"])).fit(mat["x"][tr_idx[a]])
        xa = pre.transform(mat["x"][tr_idx[a]])
        sel = mrmr_select(xa, y_tr[a], k, scfg)
        est = fit_logistic(xa[:, sel], y_tr[a], scfg)
        oof[b] = est.predict_proba(pre.transform(mat["x"][tr_idx[b]])[:, sel])[:, 1]

    pre = TrainingOnlyPreprocessor(
        abs_tol=float(cfg["pipeline"]["variance_filter"]["abs_tol"]),
        rel_tol=float(cfg["pipeline"]["variance_filter"]["rel_tol"])).fit(mat["x"][tr_idx])
    xa = pre.transform(mat["x"][tr_idx])
    sel = mrmr_select(xa, y_tr, k, scfg)
    est = fit_logistic(xa[:, sel], y_tr, scfg)
    p_test = est.predict_proba(pre.transform(mat["x"][te_idx])[:, sel])[:, 1]
    kept = pre.kept_indices
    return {"inner_oof": oof, "p_test": p_test,
            "selected_feature_names": [mat["names"][int(kept[j])] for j in sel]}


# ---------------------------------------------------------------------------
# calibration + threshold, applied to every model
# ---------------------------------------------------------------------------

def calibrate_and_threshold(cfg, inner_oof, y_tr, p_test):
    cal = PlattCalibrator().fit(inner_oof, y_tr, cfg["calibration"])
    cal_inner = cal.apply(inner_oof)
    thr, bal, _ = select_threshold(cal_inner, y_tr, cfg["threshold"]["tie_break"])
    return cal, cal_inner, thr, bal, cal.apply(p_test)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/phase7a.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    t_start = time.time()

    for key in ("models_dir", "preprocessing_dir", "predictions_dir", "results_dir"):
        os.makedirs(ppath(cfg, key), exist_ok=True)
    for sub in ("feature_selection", "calibration", "thresholds", "secondary_mrmr",
                "hyperparameter_surfaces"):
        os.makedirs(os.path.join(ppath(cfg, "results_dir"), sub), exist_ok=True)
    for m in PRIMARY:
        os.makedirs(os.path.join(ppath(cfg, "models_dir"), m), exist_ok=True)
    inner_dir = os.path.join(cfg["_project_root"], "splits", "phase7a_inner_cv")
    os.makedirs(inner_dir, exist_ok=True)
    os.makedirs(os.path.dirname(ppath(cfg, "training_log")), exist_ok=True)
    fh = open(ppath(cfg, "training_log"), "w", encoding="utf-8")

    cohort = load_cohort(cfg)
    gtv = load_feature_matrix(cfg, "gtv")
    rim = load_feature_matrix(cfg, "rim")
    preflight = verify_frozen_inputs(cfg, cohort, gtv, rim, fh)
    clinical = load_clinical(cfg, list(cohort.PatientID))
    y_map = {p: int(l) for p, l in zip(cohort.PatientID, cohort.label)}
    hist = {p: h for p, h in zip(cohort.PatientID, cohort.Histology)}
    folds = outer_folds(cfg, cohort)

    rows = {m: [] for m in PRIMARY}
    sec_rows = {}
    records = {"folds": {}, "secondary": {}}

    for k in sorted(folds):
        tr_pids, te_pids = folds[k]["train"], folds[k]["test"]
        y_tr = np.asarray([y_map[p] for p in tr_pids], dtype=int)
        inner, seed = inner_folds(cfg, tr_pids, y_tr, k)
        log("fold %d: %d train (%d ADC) / %d test (%d ADC), inner seed %d"
            % (k, len(tr_pids), int(y_tr.sum()), len(te_pids),
               sum(y_map[p] for p in te_pids), seed), fh)

        assign = {p: -1 for p in tr_pids}
        for j, (_, b) in enumerate(inner):
            for i in b:
                assign[tr_pids[i]] = j
        pd.DataFrame([{"PatientID": p, "outer_fold": k,
                       "split": "train" if p in assign else "test",
                       "inner_fold": assign.get(p, -1), "label": y_map[p]}
                      for p in tr_pids + te_pids]).to_csv(
            os.path.join(inner_dir, "fold_%d.csv" % k), index=False)

        res = {}
        res["A1"] = radiomics_fold(cfg, "gtv", gtv, k, tr_pids, te_pids, inner, fh, "A1")
        res["A2"] = radiomics_fold(cfg, "rim", rim, k, tr_pids, te_pids, inner, fh, "A2")
        res["A0"] = clinical_fold(cfg, clinical, list(cohort.PatientID), k,
                                  tr_pids, te_pids, inner, y_map, fh)
        res["A3"] = fusion_fold(cfg, res["A1"], res["A2"], y_tr, inner, fh, k)

        fold_rec = {"outer_fold": k, "inner_seed": seed,
                    "n_train": len(tr_pids), "n_test": len(te_pids),
                    "train_adc": int(y_tr.sum()),
                    "test_adc": int(sum(y_map[p] for p in te_pids))}

        for m in PRIMARY:
            r = res[m]
            cal, cal_inner, thr, bal, p_cal_test = calibrate_and_threshold(
                cfg, r["inner_oof"], y_tr, r["p_test"])
            for p, praw, pcal in zip(te_pids, r["p_test"], p_cal_test):
                rows[m].append({"PatientID": p, "Histology": hist[p], "label": y_map[p],
                                "fold": k, "p_adc_raw": float(praw),
                                "p_adc_calibrated": float(pcal),
                                "threshold": float(thr),
                                "pred_at_threshold": int(pcal >= thr),
                                "pred_at_0.5": int(pcal >= 0.5)})
            fold_rec[m] = {
                "calibration": {"a": cal.a, "b": cal.b, "n_inner_oof": cal.n_fit},
                "threshold": float(thr), "inner_balanced_accuracy": float(bal),
                "inner_oof_roc_auc": _auc(y_tr, r["inner_oof"]),
                "inner_oof_roc_auc_calibrated": _auc(y_tr, cal_inner)}
            if m in ("A1", "A2"):
                fold_rec[m].update({
                    "l1_ratio": r["selected"]["l1_ratio"], "C": r["selected"]["C"],
                    "mean_inner_roc_auc": r["selected"]["mean_inner_roc_auc"],
                    "n_kept_dimensions": int(r["pre"].keep.sum()),
                    "n_input_dimensions": int(r["pre"].n_input),
                    "n_nonzero_stage1": r["n_nonzero_stage1"],
                    "n_selected_features": len(r["selected_feature_names"]),
                    "selected_features": r["selected_feature_names"],
                    "coefficients": r["coefficients"], "intercept": r["intercept"],
                    "converged": r["converged"], "seconds": r["seconds"]})
                np.savez_compressed(
                    os.path.join(ppath(cfg, "preprocessing_dir"), "%s_fold_%d_final.npz" % (m, k)),
                    **r["pre"].arrays())
                pd.DataFrame(r["surface"]).to_csv(
                    os.path.join(ppath(cfg, "results_dir"), "hyperparameter_surfaces",
                                 "%s_fold_%d.csv" % (m, k)), index=False)
            if m == "A0":
                fold_rec[m].update({"C": r["selected"]["C"],
                                    "mean_inner_roc_auc": r["selected"]["mean_inner_roc_auc"],
                                    "encoder_columns": r["encoder_columns"],
                                    "kept_columns": r["kept_columns"],
                                    "categories": r["categories"],
                                    "training_medians": r["medians"],
                                    "training_modes": r["modes"],
                                    "coefficients": r["coefficients"],
                                    "intercept": r["intercept"]})
            if m == "A3":
                fold_rec[m].update({"coefficients": r["coefficients"],
                                    "intercept": r["intercept"],
                                    "fusion_inputs": r["fusion_inputs"],
                                    "n_fusion_training_rows": r["n_fusion_training_rows"]})
                pd.DataFrame({"PatientID": tr_pids, "label": y_tr,
                              "P_ADC_GTV": r["z_train"][:, 0],
                              "P_ADC_RIM": r["z_train"][:, 1],
                              "source": "inner_oof"}).to_csv(
                    os.path.join(ppath(cfg, "models_dir"), "A3",
                                 "fold_%d_fusion_training_inputs.csv" % k), index=False)
            with open(os.path.join(ppath(cfg, "models_dir"), m, "fold_%d.json" % k),
                      "w", encoding="utf-8") as f:
                json.dump({kk: vv for kk, vv in fold_rec[m].items()}, f, indent=1, default=float)

        records["folds"][k] = fold_rec

        # ---- SECONDARY mRMR ------------------------------------------------
        if bool(cfg["secondary"]["mrmr"]["enabled"]):
            for region, mat in (("gtv", gtv), ("rim", rim)):
                for kk in cfg["secondary"]["mrmr"]["k"]:
                    tag = "mrmr_%s_k%d" % (region, kk)
                    r = mrmr_fold(cfg, mat, tr_pids, te_pids, inner, kk)
                    cal, _, thr, _, p_cal = calibrate_and_threshold(
                        cfg, r["inner_oof"], y_tr, r["p_test"])
                    sec_rows.setdefault(tag, [])
                    for p, praw, pcal in zip(te_pids, r["p_test"], p_cal):
                        sec_rows[tag].append({"PatientID": p, "Histology": hist[p],
                                              "label": y_map[p], "fold": k,
                                              "p_adc_raw": float(praw),
                                              "p_adc_calibrated": float(pcal),
                                              "threshold": float(thr),
                                              "pred_at_threshold": int(pcal >= thr)})
                    records["secondary"].setdefault(tag, {})[k] = {
                        "selected_features": r["selected_feature_names"],
                        "threshold": float(thr)}
            log("  secondary mRMR done for fold %d" % k, fh)

    # ---- write predictions -------------------------------------------------
    for m in PRIMARY:
        df = pd.DataFrame(rows[m]).sort_values("PatientID").reset_index(drop=True)
        if len(df) != 202 or df.PatientID.duplicated().any():
            raise RuntimeError("%s produced %d rows (duplicates: %s)"
                               % (m, len(df), df.PatientID.duplicated().sum()))
        df.to_csv(os.path.join(ppath(cfg, "predictions_dir"), PRED_FILE[m]), index=False)
    for tag, r in sec_rows.items():
        pd.DataFrame(r).sort_values("PatientID").to_csv(
            os.path.join(ppath(cfg, "results_dir"), "secondary_mrmr", "%s_oof.csv" % tag),
            index=False)

    # ---- calibration / threshold / feature-selection summaries -------------
    cal_rows, thr_rows, fs_rows = [], [], []
    for k, fr in records["folds"].items():
        for m in PRIMARY:
            cal_rows.append({"model": m, "fold": k, "platt_a": fr[m]["calibration"]["a"],
                             "platt_b": fr[m]["calibration"]["b"],
                             "n_inner_oof": fr[m]["calibration"]["n_inner_oof"],
                             "inner_oof_roc_auc": fr[m]["inner_oof_roc_auc"]})
            thr_rows.append({"model": m, "fold": k, "threshold": fr[m]["threshold"],
                             "inner_balanced_accuracy": fr[m]["inner_balanced_accuracy"]})
        for m in ("A1", "A2"):
            for name, coef in zip(fr[m]["selected_features"], fr[m]["coefficients"]):
                fs_rows.append({"model": m, "fold": k, "feature": name,
                                "coefficient": coef,
                                "l1_ratio": fr[m]["l1_ratio"], "C": fr[m]["C"]})
    rd = ppath(cfg, "results_dir")
    pd.DataFrame(cal_rows).to_csv(os.path.join(rd, "calibration", "platt_parameters.csv"), index=False)
    pd.DataFrame(thr_rows).to_csv(os.path.join(rd, "thresholds", "selected_thresholds.csv"), index=False)
    pd.DataFrame(fs_rows).to_csv(os.path.join(rd, "feature_selection", "selected_features.csv"), index=False)
    with open(os.path.join(rd, "secondary_mrmr", "selected_features.json"), "w", encoding="utf-8") as f:
        json.dump(records["secondary"], f, indent=1)

    run = {"phase": "7A", "completed": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "config": os.path.relpath(cfg["_config_path"], cfg["_project_root"]).replace(os.sep, "/"),
           "protocol_doc": "docs/PHASE7_PROTOCOL.md",
           "protocol_sha256": preflight["protocol_sha256"],
           "config_sha256": preflight["config_sha256"],
           "pyradiomics_params_sha256": preflight["pyradiomics_params_sha256"],
           "preflight_verification": preflight,
           "inner_cv": {"n_splits": cfg["validation"]["inner"]["n_splits"],
                        "base_seed": cfg["validation"]["inner"]["base_seed"]},
           "folds": records["folds"],
           "wall_seconds": round(time.time() - t_start, 1),
           "environment": _environment()}
    with open(ppath(cfg, "run_json"), "w", encoding="utf-8") as f:
        json.dump(run, f, indent=1, default=float)
    log("phase 7A nested CV complete in %.1f s -> %s" % (run["wall_seconds"], ppath(cfg, "run_json")), fh)
    fh.close()


def _environment():
    import sklearn
    import scipy
    import radiomics as rad
    import SimpleITK as sitk
    return {"python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "scipy": scipy.__version__,
            "pyradiomics": rad.__version__, "SimpleITK": sitk.__version__}


if __name__ == "__main__":
    main()
