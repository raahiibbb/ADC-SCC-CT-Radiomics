"""Phase 8B - the three evaluation views, the diagnostics, the bootstraps and
the frozen Gate 8B.

Everything scored here is PATIENT level: one out-of-fold probability per patient
per model.  The bootstrap resamples PATIENTS (PatientKey), never feature rows.

PRIMARY discrimination is the RAW out-of-fold probability (protocol 9.1),
frozen before modelling.  Calibrated / thresholded metrics are reported
separately and never determine the gate.

    VIEW 1  pooled source-stratified OOF over all 343
    VIEW 2  the SAME pooled predictions subset by cohort - nothing retrained
    VIEW 3  LUNG1 -> Radiogenomics and Radiogenomics -> LUNG1

    ./.venv/Scripts/python.exe src/phase8b_evaluate.py --config config/phase8b_multicenter.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mil_metrics import curve_points, mean_sd, patient_metrics                 # noqa: E402
from phase8b_data import load_cfg, log, ppath, sha256, verify_frozen           # noqa: E402
from phase8b_run import MODELS, PRED_FILE                                      # noqa: E402

COHORTS = ("LUNG1", "RADIOGENOMICS")
DIRECTIONS = ("lung1_to_radiogenomics", "radiogenomics_to_lung1")


# ---------------------------------------------------------------------------
# metric helpers
# ---------------------------------------------------------------------------

def brier(y, p):
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def calibration_intercept_slope(y, p, clip=1e-6):
    y = np.asarray(y, dtype=int)
    pc = np.clip(np.asarray(p, dtype=float), clip, 1 - clip)
    z = np.log(pc / (1 - pc)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    slope = float(LogisticRegression(C=1e6, solver="lbfgs", max_iter=20000).fit(z, y).coef_[0][0])
    lo, hi = -20.0, 20.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        g = float(np.mean(y - 1.0 / (1.0 + np.exp(-(mid + z.ravel())))))
        lo, hi = (mid, hi) if g > 0 else (lo, mid)
    return float(0.5 * (lo + hi)), slope


def block_metrics(df, tag):
    """Raw (PRIMARY) discrimination plus calibrated / thresholded metrics."""
    y = df.label.to_numpy(int)
    raw, cal = df.p_adc_raw.to_numpy(float), df.p_adc_calibrated.to_numpy(float)
    out = {"scope": tag, "n_patients": int(len(df)), "n_adc": int((y == 1).sum()),
           "n_scc": int((y == 0).sum()), "prevalence_adc": float((y == 1).mean())}
    both = len(np.unique(y)) == 2
    out["raw"] = {"roc_auc": float(roc_auc_score(y, raw)) if both else float("nan"),
                  "pr_auc": float(average_precision_score(y, raw)) if both else float("nan")}
    m = patient_metrics(y, cal, threshold=float(df.threshold.iloc[0])) if len(
        df.threshold.unique()) == 1 else None
    if m is None:                       # pooled across folds: apply each row's own threshold
        yhat = (cal >= df.threshold.to_numpy(float)).astype(int)
        m = _metrics_from_labels(y, cal, yhat)
    m["brier"] = brier(y, cal)
    ci, cs = calibration_intercept_slope(y, cal)
    m["calibration_intercept"], m["calibration_slope"] = ci, cs
    m["roc_auc_calibrated"] = m.pop("roc_auc")
    m["pr_auc_calibrated"] = m.pop("pr_auc")
    out["calibrated"] = m
    yhat05 = (cal >= 0.5).astype(int)
    out["legacy_threshold_0.5"] = _metrics_from_labels(y, cal, yhat05)
    return out


def _metrics_from_labels(y, p, yhat):
    from sklearn.metrics import confusion_matrix, matthews_corrcoef
    y = np.asarray(y, int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * prec * sens / (prec + sens)) if (prec + sens) else 0.0
    both = len(np.unique(y)) == 2
    return {"n_patients": int(len(y)), "n_adc": int((y == 1).sum()), "n_scc": int((y == 0).sum()),
            "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
            "pr_auc": float(average_precision_score(y, p)) if both else float("nan"),
            "balanced_accuracy": float(np.nanmean([sens, spec])),
            "sensitivity_adc": float(sens), "specificity": float(spec),
            "precision": float(prec), "f1": float(f1),
            "mcc": float(matthews_corrcoef(y, yhat)),
            "accuracy": float((yhat == y).mean()),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


# ---------------------------------------------------------------------------
# bootstraps
# ---------------------------------------------------------------------------

def source_stratified_bootstrap(y, cohort, pa, pb, n_resamples, seed, ci=0.95):
    """Resample WITHIN each cohort, preserving 202 + 141, then combine.

    Both prediction vectors are scored on exactly the same sampled patients, so
    every difference is paired within a resample.  The unit is the patient.
    """
    y = np.asarray(y, int)
    pa, pb = np.asarray(pa, float), np.asarray(pb, float)
    coh = np.asarray(cohort)
    blocks = [np.flatnonzero(coh == c) for c in COHORTS]
    rng = np.random.default_rng(int(seed))
    d_auc, d_pr, used = [], [], 0
    for _ in range(int(n_resamples)):
        idx = np.concatenate([b[rng.integers(0, len(b), size=len(b))] for b in blocks])
        ys = y[idx]
        if len(np.unique(ys)) < 2:
            continue
        used += 1
        d_auc.append(roc_auc_score(ys, pa[idx]) - roc_auc_score(ys, pb[idx]))
        d_pr.append(average_precision_score(ys, pa[idx]) - average_precision_score(ys, pb[idx]))
    return _summarise(y, pa, pb, d_auc, d_pr, n_resamples, used, seed,
                      "source_stratified_patient", {c: int(len(b)) for c, b in zip(COHORTS, blocks)})


def within_cohort_bootstrap(y, pa, pb, n_resamples, seed, ci=0.95):
    y = np.asarray(y, int)
    pa, pb = np.asarray(pa, float), np.asarray(pb, float)
    n = len(y)
    rng = np.random.default_rng(int(seed))
    d_auc, d_pr, used = [], [], 0
    for _ in range(int(n_resamples)):
        idx = rng.integers(0, n, size=n)
        ys = y[idx]
        if len(np.unique(ys)) < 2:
            continue
        used += 1
        d_auc.append(roc_auc_score(ys, pa[idx]) - roc_auc_score(ys, pb[idx]))
        d_pr.append(average_precision_score(ys, pa[idx]) - average_precision_score(ys, pb[idx]))
    return _summarise(y, pa, pb, d_auc, d_pr, n_resamples, used, seed,
                      "patient_within_target_cohort", {"target": int(n)})


def _summarise(y, pa, pb, d_auc, d_pr, n_resamples, used, seed, scheme, sizes, ci=0.95):
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    out = {"n_resamples": int(n_resamples), "n_usable": int(used), "seed": int(seed),
           "unit": "patient", "scheme": scheme, "block_sizes": sizes}
    for tag, d, obs in (("roc_auc", d_auc, roc_auc_score(y, pa) - roc_auc_score(y, pb)),
                        ("pr_auc", d_pr, average_precision_score(y, pa)
                         - average_precision_score(y, pb))):
        d = np.asarray(d)
        out[tag] = {"observed_difference": float(obs),
                    "bootstrap_mean_difference": float(d.mean()) if d.size else float("nan"),
                    "ci_low": float(np.percentile(d, lo)) if d.size else float("nan"),
                    "ci_high": float(np.percentile(d, hi)) if d.size else float("nan"),
                    "ci_includes_zero": bool(d.size and
                                             np.percentile(d, lo) <= 0.0 <= np.percentile(d, hi)),
                    "fraction_favouring_a": float((d > 0).mean()) if d.size else float("nan")}
    return out


# ---------------------------------------------------------------------------
# feature analysis
# ---------------------------------------------------------------------------

def _image_type(name):
    head = name.split("_")[0]
    if head == "original":
        return "original"
    if head.startswith("log-sigma"):
        return "LoG"
    if head.startswith("wavelet"):
        return "wavelet"
    return head


def _family(name):
    parts = name.split("_")
    return parts[1] if len(parts) > 1 else "unknown"


def feature_analysis(cfg, run):
    """Selection frequency, family / image-type composition and pairwise Jaccard
    stability, plus a comparison against the frozen Phase-7A LUNG1-only picks."""
    folds = run["modelling"]["pooled_folds"]
    freq_rows, stab_rows, per_model = [], [], {}
    for m, region in (("A1", "gtv"), ("A2", "rim")):
        sets = {}
        for k in sorted(folds, key=int):
            fr = folds[k][m]
            sets[int(k)] = set(fr["selected_features"])
            for name in fr["selected_features"]:
                freq_rows.append({"model": m, "region": region, "fold": int(k),
                                  "feature": name, "family": _family(name),
                                  "image_type": _image_type(name),
                                  "selector_family": fr["selector_family"],
                                  "candidate_id": fr["candidate_id"]})
        ks = sorted(sets)
        js = []
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                a, b = sets[ks[i]], sets[ks[j]]
                v = len(a & b) / len(a | b) if (a | b) else float("nan")
                js.append(v)
                stab_rows.append({"model": m, "region": region, "fold_a": ks[i],
                                  "fold_b": ks[j], "jaccard": float(v),
                                  "n_a": len(a), "n_b": len(b), "n_shared": len(a & b)})
        allf = [f for s in sets.values() for f in s]
        counts = pd.Series(allf).value_counts()
        per_model[m] = {
            "region": region,
            "selector_family_per_fold": {str(k): folds[k][m]["selector_family"]
                                         for k in sorted(folds, key=int)},
            "candidate_per_fold": {str(k): folds[k][m]["candidate_id"]
                                   for k in sorted(folds, key=int)},
            "n_selected_per_fold": {str(k): folds[k][m]["n_selected_features"]
                                    for k in sorted(folds, key=int)},
            "n_distinct_features": int(len(counts)),
            "selected_in_all_5_folds": sorted(counts[counts == 5].index.tolist()),
            "most_frequent": [{"feature": f, "folds": int(c)}
                              for f, c in counts.head(12).items()],
            "family_composition": pd.Series([_family(f) for f in allf]).value_counts().to_dict(),
            "image_type_composition": pd.Series(
                [_image_type(f) for f in allf]).value_counts().to_dict(),
            "mean_pairwise_jaccard": float(np.mean(js)) if js else float("nan")}

    # comparison against Phase 7A (LUNG1 only)
    p7 = os.path.join(cfg["_project_root"], "results", "phase7a",
                      "feature_selection", "selected_features.csv")
    cmp = {}
    if os.path.isfile(p7):
        d7 = pd.read_csv(p7)
        for m in ("A1", "A2"):
            s7 = set(d7[d7.model == m].feature)
            s8 = set(f for s in [set(folds[k][m]["selected_features"])
                                 for k in sorted(folds, key=int)] for f in s)
            cmp[m] = {"phase7a_distinct": len(s7), "phase8b_distinct": len(s8),
                      "shared": sorted(s7 & s8), "n_shared": len(s7 & s8),
                      "jaccard_vs_phase7a": float(len(s7 & s8) / len(s7 | s8)) if (s7 | s8) else 0.0,
                      "phase8b_only_examples": sorted(s8 - s7)[:10]}
    return pd.DataFrame(freq_rows), pd.DataFrame(stab_rows), per_model, cmp


# ---------------------------------------------------------------------------
# prediction-shift diagnostic
# ---------------------------------------------------------------------------

def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = float(np.sum(a[:, None] > b[None, :]))
    lt = float(np.sum(a[:, None] < b[None, :]))
    return float((gt - lt) / (a.size * b.size))


def prediction_shift(preds):
    """Within each TRUE class, compare pooled OOF predictions across cohorts.

    Computed only AFTER the predictions are frozen on disk, and never used to
    tune anything."""
    out = {"note": ("descriptive only; computed after the OOF predictions were written "
                    "to disk; never used for tuning"), "models": {}}
    for m, df in preds.items():
        per = {}
        for cls, lab in (("ADC", 1), ("SCC", 0)):
            sub = df[df.label == lab]
            a = sub[sub.Cohort == "LUNG1"].p_adc_raw.to_numpy(float)
            b = sub[sub.Cohort == "RADIOGENOMICS"].p_adc_raw.to_numpy(float)
            per[cls] = {
                "n_lung1": int(a.size), "n_radiogenomics": int(b.size),
                "lung1": {"median": float(np.median(a)) if a.size else float("nan"),
                          "q1": float(np.percentile(a, 25)) if a.size else float("nan"),
                          "q3": float(np.percentile(a, 75)) if a.size else float("nan")},
                "radiogenomics": {"median": float(np.median(b)) if b.size else float("nan"),
                                  "q1": float(np.percentile(b, 25)) if b.size else float("nan"),
                                  "q3": float(np.percentile(b, 75)) if b.size else float("nan")},
                "median_difference_rg_minus_lung1":
                    float(np.median(b) - np.median(a)) if (a.size and b.size) else float("nan"),
                "cliffs_delta_rg_vs_lung1": cliffs_delta(b, a)}
        out["models"][m] = per
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/phase8b_multicenter.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    t0 = time.time()
    run = verify_frozen(cfg)
    if "modelling" not in run:
        raise RuntimeError("src/phase8b_run.py has not completed - nothing to evaluate")

    rd = ppath(cfg, "results_dir")
    for sub in ("roc_curves", "pr_curves", "confusion_matrices", "calibration", "thresholds"):
        os.makedirs(os.path.join(rd, sub), exist_ok=True)
    fh = open(os.path.join(cfg["_project_root"], "logs", "phase8b_evaluate.log"),
              "w", encoding="utf-8")

    # ---- load the frozen predictions ------------------------------------
    pooled = {m: pd.read_csv(os.path.join(ppath(cfg, "predictions_dir"),
                                          PRED_FILE["pooled"][m])) for m in MODELS}
    n_expect = int(cfg["cohorts"]["combined"]["expected_patients"])
    for m, df in pooled.items():
        if len(df) != n_expect or df.PatientKey.duplicated().any():
            raise RuntimeError("pooled %s has %d rows / %d duplicates"
                               % (m, len(df), int(df.PatientKey.duplicated().sum())))
    transfer = {d: {m: pd.read_csv(os.path.join(ppath(cfg, "predictions_dir"),
                                                PRED_FILE[d][m])) for m in MODELS}
                for d in DIRECTIONS}

    # =====================================================================
    # VIEW 1 - pooled source-stratified OOF
    # =====================================================================
    view1 = {"description": "source-stratified 5-fold outer CV over all 343 patients",
             "primary_probability": cfg["evaluation"]["primary_probability"],
             "models": {}, "per_fold": {}}
    fold_rows = []
    for m, df in pooled.items():
        view1["models"][m] = block_metrics(df, "pooled_343")
        per = {}
        for k in sorted(df.fold.unique()):
            sub = df[df.fold == k]
            per[int(k)] = block_metrics(sub, "pooled_fold_%d" % k)
            fold_rows.append({"model": m, "fold": int(k),
                              "roc_auc_raw": per[int(k)]["raw"]["roc_auc"],
                              "pr_auc_raw": per[int(k)]["raw"]["pr_auc"],
                              "roc_auc_calibrated": per[int(k)]["calibrated"]["roc_auc_calibrated"],
                              "balanced_accuracy": per[int(k)]["calibrated"]["balanced_accuracy"],
                              "sensitivity_adc": per[int(k)]["calibrated"]["sensitivity_adc"],
                              "specificity": per[int(k)]["calibrated"]["specificity"],
                              "mcc": per[int(k)]["calibrated"]["mcc"],
                              "brier": per[int(k)]["calibrated"]["brier"],
                              "threshold": float(sub.threshold.iloc[0])})
        view1["per_fold"][m] = per
        view1["models"][m]["fold_roc_auc_raw_mean_sd"] = mean_sd(
            [per[k]["raw"]["roc_auc"] for k in per])
        view1["models"][m]["fold_pr_auc_raw_mean_sd"] = mean_sd(
            [per[k]["raw"]["pr_auc"] for k in per])
        log("VIEW1 %s: pooled RAW ROC-AUC %.4f  PR-AUC %.4f  (fold mean %.3f +/- %.3f)"
            % (m, view1["models"][m]["raw"]["roc_auc"], view1["models"][m]["raw"]["pr_auc"],
               view1["models"][m]["fold_roc_auc_raw_mean_sd"]["mean"],
               view1["models"][m]["fold_roc_auc_raw_mean_sd"]["sd"]), fh)
    pd.DataFrame(fold_rows).to_csv(os.path.join(rd, "pooled_fold_metrics.csv"), index=False)

    # GATE 4 - A3 beating BOTH components, per fold, on the RAW probability
    folds_beating = []
    for k in sorted(pooled["A3"].fold.unique()):
        a1 = view1["per_fold"]["A1"][int(k)]["raw"]["roc_auc"]
        a2 = view1["per_fold"]["A2"][int(k)]["raw"]["roc_auc"]
        a3 = view1["per_fold"]["A3"][int(k)]["raw"]["roc_auc"]
        folds_beating.append({"fold": int(k), "A1": a1, "A2": a2, "A3": a3,
                              "A3_beats_both": bool(a3 > a1 and a3 > a2)})
    view1["a3_vs_components_per_fold"] = folds_beating
    n_beating = int(sum(f["A3_beats_both"] for f in folds_beating))

    # =====================================================================
    # VIEW 2 - cohort-specific, from the SAME pooled predictions
    # =====================================================================
    view2 = {"description": ("the identical pooled OOF predictions, subset by cohort; "
                             "NOTHING is retrained"),
             "retrained": False, "cohorts": {}}
    for c in COHORTS:
        view2["cohorts"][c] = {}
        for m, df in pooled.items():
            sub = df[df.Cohort == c]
            view2["cohorts"][c][m] = block_metrics(sub, "%s_subset_of_pooled_oof" % c)
        log("VIEW2 %s: A1 %.4f  A2 %.4f  A3 %.4f  (raw ROC-AUC, n=%d)"
            % (c, view2["cohorts"][c]["A1"]["raw"]["roc_auc"],
               view2["cohorts"][c]["A2"]["raw"]["roc_auc"],
               view2["cohorts"][c]["A3"]["raw"]["roc_auc"],
               view2["cohorts"][c]["A1"]["n_patients"]), fh)

    # =====================================================================
    # VIEW 3 - true cross-cohort transport
    # =====================================================================
    view3 = {"description": "each direction trained inside one cohort and scored ONCE",
             "directions": {}}
    for d in DIRECTIONS:
        view3["directions"][d] = {m: block_metrics(transfer[d][m], d) for m in MODELS}
        log("VIEW3 %s: A1 %.4f  A2 %.4f  A3 %.4f  (raw ROC-AUC)"
            % (d, view3["directions"][d]["A1"]["raw"]["roc_auc"],
               view3["directions"][d]["A2"]["raw"]["roc_auc"],
               view3["directions"][d]["A3"]["raw"]["roc_auc"]), fh)
    t_a3 = [view3["directions"][d]["A3"]["raw"]["roc_auc"] for d in DIRECTIONS]
    view3["A3_mean_bidirectional_roc_auc"] = float(np.mean(t_a3))
    view3["A3_min_direction_roc_auc"] = float(np.min(t_a3))

    # =====================================================================
    # bootstraps
    # =====================================================================
    boot = {}
    bcfg = cfg["evaluation"]["bootstrap"]
    for comp in bcfg["comparisons"]:
        a, b, scope = comp["a"], comp["b"], comp["scope"]
        if scope == "pooled":
            base = pooled[a][["PatientKey", "Cohort", "label"]].copy()
            pa = pooled[a].sort_values("PatientKey").p_adc_raw.to_numpy(float)
            pb = pooled[b].sort_values("PatientKey").p_adc_raw.to_numpy(float)
            base = base.sort_values("PatientKey")
            boot[comp["tag"]] = source_stratified_bootstrap(
                base.label.to_numpy(int), base.Cohort.to_numpy(), pa, pb,
                bcfg["n_resamples"], comp["seed"])
        else:
            da = transfer[scope][a].sort_values("PatientKey")
            db = transfer[scope][b].sort_values("PatientKey")
            boot[comp["tag"]] = within_cohort_bootstrap(
                da.label.to_numpy(int), da.p_adc_raw.to_numpy(float),
                db.p_adc_raw.to_numpy(float), bcfg["n_resamples"], comp["seed"])
        boot[comp["tag"]]["comparison"] = "%s - %s" % (a, b)
        boot[comp["tag"]]["scope"] = scope
        log("bootstrap %-24s dROC %+.4f [%+.4f, %+.4f]  (%d/%d usable)"
            % (comp["tag"], boot[comp["tag"]]["roc_auc"]["observed_difference"],
               boot[comp["tag"]]["roc_auc"]["ci_low"], boot[comp["tag"]]["roc_auc"]["ci_high"],
               boot[comp["tag"]]["n_usable"], boot[comp["tag"]]["n_resamples"]), fh)

    # =====================================================================
    # feature analysis, diagnostics, references
    # =====================================================================
    freq, stab, per_model, cmp7a = feature_analysis(cfg, run)
    freq.to_csv(os.path.join(rd, "feature_selection_frequency.csv"), index=False)
    stab.to_csv(os.path.join(rd, "feature_selection_stability.csv"), index=False)

    shift = prediction_shift(pooled)
    domain = run["modelling"]["domain_diagnostic"]

    refs = {}
    for tag, r in cfg["references"].items():
        p = os.path.join(cfg["_project_root"], *r["predictions"].split("/"))
        if not os.path.isfile(p):
            refs[tag] = {"available": False, "path": r["predictions"]}
            continue
        d = pd.read_csv(p)
        col_y = next(c for c in ("label", "true_label") if c in d.columns)
        col_raw = next(c for c in ("p_adc_raw", "predicted_probability_ADC", "p_adc")
                       if c in d.columns)
        col_cal = "p_adc_calibrated" if "p_adc_calibrated" in d.columns else col_raw
        e = {"available": True, "path": r["predictions"], "sha256": sha256(p),
             "n": int(len(d)), "retrained": False,
             "label_column": col_y, "probability_column": col_raw,
             "roc_auc_raw": float(roc_auc_score(d[col_y], d[col_raw])),
             "roc_auc_calibrated": float(roc_auc_score(d[col_y], d[col_cal]))}
        for key, want in r.items():
            if key.startswith("expected_roc_auc"):
                got = e["roc_auc_calibrated"] if key.endswith("calibrated") else (
                    e["roc_auc_raw"] if key.endswith("raw") else e["roc_auc_calibrated"])
                e[key] = {"expected": float(want), "observed": got,
                          "matches": bool(abs(got - float(want)) < 5e-4)}
        refs[tag] = e

    # =====================================================================
    # GATE 8B - mechanical
    # =====================================================================
    # Every threshold, operator and target is read from the FROZEN config; none
    # is written here.  Each observed value is looked up by the metric name the
    # protocol froze, so a condition cannot silently be scored on the wrong
    # quantity or against a locally invented threshold.
    observed = {
        "pooled_A3_roc_auc": view1["models"]["A3"]["raw"]["roc_auc"],
        "lung1_subset_A3_roc_auc": view2["cohorts"]["LUNG1"]["A3"]["raw"]["roc_auc"],
        "radiogenomics_subset_A3_roc_auc":
            view2["cohorts"]["RADIOGENOMICS"]["A3"]["raw"]["roc_auc"],
        "A3_folds_beating_both_A1_and_A2": float(n_beating),
        "mean_bidirectional_transport_A3_roc_auc":
            view3["A3_mean_bidirectional_roc_auc"],
        "min_direction_A3_roc_auc": view3["A3_min_direction_roc_auc"]}
    ops = {">=": lambda a, b: a >= b, ">": lambda a, b: a > b,
           "<=": lambda a, b: a <= b, "<": lambda a, b: a < b}

    conditions = []
    for c in cfg["gate_8b"]["conditions"]:
        v = float(observed[c["metric"]])
        ok = bool(ops[c["operator"]](v, float(c["value"])))
        rec = {"id": int(c["id"]), "metric": c["metric"], "operator": c["operator"],
               "threshold": float(c["value"]), "observed": v, "pass": ok}
        if "of" in c:
            rec["of"] = int(c["of"])
            rec["per_fold"] = folds_beating
        if "and_also" in c:
            a = c["and_also"]
            va = float(observed[a["metric"]])
            oka = bool(ops[a["operator"]](va, float(a["value"])))
            rec["and_also"] = {"metric": a["metric"], "operator": a["operator"],
                               "threshold": float(a["value"]), "observed": va,
                               "pass": oka}
            rec["per_direction"] = {d: view3["directions"][d]["A3"]["raw"]["roc_auc"]
                                    for d in DIRECTIONS}
            rec["pass"] = bool(ok and oka)
        conditions.append(rec)

    if cfg["gate_8b"]["rule"] != "AND":
        raise RuntimeError("unknown gate rule %r" % cfg["gate_8b"]["rule"])
    passed = bool(all(c["pass"] for c in conditions))
    gate = {"rule": cfg["gate_8b"]["rule"],
            "probability": cfg["gate_8b"]["probability"],
            "conditions": conditions,
            "passed": passed, "result": passed,
            "on_fail": cfg["gate_8b"]["on_fail"], "on_pass": cfg["gate_8b"]["on_pass"],
            "phase8c_authorised": passed, "phase8c_implemented": False,
            "external_validation_claimable": False}
    log("\nGATE 8B: " + ("PASS" if passed else "FAIL"), fh)
    for c in conditions:
        log("  %d %-44s observed %.4f %s %.4f -> %s"
            % (c["id"], c["metric"], c["observed"], c["operator"], c["threshold"],
               "PASS" if c["pass"] else "FAIL"), fh)

    # =====================================================================
    # curves, confusion matrices, calibration / threshold tables
    # =====================================================================
    for m, df in pooled.items():
        c = curve_points(df.label.to_numpy(int), df.p_adc_raw.to_numpy(float))
        pd.DataFrame({"fpr": c["fpr"], "tpr": c["tpr"]}).to_csv(
            os.path.join(rd, "roc_curves", "pooled_%s_raw.csv" % m), index=False)
        pd.DataFrame({"recall": c["recall"], "precision": c["precision"]}).to_csv(
            os.path.join(rd, "pr_curves", "pooled_%s_raw.csv" % m), index=False)
        with open(os.path.join(rd, "confusion_matrices", "pooled_%s.json" % m),
                  "w", encoding="utf-8") as f:
            json.dump({"pooled": view1["models"][m]["calibrated"]["confusion"],
                       "by_cohort": {c2: view2["cohorts"][c2][m]["calibrated"]["confusion"]
                                     for c2 in COHORTS}}, f, indent=1)
    cal_rows, thr_rows = [], []
    for k, fr in run["modelling"]["pooled_folds"].items():
        for m in MODELS:
            cal_rows.append({"scope": "pooled", "model": m, "fold": int(k),
                             "platt_a": fr[m]["calibration"]["a"],
                             "platt_b": fr[m]["calibration"]["b"],
                             "n_inner_oof": fr[m]["calibration"]["n_inner_oof"],
                             "valid": fr[m]["calibration"]["valid"],
                             "inner_oof_roc_auc_raw": fr[m]["inner_oof_roc_auc_raw"]})
            thr_rows.append({"scope": "pooled", "model": m, "fold": int(k),
                             "threshold": fr[m]["threshold"],
                             "inner_balanced_accuracy": fr[m]["inner_balanced_accuracy"]})
    for d, rec in run["modelling"]["cross_cohort"].items():
        for m in MODELS:
            cal_rows.append({"scope": d, "model": m, "fold": -1,
                             "platt_a": rec[m]["calibration"]["a"],
                             "platt_b": rec[m]["calibration"]["b"],
                             "n_inner_oof": rec[m]["calibration"]["n_inner_oof"],
                             "valid": rec[m]["calibration"]["valid"],
                             "inner_oof_roc_auc_raw": rec[m]["inner_oof_roc_auc_raw"]})
            thr_rows.append({"scope": d, "model": m, "fold": -1,
                             "threshold": rec[m]["threshold"],
                             "inner_balanced_accuracy": rec[m]["inner_balanced_accuracy"]})
    pd.DataFrame(cal_rows).to_csv(os.path.join(rd, "calibration", "platt_parameters.csv"),
                                  index=False)
    pd.DataFrame(thr_rows).to_csv(os.path.join(rd, "thresholds", "selected_thresholds.csv"),
                                  index=False)

    # =====================================================================
    # write the result files
    # =====================================================================
    _dump(os.path.join(rd, "pooled_metrics.json"), {"view": 1, **view1})
    _dump(os.path.join(rd, "cohort_specific_metrics.json"), {"view": 2, **view2})
    _dump(os.path.join(rd, "cross_cohort_metrics.json"), {"view": 3, **view3})
    _dump(os.path.join(rd, "paired_bootstrap.json"), boot)
    _dump(os.path.join(rd, "domain_shift_diagnostic.json"), domain)
    _dump(os.path.join(rd, "prediction_shift_diagnostic.json"), shift)
    _dump(os.path.join(rd, "feature_selection_summary.json"),
          {"per_model": per_model, "versus_phase7a_lung1_only": cmp7a})
    _dump(os.path.join(rd, "gate_8b.json"), gate)
    _dump(os.path.join(rd, "reference_comparisons.json"), refs)

    rec = json.load(open(ppath(cfg, "run_json"), encoding="utf-8"))
    rec["evaluation"] = {"completed": time.strftime("%Y-%m-%dT%H:%M:%S"),
                         "wall_seconds": round(time.time() - t0, 1),
                         "view1": view1["models"], "view2": view2["cohorts"],
                         "view3": view3, "gate_8b": gate,
                         "feature_analysis": per_model,
                         "versus_phase7a": cmp7a,
                         "domain_diagnostic": domain,
                         "prediction_shift": shift,
                         "references": refs, "bootstrap": boot}
    with open(ppath(cfg, "run_json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, default=_default)
    log("evaluation complete in %.1f s" % (time.time() - t0), fh)
    fh.close()


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _dump(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=_default)


if __name__ == "__main__":
    main()
