"""Phase 7A - patient-level evaluation, paired bootstrap and the frozen Gate 7A.

Everything scored here is PATIENT level: one out-of-fold probability per patient
per model.  The bootstrap resamples PATIENTS, never feature rows and never
patches.

Primary threshold-dependent metrics use the fold's INNER-SELECTED threshold on
the CALIBRATED probability.  Threshold 0.5 is reported as a secondary legacy
comparison.  Primary pooled discrimination uses the calibrated out-of-fold
vector, as frozen in docs/PHASE7_PROTOCOL.md section 3.3, with the uncalibrated
pooled value reported alongside.

    python src/phase7a_evaluate.py --config config/phase7a.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mil_metrics import curve_points, mean_sd, patient_metrics                # noqa: E402
from phase7a_cv import PRED_FILE, PRIMARY                                     # noqa: E402
from phase7a_extract import load_cfg, ppath                                   # noqa: E402

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def brier(y, p):
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def calibration_intercept_slope(y, p, clip=1e-6):
    """Slope = coefficient of logit(p) in a logistic regression of y on logit(p).
    Intercept = the intercept of the same regression with the slope FIXED at 1
    (calibration-in-the-large, fitted as an offset).  Perfect calibration gives
    slope 1 and intercept 0."""
    y = np.asarray(y, dtype=int)
    z = np.log(np.clip(p, clip, 1 - clip) / (1 - np.clip(p, clip, 1 - clip))).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    slope = float(LogisticRegression(C=1e6, solver="lbfgs", max_iter=20000
                                     ).fit(z, y).coef_[0][0])
    # offset model: logit(pi) = intercept + 1 * z  ->  fit intercept only
    lo, hi = -20.0, 20.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        g = float(np.mean(y - 1.0 / (1.0 + np.exp(-(mid + z.ravel())))))
        if g > 0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi)), slope


def full_metrics(y, p, threshold):
    m = patient_metrics(y, p, threshold=float(threshold))
    m["brier"] = brier(y, p)
    ci, cs = calibration_intercept_slope(y, p)
    m["calibration_intercept"] = ci
    m["calibration_slope"] = cs
    return m


def paired_bootstrap(y, pa, pb, n_resamples, seed, ci=0.95):
    """Resample PATIENTS with replacement; score both vectors on exactly the
    same sampled patients, so every difference is paired within a resample."""
    y = np.asarray(y, dtype=int)
    pa = np.asarray(pa, dtype=float)
    pb = np.asarray(pb, dtype=float)
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
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    out = {"n_resamples": int(n_resamples), "n_usable": used, "seed": int(seed),
           "unit": "patient", "n_patients": int(n)}
    for tag, d, obs in (("roc_auc", d_auc, roc_auc_score(y, pa) - roc_auc_score(y, pb)),
                        ("pr_auc", d_pr, average_precision_score(y, pa)
                         - average_precision_score(y, pb))):
        d = np.asarray(d)
        out[tag] = {"observed_difference": float(obs),
                    "bootstrap_mean_difference": float(d.mean()),
                    "ci_low": float(np.percentile(d, lo)),
                    "ci_high": float(np.percentile(d, hi)),
                    "ci_includes_zero": bool(np.percentile(d, lo) <= 0.0 <= np.percentile(d, hi)),
                    "fraction_favouring_a": float((d > 0).mean())}
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/phase7a.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    t0 = time.time()
    rd = ppath(cfg, "results_dir")
    for sub in ("roc_curves", "pr_curves", "confusion_matrices", "paired_bootstrap"):
        os.makedirs(os.path.join(rd, sub), exist_ok=True)

    preds = {}
    for m in PRIMARY:
        df = pd.read_csv(os.path.join(ppath(cfg, "predictions_dir"), PRED_FILE[m]))
        df = df.sort_values("PatientID").reset_index(drop=True)
        if len(df) != 202 or df.PatientID.duplicated().any():
            raise RuntimeError("%s: expected 202 unique patients" % m)
        preds[m] = df
    ref = preds["A1"]
    y = ref.label.to_numpy(dtype=int)
    pids = list(ref.PatientID)
    for m in PRIMARY:
        if list(preds[m].PatientID) != pids or not np.array_equal(preds[m].label.to_numpy(), y):
            raise RuntimeError("%s: patient order or labels differ" % m)

    # ---- baselines, read from disk and never retrained ---------------------
    baselines = {}
    for name, b in cfg["baselines"].items():
        p = os.path.join(cfg["_project_root"], *b["predictions"].split("/"))
        df = pd.read_csv(p).sort_values("PatientID").reset_index(drop=True)
        col = next(c for c in ("predicted_probability_ADC", "p_adc_calibrated",
                               "p_adc_raw", "p_adc") if c in df.columns)
        lab = next(c for c in ("true_label", "label") if c in df.columns)
        if list(df.PatientID) != pids:
            raise RuntimeError("%s baseline covers different patients" % name)
        yb = df[lab].to_numpy(dtype=int)
        if not np.array_equal(yb, y):
            raise RuntimeError("%s baseline labels differ from the Phase-7A cohort" % name)
        auc = float(roc_auc_score(yb, df[col]))
        if abs(auc - float(b["expected_roc_auc"])) > 5e-4:
            raise RuntimeError("%s baseline scores %.4f, expected %.4f"
                               % (name, auc, b["expected_roc_auc"]))
        baselines[name] = {"p": df[col].to_numpy(dtype=float), "roc_auc": auc,
                           "pr_auc": float(average_precision_score(yb, df[col])),
                           "path": b["predictions"], "column": col,
                           "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}

    # ---- per-fold and pooled ----------------------------------------------
    fold_rows, pooled = [], {}
    for m in PRIMARY:
        df = preds[m]
        for k in sorted(df.fold.unique()):
            f = df[df.fold == k]
            thr = float(f.threshold.iloc[0])
            mm = full_metrics(f.label.to_numpy(dtype=int),
                              f.p_adc_calibrated.to_numpy(dtype=float), thr)
            legacy = patient_metrics(f.label.to_numpy(dtype=int),
                                     f.p_adc_calibrated.to_numpy(dtype=float), 0.5)
            row = {"model": m, "fold": int(k), "n": len(f),
                   "n_adc": int(f.label.sum()), "threshold": thr}
            row.update({kk: vv for kk, vv in mm.items() if kk != "threshold"})
            row.update({"legacy05_%s" % kk: vv for kk, vv in legacy.items()
                        if kk in ("balanced_accuracy", "sensitivity_adc", "specificity",
                                  "mcc", "accuracy", "f1", "precision")})
            fold_rows.append(row)

        p_cal = df.p_adc_calibrated.to_numpy(dtype=float)
        p_raw = df.p_adc_raw.to_numpy(dtype=float)
        thr_pooled = float(np.mean(df.threshold))
        pooled_thresholded = df.pred_at_threshold.to_numpy(dtype=int)
        pm = full_metrics(y, p_cal, 0.5)          # placeholder threshold, replaced below
        # threshold-dependent pooled metrics use each patient's OWN fold threshold
        tp = int(((pooled_thresholded == 1) & (y == 1)).sum())
        fn = int(((pooled_thresholded == 0) & (y == 1)).sum())
        tn = int(((pooled_thresholded == 0) & (y == 0)).sum())
        fp = int(((pooled_thresholded == 1) & (y == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
        denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc = float((tp * tn - fp * fn) / denom) if denom > 0 else float("nan")
        legacy = patient_metrics(y, p_cal, 0.5)
        ci_, cs_ = calibration_intercept_slope(y, p_cal)
        fold_auc = [r["roc_auc"] for r in fold_rows if r["model"] == m]
        fold_pr = [r["pr_auc"] for r in fold_rows if r["model"] == m]
        pooled[m] = {
            "roc_auc": float(roc_auc_score(y, p_cal)),
            "pr_auc": float(average_precision_score(y, p_cal)),
            "roc_auc_uncalibrated": float(roc_auc_score(y, p_raw)),
            "pr_auc_uncalibrated": float(average_precision_score(y, p_raw)),
            "balanced_accuracy": 0.5 * (sens + spec),
            "sensitivity_adc": sens, "specificity": spec, "precision": prec,
            "f1": f1, "mcc": mcc,
            "accuracy": float((pooled_thresholded == y).mean()),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "brier": brier(y, p_cal),
            "calibration_intercept": ci_, "calibration_slope": cs_,
            "mean_inner_selected_threshold": thr_pooled,
            "fold_roc_auc_mean_sd": mean_sd(fold_auc),
            "fold_pr_auc_mean_sd": mean_sd(fold_pr),
            "legacy_threshold_0.5": {kk: legacy[kk] for kk in
                                     ("balanced_accuracy", "sensitivity_adc", "specificity",
                                      "precision", "f1", "mcc", "accuracy",
                                      "tp", "fp", "tn", "fn")},
        }
        del pm

        c = curve_points(y, p_cal)
        pd.DataFrame({"fpr": c["fpr"], "tpr": c["tpr"]}).to_csv(
            os.path.join(rd, "roc_curves", "%s.csv" % m), index=False)
        pd.DataFrame({"recall": c["recall"], "precision": c["precision"]}).to_csv(
            os.path.join(rd, "pr_curves", "%s.csv" % m), index=False)
        with open(os.path.join(rd, "confusion_matrices", "%s.json" % m), "w",
                  encoding="utf-8") as f:
            json.dump({"model": m, "threshold": "inner-selected, per fold",
                       "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                       "legacy_0.5": {kk: legacy[kk] for kk in ("tp", "fp", "tn", "fn")}},
                      f, indent=1)

    fm = pd.DataFrame(fold_rows)
    fm.to_csv(os.path.join(rd, "fold_metrics.csv"), index=False)

    # ---- secondary mRMR ----------------------------------------------------
    secondary = {}
    sdir = os.path.join(rd, "secondary_mrmr")
    for fn in sorted(os.listdir(sdir)) if os.path.isdir(sdir) else []:
        if not fn.endswith("_oof.csv"):
            continue
        df = pd.read_csv(os.path.join(sdir, fn)).sort_values("PatientID").reset_index(drop=True)
        tag = fn[:-len("_oof.csv")]
        secondary[tag] = {
            "roc_auc": float(roc_auc_score(df.label, df.p_adc_calibrated)),
            "pr_auc": float(average_precision_score(df.label, df.p_adc_calibrated)),
            "roc_auc_uncalibrated": float(roc_auc_score(df.label, df.p_adc_raw)),
            "fold_roc_auc": {int(k): float(roc_auc_score(g.label, g.p_adc_calibrated))
                             for k, g in df.groupby("fold") if g.label.nunique() > 1},
            "role": "SECONDARY - never determines Gate 7A"}

    # ---- paired bootstrap --------------------------------------------------
    vec = {m: preds[m].p_adc_calibrated.to_numpy(dtype=float) for m in PRIMARY}
    for name, b in baselines.items():
        vec[name] = b["p"]
    boots = {}
    bcfg = cfg["evaluation"]["bootstrap"]
    for comp in bcfg["comparisons"]:
        a, b = comp["a"], comp["b"]
        boots[comp["tag"]] = paired_bootstrap(y, vec[a], vec[b],
                                              bcfg["n_resamples"], comp["seed"],
                                              float(bcfg["ci"]))
        boots[comp["tag"]]["a"] = a
        boots[comp["tag"]]["b"] = b
    with open(os.path.join(rd, "paired_bootstrap", "comparisons.json"), "w",
              encoding="utf-8") as f:
        json.dump(boots, f, indent=1)

    # ---- Gate 7A, applied mechanically -------------------------------------
    a3 = fm[fm.model == "A3"].set_index("fold").roc_auc
    a1 = fm[fm.model == "A1"].set_index("fold").roc_auc
    a2 = fm[fm.model == "A2"].set_index("fold").roc_auc
    beats = {int(k): bool(a3[k] > a1[k] and a3[k] > a2[k]) for k in sorted(a3.index)}
    n_beats = int(sum(beats.values()))
    c1 = bool(pooled["A3"]["roc_auc"] >= 0.63)
    c2 = bool(n_beats >= 3)
    gate = {"frozen_rule": ("A3 pooled OOF ROC-AUC >= 0.63 AND A3 beats BOTH A1 and A2 "
                            "in at least 3 of 5 outer folds"),
            "condition_1_pooled_auc": {"value": pooled["A3"]["roc_auc"],
                                       "threshold": 0.63, "pass": c1},
            "condition_2_folds": {"folds_beating_both": beats, "n": n_beats,
                                  "required": 3, "of": 5, "pass": c2},
            "result": "PASS" if (c1 and c2) else "FAIL",
            "consequence": (cfg["gate_7a"]["on_pass"] if (c1 and c2)
                            else cfg["gate_7a"]["on_fail"])}

    out = {"phase": "7A", "n_patients": int(len(y)), "n_adc": int(y.sum()),
           "n_scc": int((y == 0).sum()), "adc_prevalence": float(y.mean()),
           "primary_pooled_probability": cfg["evaluation"]["pooled_probability"],
           "pooled": pooled, "baselines": {k: {"roc_auc": v["roc_auc"],
                                               "pr_auc": v["pr_auc"],
                                               "path": v["path"]}
                                           for k, v in baselines.items()},
           "secondary_mrmr": secondary, "paired_bootstrap": boots, "gate_7a": gate,
           "evaluation_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(rd, "oof_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)

    print("\npooled out-of-fold (202 patients, calibrated):")
    for m in PRIMARY:
        p = pooled[m]
        print("  %-3s ROC-AUC %.4f  PR-AUC %.4f  fold mean %.3f +/- %.3f  bal.acc %.4f  Brier %.4f"
              % (m, p["roc_auc"], p["pr_auc"], p["fold_roc_auc_mean_sd"]["mean"],
                 p["fold_roc_auc_mean_sd"]["sd"], p["balanced_accuracy"], p["brier"]))
    print("\nGate 7A: %s  (pooled %.4f >= 0.63 -> %s ; folds beating both = %d/5 -> %s)"
          % (gate["result"], pooled["A3"]["roc_auc"], c1, n_beats, c2))
    print("-> %s" % os.path.join(rd, "oof_metrics.json"))


if __name__ == "__main__":
    main()
