"""Phase 6C - POST-HOC paired comparison of the two FROZEN Phase-6B poolings.

Phase 6B reported that gated attention beat mean pooling by +0.0656 pooled OOF
ROC-AUC and in 4 of 5 folds, but deliberately reported NO paired test of that
difference because none had been pre-specified.  This script supplies that test
now, and it is therefore **POST-HOC**: the comparison was chosen after seeing the
Phase-6B result and carries no confirmatory weight.  It is descriptive evidence
about how large the pooling difference is relative to its uncertainty, nothing
more.

NOTHING is retrained.  Both Phase-6B out-of-fold prediction files are read from
disk and verified byte-identical against reports/phase6c_pre_snapshot.json.

Uncertainty is a PATIENT-level paired bootstrap over the 202 patients: every
resample draws patients with replacement and scores BOTH prediction vectors on
exactly the same sampled PatientIDs.  Patches are never resampled.

    python src/phase6c_pooling_bootstrap.py --config config/phase6c.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mil_data import load_phase3_config, ppath, sha256_file  # noqa: E402
from mil_metrics import patient_metrics  # noqa: E402
from repr_paired_compare import paired_bootstrap, read_oof  # noqa: E402

METRIC_KEYS = ("roc_auc", "pr_auc", "balanced_accuracy", "sensitivity_adc",
               "specificity", "precision", "f1", "mcc", "accuracy")
SNAPSHOT = "reports/phase6c_pre_snapshot.json"


def verify_frozen(root: str, rel: str) -> str:
    """Byte-identity of a frozen Phase-6B prediction file against the snapshot."""
    with open(os.path.join(root, *SNAPSHOT.split("/")), "r", encoding="utf-8") as fh:
        snap = json.load(fh)["files"]
    got = sha256_file(os.path.join(root, *rel.split("/")))
    want = snap.get(rel)
    if want is None:
        raise RuntimeError("%s is not in the pre-Phase-6C snapshot" % rel)
    if got != want:
        raise RuntimeError("frozen Phase-6B baseline %s CHANGED (%s != %s)"
                           % (rel, got, want))
    return got


def main() -> int:
    ap = argparse.ArgumentParser(
        description="POST-HOC paired bootstrap: Phase-6B attention vs Phase-6B mean")
    ap.add_argument("--config", default="config/phase6c.yaml")
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_phase3_config(os.path.join(root, *args.config.split("/")))
    spec = cfg["comparison"]["post_hoc_phase6b_pooling"]
    boot = cfg["comparison"]["bootstrap"]
    thr = float(cfg["evaluation"]["threshold"])
    expected = int(cfg["comparison"]["expected_paired_patients"])

    rel_new = str(spec["new_predictions"])
    rel_ref = str(spec["reference_predictions"])
    sha_new = verify_frozen(root, rel_new)
    sha_ref = verify_frozen(root, rel_ref)

    new = read_oof(os.path.join(root, *rel_new.split("/")), expected)
    ref = read_oof(os.path.join(root, *rel_ref.split("/")), expected)
    pids = sorted(set(new.index) & set(ref.index))
    if len(pids) != expected:
        raise RuntimeError("%d paired patients, expected %d" % (len(pids), expected))

    y = new.loc[pids, "true_label"].values.astype(int)
    if not np.array_equal(y, ref.loc[pids, "true_label"].values.astype(int)):
        raise RuntimeError("label disagreement between the two Phase-6B files")
    p_new = new.loc[pids, "predicted_probability_ADC"].values.astype(float)
    p_ref = ref.loc[pids, "predicted_probability_ADC"].values.astype(float)

    m_new = patient_metrics(y, p_new, thr)
    m_ref = patient_metrics(y, p_ref, thr)

    # per-fold difference, descriptive only
    fold_rows = []
    for fold in sorted(new.loc[pids, "outer_fold"].unique()):
        sel = [p for p in pids if int(new.loc[p, "outer_fold"]) == int(fold)]
        a = patient_metrics(new.loc[sel, "true_label"].values.astype(int),
                            new.loc[sel, "predicted_probability_ADC"].values.astype(float),
                            thr)
        b = patient_metrics(ref.loc[sel, "true_label"].values.astype(int),
                            ref.loc[sel, "predicted_probability_ADC"].values.astype(float),
                            thr)
        fold_rows.append({"fold": int(fold), "n_patients": len(sel),
                          "attention_roc_auc": a["roc_auc"],
                          "mean_roc_auc": b["roc_auc"],
                          "difference_roc_auc": a["roc_auc"] - b["roc_auc"],
                          "attention_pr_auc": a["pr_auc"],
                          "mean_pr_auc": b["pr_auc"],
                          "difference_pr_auc": a["pr_auc"] - b["pr_auc"]})

    bs = paired_bootstrap(y, p_new, p_ref, pids, int(boot["n_resamples"]),
                          int(spec["seed"]), float(boot["ci"]))

    out = {
        "phase": "6c",
        "analysis": "POST-HOC",
        "post_hoc": True,
        "pre_specified": False,
        "post_hoc_statement": (
            "This comparison was NOT pre-specified in Phase 6B.  It was chosen "
            "after seeing that gated attention beat mean pooling by +0.0656 pooled "
            "OOF ROC-AUC and in 4 of 5 folds.  It is therefore exploratory and "
            "carries no confirmatory weight; no p-value is reported and the "
            "interval must not be described as establishing significance."),
        "label": str(spec["label"]),
        "nothing_retrained": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frozen_inputs": {
            rel_new: sha_new, rel_ref: sha_ref,
            "verified_against": SNAPSHOT,
        },
        "cohort": {"n_paired_patients": len(pids),
                   "n_adc": int((y == 1).sum()), "n_scc": int((y == 0).sum()),
                   "prevalence_adc": float((y == 1).mean())},
        "threshold": thr,
        "positive_class": "ADC (label 1)",
        "new_label": str(spec["new_label"]),
        "reference_label": str(spec["reference_label"]),
        "new": m_new,
        "reference": m_ref,
        "difference_attention_minus_mean": {k: m_new[k] - m_ref[k] for k in METRIC_KEYS},
        "spearman_probability_agreement": float(
            pd.Series(p_new).corr(pd.Series(p_ref), method="spearman")),
        "same_label_at_threshold_fraction": float(
            ((p_new >= thr).astype(int) == (p_ref >= thr).astype(int)).mean()),
        "per_fold": fold_rows,
        "n_folds_favouring_attention": int(sum(r["difference_roc_auc"] > 0
                                               for r in fold_rows)),
        "bootstrap": bs,
    }

    res_dir = ppath(cfg, cfg["outputs"]["results_dir"])
    os.makedirs(res_dir, exist_ok=True)
    dest = os.path.join(root, *str(spec["output"]).split("/"))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    d_roc = bs["metrics"]["roc_auc"]
    d_pr = bs["metrics"]["pr_auc"]
    pct = "%"
    print("POST-HOC  Phase-6B attention vs Phase-6B mean  (%d paired patients, "
          "%d resamples, seed %d)"
          % (len(pids), int(boot["n_resamples"]), int(spec["seed"])))
    print("  observed dROC-AUC %+.4f   bootstrap mean %+.4f   95%s CI [%+.4f, %+.4f]  "
          "includes 0: %s   favouring attention %.1f%s"
          % (m_new["roc_auc"] - m_ref["roc_auc"], d_roc["difference_mean"], pct,
             d_roc["ci_low"], d_roc["ci_high"], d_roc["ci_includes_zero"],
             100.0 * d_roc["fraction_of_resamples_favouring_new"], pct))
    print("  observed dPR-AUC  %+.4f   bootstrap mean %+.4f   95%s CI [%+.4f, %+.4f]  "
          "includes 0: %s   favouring attention %.1f%s"
          % (m_new["pr_auc"] - m_ref["pr_auc"], d_pr["difference_mean"], pct,
             d_pr["ci_low"], d_pr["ci_high"], d_pr["ci_includes_zero"],
             100.0 * d_pr["fraction_of_resamples_favouring_new"], pct))
    print("  wrote %s" % str(spec["output"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
