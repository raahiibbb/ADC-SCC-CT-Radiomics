"""Phase 5 - paired ROI comparison: Gradient (Phase 3B) vs GTV+Rim (Phase 5).

Both experiments produce ONE out-of-fold probability per patient from their own
independent outer split.  The two splits are NOT aligned and must never be:
fold indices are meaningless across experiments and fold-wise differences are
never averaged.  The comparison therefore merges the two out-of-fold prediction
vectors BY PatientID over the patients shared by both cohorts, and evaluates the
paired vectors on exactly that shared set.

Uncertainty comes from a PATIENT-level bootstrap: each resample draws patients
with replacement and scores BOTH ROI prediction vectors on the same sampled
patients, so the ROI difference is paired within every resample.  Patches are
never resampled.

Nothing is retrained and no radiomics is recomputed - only the two stored
prediction files are read.

    python src/roi_paired_compare.py --config config/phase5.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mil_data import load_phase3_config, model_tag, ppath  # noqa: E402
from mil_metrics import patient_metrics  # noqa: E402

MODELS = ("mean_mil", "attention_mil")
REPORT_KEYS = ("roc_auc", "pr_auc", "balanced_accuracy", "sensitivity_adc",
               "specificity", "f1", "mcc")
DIFF_KEYS = ("roc_auc", "pr_auc", "balanced_accuracy", "mcc")
BOOTSTRAP_KEYS = ("roc_auc", "pr_auc")


# ---------------------------------------------------------------------- loading
def load_oof(cfg: dict, model: str) -> pd.DataFrame:
    path = os.path.join(ppath(cfg, cfg["outputs"]["predictions_dir"]),
                        "%s_oof.csv" % model_tag(cfg, model))
    df = pd.read_csv(path)
    if df["PatientID"].duplicated().any():
        raise RuntimeError("duplicate out-of-fold predictions in %s" % path)
    return df.set_index("PatientID").sort_index()


def shared_patients(cfg: dict, new: pd.DataFrame, ref: pd.DataFrame) -> List[str]:
    """Shared set derived from the prediction files, cross-checked against the
    frozen Phase-4 overlap report."""
    shared = sorted(set(new.index) & set(ref.index))
    overlap_path = ppath(cfg, cfg["roi_comparison"]["overlap_file"])
    ov = pd.read_csv(overlap_path)
    declared = sorted(ov.loc[ov["membership"] == "shared", "PatientID"].tolist())
    if shared != declared:
        raise RuntimeError("shared set (%d) disagrees with %s (%d)"
                           % (len(shared), overlap_path, len(declared)))
    expected = int(cfg["roi_comparison"]["expected_shared_patients"])
    if len(shared) != expected:
        raise RuntimeError("%d shared patients, expected %d" % (len(shared), expected))
    return shared


# ------------------------------------------------------------------- bootstrap
def bootstrap_differences(y: np.ndarray, p_new: np.ndarray, p_ref: np.ndarray,
                          n_resamples: int, seed: int, ci: float) -> dict:
    """Patient-level paired bootstrap of (new - reference) for ROC-AUC and PR-AUC.

    One resample = one draw of PATIENT indices with replacement, scored on both
    prediction vectors.  Resamples that lose a class are skipped and counted.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.RandomState(int(seed))
    n = y.size
    stats = {k: {"new": [], "reference": [], "difference": []} for k in BOOTSTRAP_KEYS}
    n_skipped = 0
    fn = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score}

    for _ in range(int(n_resamples)):
        idx = rng.randint(0, n, size=n)          # PATIENTS, with replacement
        yy = y[idx]
        if np.unique(yy).size < 2:
            n_skipped += 1
            continue
        a, b = p_new[idx], p_ref[idx]
        for k in BOOTSTRAP_KEYS:
            va, vb = float(fn[k](yy, a)), float(fn[k](yy, b))
            stats[k]["new"].append(va)
            stats[k]["reference"].append(vb)
            stats[k]["difference"].append(va - vb)

    lo_q, hi_q = (1.0 - ci) / 2.0 * 100.0, (1.0 + ci) / 2.0 * 100.0
    out = {"n_resamples_requested": int(n_resamples),
           "n_resamples_used": int(n_resamples) - n_skipped,
           "n_resamples_skipped_single_class": int(n_skipped),
           "seed": int(seed), "ci_level": float(ci),
           "resampling_unit": "patient", "paired": True,
           "note": ("each resample draws PATIENTS with replacement and scores both ROI "
                    "prediction vectors on the same sampled patients; patches are never "
                    "resampled"),
           "metrics": {}}
    for k in BOOTSTRAP_KEYS:
        d = np.asarray(stats[k]["difference"], dtype=float)
        lo, hi = float(np.percentile(d, lo_q)), float(np.percentile(d, hi_q))
        out["metrics"][k] = {
            "difference_mean": float(d.mean()),
            "difference_median": float(np.median(d)),
            "difference_sd": float(d.std(ddof=1)),
            "ci_low": lo, "ci_high": hi,
            "ci_includes_zero": bool(lo <= 0.0 <= hi),
            "ci_excludes_zero": bool(not (lo <= 0.0 <= hi)),
            "fraction_of_resamples_favouring_new": float((d > 0).mean()),
            "new_mean": float(np.mean(stats[k]["new"])),
            "reference_mean": float(np.mean(stats[k]["reference"])),
        }
    return out


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description="paired Gradient vs GTV+Rim ROI comparison")
    ap.add_argument("--config", default="config/phase5.yaml")
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_phase3_config(os.path.join(root, *args.config.split("/")))
    rc = cfg["roi_comparison"]
    ref_cfg = load_phase3_config(os.path.join(root, *rc["reference_config"].split("/")))
    thr = float(cfg["evaluation"]["threshold"])

    new_all = {m: load_oof(cfg, m) for m in MODELS}
    ref_all = {m: load_oof(ref_cfg, m) for m in MODELS}

    shared = shared_patients(cfg, new_all["mean_mil"], ref_all["mean_mil"])
    for m in MODELS:
        if sorted(set(new_all[m].index) & set(ref_all[m].index)) != shared:
            raise RuntimeError("model %s does not cover the same shared patients" % m)
        ln = new_all[m].loc[shared, "true_label"].values
        lr = ref_all[m].loc[shared, "true_label"].values
        if not np.array_equal(ln, lr):
            raise RuntimeError("label disagreement between the two ROI experiments (%s)" % m)

    y = new_all["mean_mil"].loc[shared, "true_label"].values.astype(int)
    n_adc, n_scc = int((y == 1).sum()), int((y == 0).sum())

    rows: List[dict] = []
    summary: Dict[str, object] = {
        "shared_patients": len(shared), "n_adc": n_adc, "n_scc": n_scc,
        "threshold": thr, "positive_class": "ADC (label 1)",
        "reference": rc["reference_name"], "new": "gtv_rim_phase5",
        "fold_alignment": ("NONE - the two outer splits are independent; predictions are "
                           "merged by PatientID and fold indices are never compared"),
        "models": {}, "bootstrap": {},
    }

    for m in MODELS:
        p_new = new_all[m].loc[shared, "predicted_probability_ADC"].values.astype(float)
        p_ref = ref_all[m].loc[shared, "predicted_probability_ADC"].values.astype(float)
        m_new = patient_metrics(y, p_new, thr)
        m_ref = patient_metrics(y, p_ref, thr)
        for k in REPORT_KEYS:
            rows.append({"model": m, "metric": k,
                         "gradient_phase3b": m_ref[k], "gtv_rim_phase5": m_new[k],
                         "difference_gtv_rim_minus_gradient": m_new[k] - m_ref[k],
                         "n_patients": len(shared), "scope": "paired_shared_oof"})
        summary["models"][m] = {
            "gradient_phase3b": m_ref, "gtv_rim_phase5": m_new,
            "difference_gtv_rim_minus_gradient": {k: m_new[k] - m_ref[k] for k in DIFF_KEYS},
            "spearman_probability_agreement": float(
                pd.Series(p_new).corr(pd.Series(p_ref), method="spearman")),
            "same_label_at_threshold_fraction": float(
                ((p_new >= thr).astype(int) == (p_ref >= thr).astype(int)).mean()),
        }

        bs = bootstrap_differences(y, p_new, p_ref,
                                   int(rc["bootstrap"]["n_resamples"]),
                                   int(rc["bootstrap"]["seed"]) + (0 if m == "mean_mil" else 1),
                                   float(rc["bootstrap"]["ci"]))
        summary["bootstrap"][m] = bs
        for k in BOOTSTRAP_KEYS:
            b = bs["metrics"][k]
            rows.append({"model": m, "metric": "%s_bootstrap_ci" % k,
                         "gradient_phase3b": b["reference_mean"],
                         "gtv_rim_phase5": b["new_mean"],
                         "difference_gtv_rim_minus_gradient": b["difference_mean"],
                         "ci_low": b["ci_low"], "ci_high": b["ci_high"],
                         "ci_includes_zero": b["ci_includes_zero"],
                         "n_patients": len(shared), "scope": "patient_bootstrap"})

    out_dir = ppath(cfg, cfg["outputs"]["results_dir"])
    os.makedirs(out_dir, exist_ok=True)
    paired_csv = ppath(cfg, rc["outputs"]["paired_csv"])
    boot_json = ppath(cfg, rc["outputs"]["bootstrap_json"])
    pd.DataFrame(rows).to_csv(paired_csv, index=False)
    with open(boot_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print("paired comparison on %d shared patients (%d ADC / %d SCC)"
          % (len(shared), n_adc, n_scc))
    for m in MODELS:
        s = summary["models"][m]
        print("  %-14s Gradient-3B ROC %.4f PR %.4f | GTV+Rim ROC %.4f PR %.4f | "
              "dROC %+.4f dPR %+.4f dBalAcc %+.4f dMCC %+.4f"
              % (m, s["gradient_phase3b"]["roc_auc"], s["gradient_phase3b"]["pr_auc"],
                 s["gtv_rim_phase5"]["roc_auc"], s["gtv_rim_phase5"]["pr_auc"],
                 s["difference_gtv_rim_minus_gradient"]["roc_auc"],
                 s["difference_gtv_rim_minus_gradient"]["pr_auc"],
                 s["difference_gtv_rim_minus_gradient"]["balanced_accuracy"],
                 s["difference_gtv_rim_minus_gradient"]["mcc"]))
        for k in BOOTSTRAP_KEYS:
            b = summary["bootstrap"][m]["metrics"][k]
            print("      %-7s difference %+.4f  95%% CI [%+.4f, %+.4f]  includes 0: %s"
                  % (k, b["difference_mean"], b["ci_low"], b["ci_high"],
                     b["ci_includes_zero"]))
    print("wrote %s" % os.path.relpath(paired_csv, root))
    print("wrote %s" % os.path.relpath(boot_json, root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
