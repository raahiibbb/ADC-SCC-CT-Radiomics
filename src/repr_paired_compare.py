"""Phase 6B - representation comparison: frozen CNN embeddings vs handcrafted radiomics.

Both sides produce ONE out-of-fold probability per patient over the SAME frozen
202-patient GTV+Rim cohort and the SAME frozen outer split, so the comparison is
genuinely paired at patient level and needs no fold alignment.

NOTHING is retrained here.  The Phase-5 and Phase-5C out-of-fold prediction files
are read from disk and verified byte-identical against the pre-Phase-6B snapshot
before they are used.

Two PRIMARY comparisons are same-architecture - identical pooling operator,
identical splits, identical preprocessing rule, identical training settings - so
the REPRESENTATION is the only changed variable:

    A.  Phase-6B CNN Mean MIL       vs  Phase-5 radiomics Mean MIL
    B.  Phase-6B CNN Attention MIL  vs  Phase-5 radiomics Attention MIL

One SECONDARY benchmark compares the better Phase-6B model against the
numerically best handcrafted pipeline in the project (Phase-5C mRMR + attention
MIL, pooled OOF ROC-AUC 0.5929).  Representation AND pipeline differ there, so it
is labelled a benchmark and never described as a same-architecture comparison.

Uncertainty is a PATIENT-level paired bootstrap: every resample draws patients
with replacement and scores BOTH prediction vectors on exactly the same sampled
PatientIDs.  Patches are never resampled.

    python src/repr_paired_compare.py --config config/phase6b.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mil_data import load_phase3_config, model_tag, ppath  # noqa: E402
from mil_metrics import mean_sd, patient_metrics  # noqa: E402

MODELS = ("mean_mil", "attention_mil")
METRIC_KEYS = ("roc_auc", "pr_auc", "balanced_accuracy", "sensitivity_adc",
               "specificity", "precision", "f1", "mcc", "accuracy")
BOOTSTRAP_KEYS = ("roc_auc", "pr_auc")
PRETTY = {"mean_mil": "CNN Mean MIL", "attention_mil": "CNN Attention MIL"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------- loading
def read_oof(path: str, expected: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df["PatientID"].duplicated().any():
        raise RuntimeError("duplicate out-of-fold predictions in %s" % path)
    if len(df) != expected:
        raise RuntimeError("%s has %d rows, expected %d" % (path, len(df), expected))
    if not np.all(np.isfinite(df["predicted_probability_ADC"].values)):
        raise RuntimeError("non-finite predicted probabilities in %s" % path)
    return df.set_index("PatientID").sort_index()


def verify_frozen(root: str, rel: str) -> str:
    """Byte-identity of a frozen baseline prediction file against the snapshot."""
    snap_path = os.path.join(root, "reports", "phase6b_pre_snapshot.json")
    with open(snap_path, "r", encoding="utf-8") as fh:
        snap = json.load(fh)["files"]
    got = sha256_file(os.path.join(root, *rel.split("/")))
    want = snap.get(rel)
    if want is None:
        raise RuntimeError("%s is not in the pre-Phase-6B snapshot" % rel)
    if got != want:
        raise RuntimeError("frozen baseline %s CHANGED (%s != %s)" % (rel, got, want))
    return got


# ------------------------------------------------------------------- bootstrap
def paired_bootstrap(y: np.ndarray, p_new: np.ndarray, p_ref: np.ndarray,
                     pids: List[str], n_resamples: int, seed: int, ci: float) -> dict:
    """Patient-level paired bootstrap of (new - reference).

    One resample = one draw of PATIENT indices with replacement, used to index
    BOTH prediction vectors, so every resample scores the same patients on both
    sides.  Resamples that lose a class are skipped and counted.
    """
    rng = np.random.RandomState(int(seed))
    n = y.size
    fn = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score}
    stats = {k: {"new": [], "reference": [], "difference": []} for k in BOOTSTRAP_KEYS}
    n_skipped = 0
    first_draw = None

    for r in range(int(n_resamples)):
        idx = rng.randint(0, n, size=n)          # PATIENTS, with replacement
        if r == 0:
            first_draw = [pids[i] for i in idx[:10]]
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
           "resampling_unit": "patient", "n_patients": int(n), "paired": True,
           "first_resample_first_10_patient_ids": first_draw,
           "note": ("each resample draws PATIENTS with replacement and scores BOTH "
                    "prediction vectors on exactly the same sampled PatientIDs; "
                    "patches are never resampled"),
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
            "percent_of_resamples_favouring_new": float(100.0 * (d > 0).mean()),
            "new_mean": float(np.mean(stats[k]["new"])),
            "reference_mean": float(np.mean(stats[k]["reference"])),
        }
    return out


def compare_pair(new: pd.DataFrame, ref: pd.DataFrame, thr: float, expected: int,
                 n_resamples: int, seed: int, ci: float) -> dict:
    pids = sorted(set(new.index) & set(ref.index))
    if len(pids) != expected:
        raise RuntimeError("%d paired patients, expected %d" % (len(pids), expected))
    y_new = new.loc[pids, "true_label"].values.astype(int)
    y_ref = ref.loc[pids, "true_label"].values.astype(int)
    if not np.array_equal(y_new, y_ref):
        raise RuntimeError("label disagreement between the paired prediction files")
    p_new = new.loc[pids, "predicted_probability_ADC"].values.astype(float)
    p_ref = ref.loc[pids, "predicted_probability_ADC"].values.astype(float)
    m_new = patient_metrics(y_new, p_new, thr)
    m_ref = patient_metrics(y_ref, p_ref, thr)
    return {
        "n_paired_patients": len(pids),
        "n_adc": int((y_new == 1).sum()), "n_scc": int((y_new == 0).sum()),
        "new": m_new, "reference": m_ref,
        "difference_new_minus_reference": {k: m_new[k] - m_ref[k] for k in METRIC_KEYS},
        "spearman_probability_agreement": float(
            pd.Series(p_new).corr(pd.Series(p_ref), method="spearman")),
        "same_label_at_threshold_fraction": float(
            ((p_new >= thr).astype(int) == (p_ref >= thr).astype(int)).mean()),
        "bootstrap": paired_bootstrap(y_new, p_new, p_ref, pids,
                                      n_resamples, seed, ci),
    }


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 6B representation comparison + patient-level paired bootstrap")
    ap.add_argument("--config", default="config/phase6b.yaml")
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_phase3_config(os.path.join(root, *args.config.split("/")))
    rc = cfg["representation_comparison"]
    thr = float(cfg["evaluation"]["threshold"])
    expected = int(rc["expected_paired_patients"])
    n_res = int(rc["bootstrap"]["n_resamples"])
    ci = float(rc["bootstrap"]["ci"])
    res_dir = ppath(cfg, cfg["outputs"]["results_dir"])
    os.makedirs(res_dir, exist_ok=True)

    pred_dir = ppath(cfg, cfg["outputs"]["predictions_dir"])
    new_all = {m: read_oof(os.path.join(pred_dir, "%s_oof.csv" % model_tag(cfg, m)),
                           expected) for m in MODELS}

    summary: Dict[str, object] = {
        "phase": "6b",
        "new_name": str(rc["new_name"]),
        "cohort": {"n_patients": expected,
                   "n_adc": int((new_all["mean_mil"]["true_label"] == 1).sum()),
                   "n_scc": int((new_all["mean_mil"]["true_label"] == 0).sum())},
        "threshold": thr, "positive_class": "ADC (label 1)",
        "merge": "by PatientID over the same frozen 202-patient GTV+Rim cohort",
        "nothing_retrained": True,
        "primary_same_architecture": {}, "secondary_benchmark": {},
        "frozen_reference_sha256": {},
    }

    rows: List[dict] = []

    # ------------------------------------------------------------ pooled tables
    pooled = {}
    for m in MODELS:
        pooled[m] = patient_metrics(new_all[m]["true_label"].values,
                                    new_all[m]["predicted_probability_ADC"].values, thr)
    for k in METRIC_KEYS:
        rows.append({"scope": "phase6b_pooled_oof", "comparison": "pooling",
                     "metric": k, "value_mean_pooling": pooled["mean_mil"][k],
                     "value_attention_pooling": pooled["attention_mil"][k],
                     "difference_attention_minus_mean":
                         pooled["attention_mil"][k] - pooled["mean_mil"][k],
                     "n_patients": expected})
    for m in MODELS:
        ft = [patient_metrics(sub["true_label"].values,
                              sub["predicted_probability_ADC"].values, thr)["roc_auc"]
              for _, sub in new_all[m].groupby("outer_fold")]
        ms = mean_sd(ft)
        rows.append({"scope": "phase6b_fold_mean", "comparison": "pooling",
                     "metric": "roc_auc", "model": m,
                     "fold_mean": ms["mean"], "fold_sd": ms["sd"], "n_patients": expected})

    # --------------------------------------------------------- primary + secondary
    for group, entries in (("primary_same_architecture", rc["primary"]),
                           ("secondary_benchmark", rc["secondary"])):
        for spec in entries:
            rel = str(spec["reference_predictions"])
            sha = verify_frozen(root, rel)
            summary["frozen_reference_sha256"][rel] = sha
            ref = read_oof(os.path.join(root, *rel.split("/")), expected)

            if group == "primary_same_architecture":
                model = str(spec["model"])
                new = new_all[model]
                new_label = PRETTY[model]
            else:
                # the better Phase-6B model by pooled OOF ROC-AUC
                model = max(MODELS, key=lambda m: pooled[m]["roc_auc"])
                new = new_all[model]
                new_label = "%s (best Phase-6B model)" % PRETTY[model]

            cmp_ = compare_pair(new, ref, thr, expected, n_res, int(spec["seed"]), ci)
            cmp_.update({
                "key": str(spec["key"]), "label": str(spec["label"]),
                "phase6b_model": model, "phase6b_label": new_label,
                "reference_name": str(spec["reference_name"]),
                "reference_predictions": rel,
                "reference_sha256": sha,
                "same_architecture": group == "primary_same_architecture",
                "note": ("identical pooling operator, folds, preprocessing rule and "
                         "training settings; the representation is the only changed "
                         "variable"
                         if group == "primary_same_architecture" else
                         "NOT a same-architecture comparison: representation AND "
                         "pipeline differ (the reference adds mRMR feature selection). "
                         "Report as a best-handcrafted-pipeline benchmark only."),
            })
            if "reference_pooled_roc_auc" in spec:
                declared = float(spec["reference_pooled_roc_auc"])
                got = float(cmp_["reference"]["roc_auc"])
                cmp_["reference_pooled_roc_auc_declared"] = declared
                cmp_["reference_pooled_roc_auc_recomputed"] = got
                if abs(declared - got) > 5e-4:
                    raise RuntimeError("declared reference ROC-AUC %.4f != recomputed %.4f"
                                       % (declared, got))
            summary[group][str(spec["key"])] = cmp_

            for k in METRIC_KEYS:
                rows.append({
                    "scope": group, "comparison": str(spec["key"]),
                    "label": str(spec["label"]), "metric": k,
                    "phase6b_model": model,
                    "value_phase6b": cmp_["new"][k],
                    "value_reference": cmp_["reference"][k],
                    "reference_name": str(spec["reference_name"]),
                    "difference_phase6b_minus_reference":
                        cmp_["difference_new_minus_reference"][k],
                    "same_architecture": group == "primary_same_architecture",
                    "n_patients": expected})
            for k in BOOTSTRAP_KEYS:
                b = cmp_["bootstrap"]["metrics"][k]
                rows.append({
                    "scope": "%s_bootstrap" % group, "comparison": str(spec["key"]),
                    "label": str(spec["label"]), "metric": k,
                    "phase6b_model": model,
                    "value_phase6b": cmp_["new"][k],
                    "value_reference": cmp_["reference"][k],
                    "reference_name": str(spec["reference_name"]),
                    "difference_phase6b_minus_reference":
                        cmp_["difference_new_minus_reference"][k],
                    "bootstrap_difference_mean": b["difference_mean"],
                    "ci_low": b["ci_low"], "ci_high": b["ci_high"],
                    "ci_includes_zero": b["ci_includes_zero"],
                    "percent_resamples_favouring_cnn":
                        b["percent_of_resamples_favouring_new"],
                    "bootstrap_seed": int(spec["seed"]),
                    "same_architecture": group == "primary_same_architecture",
                    "n_patients": expected})

    # ------------------------------------------------------------- small bags
    small_thr = int(cfg["evaluation"]["small_bag_threshold"])
    base = new_all["mean_mil"]
    small = sorted(base.index[base["n_patches"] < small_thr].tolist())
    p5 = {m: read_oof(os.path.join(root, "predictions",
                                   "gtv_rim_%s_oof.csv" % m), expected) for m in MODELS}
    for m in MODELS:
        verify_frozen(root, "predictions/gtv_rim_%s_oof.csv" % m)
    sb_rows = []
    for pid in small:
        r = {"PatientID": pid,
             "histology": str(base.loc[pid, "histology"]),
             "true_label": int(base.loc[pid, "true_label"]),
             "bag_size": int(base.loc[pid, "n_patches"]),
             "outer_fold": int(base.loc[pid, "outer_fold"])}
        for m, tag in (("mean_mil", "cnn_mean"), ("attention_mil", "cnn_attention")):
            p = float(new_all[m].loc[pid, "predicted_probability_ADC"])
            r["%s_oof_probability_ADC" % tag] = p
            r["%s_predicted_label" % tag] = int(p >= thr)
            r["%s_correct" % tag] = bool(int(p >= thr) == r["true_label"])
        for m, tag in (("mean_mil", "phase5_mean"), ("attention_mil", "phase5_attention")):
            p = float(p5[m].loc[pid, "predicted_probability_ADC"])
            r["%s_oof_probability_ADC" % tag] = p
            r["%s_predicted_label" % tag] = int(p >= thr)
            r["%s_correct" % tag] = bool(int(p >= thr) == r["true_label"])
        sb_rows.append(r)
    sb = pd.DataFrame(sb_rows).sort_values("bag_size")
    sb.to_csv(os.path.join(res_dir, "small_bag_predictions.csv"), index=False)
    summary["small_bags"] = {
        "threshold_patches": small_thr, "n_patients": len(sb),
        "n_correct_cnn_mean": int(sb["cnn_mean_correct"].sum()),
        "n_correct_cnn_attention": int(sb["cnn_attention_correct"].sum()),
        "n_correct_phase5_mean": int(sb["phase5_mean_correct"].sum()),
        "n_correct_phase5_attention": int(sb["phase5_attention_correct"].sum()),
        "rows": sb.to_dict(orient="records"),
        "note": "reported separately, never excluded from training or evaluation",
    }

    # --------------------------------------------------------------- persistence
    comp_csv = ppath(cfg, rc["outputs"]["comparison_csv"])
    boot_json = ppath(cfg, rc["outputs"]["bootstrap_json"])
    pd.DataFrame(rows).to_csv(comp_csv, index=False)
    with open(boot_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print("Phase 6B pooled OOF (202 patients):")
    for m in MODELS:
        print("  %-14s ROC-AUC %.4f  PR-AUC %.4f  balacc %.4f  MCC %.4f"
              % (m, pooled[m]["roc_auc"], pooled[m]["pr_auc"],
                 pooled[m]["balanced_accuracy"], pooled[m]["mcc"]))
    for group in ("primary_same_architecture", "secondary_benchmark"):
        for key, c in summary[group].items():
            print("%s / %s  [%s]" % (group, key, c["label"]))
            print("    CNN %.4f vs reference %.4f  dROC %+.4f"
                  % (c["new"]["roc_auc"], c["reference"]["roc_auc"],
                     c["difference_new_minus_reference"]["roc_auc"]))
            for k in BOOTSTRAP_KEYS:
                b = c["bootstrap"]["metrics"][k]
                print("      %-7s d %+.4f  95%% CI [%+.4f, %+.4f]  includes 0: %s  "
                      "favouring CNN %.1f%%"
                      % (k, b["difference_mean"], b["ci_low"], b["ci_high"],
                         b["ci_includes_zero"],
                         b["percent_of_resamples_favouring_new"]))
    print("wrote %s" % os.path.relpath(comp_csv, root))
    print("wrote %s" % os.path.relpath(boot_json, root))
    print("wrote %s" % os.path.relpath(os.path.join(res_dir,
                                                    "small_bag_predictions.csv"), root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
