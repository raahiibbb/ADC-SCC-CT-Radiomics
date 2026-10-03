"""Phase 3 - patient-level evaluation of the two MIL models.

Consumes only `predictions/<model>_oof.csv` (194 out-of-fold patient predictions
per model) and `reports/phase3_training_run.json`.  Recomputes every metric from
the stored predictions rather than trusting the training run.

Produces per-fold metrics, pooled out-of-fold metrics, ROC/PR curves, confusion
matrices and the mean-vs-attention comparison.  No interpretation of attention.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mil_data import load_phase3_config, model_tag, ppath  # noqa: E402
from mil_metrics import curve_points, mean_sd, patient_metrics  # noqa: E402

MODELS = ("mean_mil", "attention_mil")
FOLD_METRIC_KEYS = ("roc_auc", "pr_auc", "balanced_accuracy", "sensitivity_adc",
                    "specificity", "precision", "f1", "mcc", "accuracy")
LABELS = {"mean_mil": "Mean-pooling MIL", "attention_mil": "Gated attention MIL"}


def model_suffix(cfg: dict) -> str:
    """Phase 3 writes `<model>_oof.csv`; Phase 3B writes `<model>_3b_oof.csv`."""
    return str(cfg["outputs"].get("model_suffix", ""))


def results_tag(cfg: dict, model_name: str) -> str:
    """File stem inside results_dir.

    Phase 3 / 3B share one `results/` root, so their files carry the full model
    tag.  A phase with its own results directory (Phase 5 -> `results/phase5/`)
    may set `outputs.results_plain_model_names: true` and use the bare model
    name, since the directory already provides the namespace.
    """
    explicit = cfg["outputs"].get("results_model_names") or {}
    if model_name in explicit:
        return str(explicit[model_name])
    if bool(cfg["outputs"].get("results_plain_model_names", False)):
        return model_name
    return model_tag(cfg, model_name)


def n_expected(cfg: dict) -> int:
    return int(cfg["phase3"]["expected_patients"])


def load_predictions(cfg: dict) -> dict:
    pred_dir = ppath(cfg, cfg["outputs"]["predictions_dir"])
    out = {}
    for m in MODELS:
        df = pd.read_csv(os.path.join(pred_dir, "%s_oof.csv" % model_tag(cfg, m)))
        if len(df) != int(cfg["phase3"]["expected_patients"]):
            raise RuntimeError("%s has %d rows, expected %d"
                               % (m, len(df), cfg["phase3"]["expected_patients"]))
        if df["PatientID"].duplicated().any():
            raise RuntimeError("duplicate out-of-fold predictions in %s" % m)
        out[m] = df.sort_values("PatientID").reset_index(drop=True)
    return out


def fold_table(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    rows = []
    for fold, sub in df.groupby("outer_fold"):
        m = patient_metrics(sub["true_label"].values,
                            sub["predicted_probability_ADC"].values, thr)
        m["fold"] = int(fold)
        rows.append(m)
    cols = ["fold", "n_patients", "n_adc", "n_scc"] + list(FOLD_METRIC_KEYS) + \
           ["tp", "fp", "tn", "fn", "threshold"]
    return pd.DataFrame(rows)[cols].sort_values("fold").reset_index(drop=True)


def plot_curves(cfg: dict, preds: dict, thr: float) -> None:
    res = ppath(cfg, cfg["outputs"]["results_dir"])
    roc_dir, pr_dir, cm_dir = (os.path.join(res, d) for d in
                               ("roc_curves", "pr_curves", "confusion_matrices"))
    for d in (roc_dir, pr_dir, cm_dir):
        os.makedirs(d, exist_ok=True)

    fig_r, ax_r = plt.subplots(figsize=(5.2, 5))
    fig_p, ax_p = plt.subplots(figsize=(5.2, 5))
    for m in MODELS:
        df = preds[m]
        y, p = df["true_label"].values, df["predicted_probability_ADC"].values
        c = curve_points(y, p)
        stats = patient_metrics(y, p, thr)
        pd.DataFrame({"fpr": c["fpr"], "tpr": c["tpr"]}).to_csv(
            os.path.join(roc_dir, "%s_oof_roc.csv" % results_tag(cfg, m)), index=False)
        pd.DataFrame({"recall": c["recall"], "precision": c["precision"]}).to_csv(
            os.path.join(pr_dir, "%s_oof_pr.csv" % results_tag(cfg, m)), index=False)

        ax_r.plot(c["fpr"], c["tpr"], label="%s (AUC %.3f)" % (LABELS[m], stats["roc_auc"]))
        ax_p.plot(c["recall"], c["precision"],
                  label="%s (AP %.3f)" % (LABELS[m], stats["pr_auc"]))

        f1, a1 = plt.subplots(figsize=(4.6, 4.4))
        a1.plot(c["fpr"], c["tpr"], color="C0")
        a1.plot([0, 1], [0, 1], "k--", lw=0.8)
        a1.set_xlabel("1 - specificity")
        a1.set_ylabel("sensitivity (ADC)")
        a1.set_title("%s - OOF ROC (AUC %.3f)" % (LABELS[m], stats["roc_auc"]))
        f1.tight_layout()
        f1.savefig(os.path.join(roc_dir, "%s_oof_roc.png" % results_tag(cfg, m)), dpi=150)
        plt.close(f1)

        f2, a2 = plt.subplots(figsize=(4.6, 4.4))
        a2.plot(c["recall"], c["precision"], color="C1")
        prev = float((y == 1).mean())
        a2.axhline(prev, color="k", ls="--", lw=0.8, label="prevalence %.3f" % prev)
        a2.set_xlabel("recall (ADC)")
        a2.set_ylabel("precision")
        a2.set_ylim(0, 1)
        a2.set_title("%s - OOF PR (AP %.3f)" % (LABELS[m], stats["pr_auc"]))
        a2.legend(fontsize=8)
        f2.tight_layout()
        f2.savefig(os.path.join(pr_dir, "%s_oof_pr.png" % results_tag(cfg, m)), dpi=150)
        plt.close(f2)

        cm = np.array([[stats["tn"], stats["fp"]], [stats["fn"], stats["tp"]]])
        f3, a3 = plt.subplots(figsize=(4.2, 4))
        a3.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                a3.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="black", fontsize=14)
        a3.set_xticks([0, 1], ["pred SCC", "pred ADC"])
        a3.set_yticks([0, 1], ["true SCC", "true ADC"])
        a3.set_title("%s - OOF (thr %.2f)" % (LABELS[m], thr))
        f3.tight_layout()
        f3.savefig(os.path.join(cm_dir, "%s_oof_confusion.png" % results_tag(cfg, m)),
                   dpi=150)
        plt.close(f3)

    ax_r.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax_r.set_xlabel("1 - specificity")
    ax_r.set_ylabel("sensitivity (ADC)")
    ax_r.set_title("Out-of-fold ROC, %d patients" % n_expected(cfg))
    ax_r.legend(fontsize=8, loc="lower right")
    fig_r.tight_layout()
    fig_r.savefig(os.path.join(roc_dir, "comparison_oof_roc.png"), dpi=150)
    plt.close(fig_r)

    prev = float((preds["mean_mil"]["true_label"] == 1).mean())
    ax_p.axhline(prev, color="k", ls="--", lw=0.8, label="prevalence %.3f" % prev)
    ax_p.set_xlabel("recall (ADC)")
    ax_p.set_ylabel("precision")
    ax_p.set_ylim(0, 1)
    ax_p.set_title("Out-of-fold precision-recall, %d patients" % n_expected(cfg))
    ax_p.legend(fontsize=8)
    fig_p.tight_layout()
    fig_p.savefig(os.path.join(pr_dir, "comparison_oof_pr.png"), dpi=150)
    plt.close(fig_p)


def main() -> int:
    ap = argparse.ArgumentParser(description="patient-level evaluation of the two MIL models")
    ap.add_argument("--config", default=None,
                    help="config to evaluate (default config/phase3.yaml; "
                         "pass config/phase3b.yaml for the Phase-3B run)")
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = (load_phase3_config(os.path.join(root, *args.config.split("/")))
           if args.config else load_phase3_config())
    thr = float(cfg["evaluation"]["threshold"])
    res_dir = ppath(cfg, cfg["outputs"]["results_dir"])
    os.makedirs(res_dir, exist_ok=True)
    preds = load_predictions(cfg)

    run_path = ppath(cfg, str(cfg["outputs"].get("run_json",
                                                 "reports/phase3_training_run.json")))
    run = json.load(open(run_path, "r", encoding="utf-8")) if os.path.exists(run_path) else {}

    summary = {"threshold": thr, "positive_class": "ADC (label 1)", "models": {},
               "comparison": {}, "small_bags": {}, "runtime": {}}
    fold_tables = {}
    for m in MODELS:
        ft = fold_table(preds[m], thr)
        ft.insert(0, "model", m)
        ft.to_csv(os.path.join(res_dir, "%s_fold_metrics.csv" % results_tag(cfg, m)),
                  index=False)
        fold_tables[m] = ft

        oof = patient_metrics(preds[m]["true_label"].values,
                              preds[m]["predicted_probability_ADC"].values, thr)
        across = {k: mean_sd(ft[k].values) for k in FOLD_METRIC_KEYS}
        summary["models"][m] = {
            "pooled_oof": oof,
            "across_folds_mean_sd": across,
            "per_fold": ft.to_dict(orient="records"),
            "confusion_matrix_oof": {"tn": oof["tn"], "fp": oof["fp"],
                                     "fn": oof["fn"], "tp": oof["tp"]},
        }

    # ------------------------------------------------------------- comparison
    rows = []
    for k in FOLD_METRIC_KEYS:
        a = summary["models"]["attention_mil"]["pooled_oof"][k]
        b = summary["models"]["mean_mil"]["pooled_oof"][k]
        rows.append({"metric": k, "scope": "pooled_oof", "mean_mil": b,
                     "attention_mil": a, "difference_attention_minus_mean": a - b})
    for k in FOLD_METRIC_KEYS:
        a = summary["models"]["attention_mil"]["across_folds_mean_sd"][k]
        b = summary["models"]["mean_mil"]["across_folds_mean_sd"][k]
        rows.append({"metric": k, "scope": "fold_mean",
                     "mean_mil": b["mean"], "attention_mil": a["mean"],
                     "difference_attention_minus_mean": a["mean"] - b["mean"],
                     "mean_mil_sd": b["sd"], "attention_mil_sd": a["sd"]})
    for fold in sorted(fold_tables["mean_mil"]["fold"]):
        for k in ("roc_auc", "pr_auc", "balanced_accuracy", "mcc"):
            b = float(fold_tables["mean_mil"].set_index("fold").loc[fold, k])
            a = float(fold_tables["attention_mil"].set_index("fold").loc[fold, k])
            rows.append({"metric": k, "scope": "fold_%d" % fold, "mean_mil": b,
                         "attention_mil": a, "difference_attention_minus_mean": a - b})
    comp = pd.DataFrame(rows)
    # A phase that publishes its own combined comparison table (Phase 6B) points
    # this key elsewhere so the two files cannot overwrite each other.
    comp.to_csv(os.path.join(res_dir, str(cfg["outputs"].get(
        "model_comparison_csv", "model_comparison.csv"))), index=False)

    fa = fold_tables["attention_mil"].set_index("fold")["roc_auc"]
    fb = fold_tables["mean_mil"].set_index("fold")["roc_auc"]
    summary["comparison"] = {
        "pooled_oof_roc_auc": {"mean_mil": summary["models"]["mean_mil"]["pooled_oof"]["roc_auc"],
                               "attention_mil": summary["models"]["attention_mil"]["pooled_oof"]["roc_auc"]},
        "differences_attention_minus_mean_pooled_oof": {
            k: (summary["models"]["attention_mil"]["pooled_oof"][k]
                - summary["models"]["mean_mil"]["pooled_oof"][k]) for k in FOLD_METRIC_KEYS},
        "folds_where_attention_better_roc_auc": [int(f) for f in fa.index if fa[f] > fb[f]],
        "folds_where_mean_better_roc_auc": [int(f) for f in fa.index if fb[f] > fa[f]],
        "attention_consistently_better_roc_auc": bool(all(fa[f] > fb[f] for f in fa.index)),
    }

    # ------------------------------------------------------ small bags (< 20 patches)
    small_thr = int(cfg["evaluation"]["small_bag_threshold"])
    small = preds["mean_mil"][preds["mean_mil"]["n_patches"] < small_thr]["PatientID"].tolist()
    small_rows = []
    for m in MODELS:
        sub = preds[m][preds[m]["PatientID"].isin(small)]
        for _, r in sub.iterrows():
            small_rows.append({"model": m, "PatientID": r["PatientID"],
                               "histology": r["histology"], "n_patches": int(r["n_patches"]),
                               "outer_fold": int(r["outer_fold"]),
                               "true_label": int(r["true_label"]),
                               "predicted_probability_ADC": float(r["predicted_probability_ADC"]),
                               "predicted_label": int(r["predicted_label"]),
                               "correct": bool(int(r["predicted_label"]) == int(r["true_label"]))})
    small_df = pd.DataFrame(small_rows).sort_values(["model", "n_patches"])
    # Optional (Phase 5): flag which small bags belong to the paired shared subset.
    # Off by default, so the Phase-3 and Phase-3B tables are byte-identical.
    ov_rel = str(cfg.get("roi_comparison", {}).get("overlap_file", ""))         if bool(cfg["evaluation"].get("small_bag_annotate_shared", False)) else ""
    if ov_rel:
        ov = pd.read_csv(ppath(cfg, ov_rel)).set_index("PatientID")
        shared = set(ov.index[ov["membership"] == "shared"])
        small_df["in_shared_194_subset"] = small_df["PatientID"].isin(shared)
        small_df["reference_patches"] = small_df["PatientID"].map(
            ov["reference_patches"]).astype("Int64")
    small_df.to_csv(os.path.join(res_dir, str(cfg["outputs"].get(
        "small_bag_csv", "small_bag_predictions.csv"))), index=False)
    summary["small_bags"] = {
        "threshold_patches": small_thr,
        "patients": small,
        "n_patients": len(small),
        "rows": small_df.to_dict(orient="records"),
        "note": "reported separately, never excluded from training or evaluation",
    }

    summary["runtime"] = {k: run.get(k) for k in
                          ("total_runtime_seconds", "training_seconds_by_model",
                           "selection_seconds_by_model", "refit_seconds_by_model")
                          if run.get(k) is not None}
    if "selected_epochs" in run:
        summary["selected_epochs"] = run["selected_epochs"]
    summary["sanity"] = {
        "all_probabilities_finite": bool(all(
            np.isfinite(preds[m]["predicted_probability_ADC"].values).all() for m in MODELS)),
        "n_predictions_per_model": {m: int(len(preds[m])) for m in MODELS},
        "identical_patient_sets": bool(
            preds["mean_mil"]["PatientID"].tolist() == preds["attention_mil"]["PatientID"].tolist()),
        "identical_fold_assignment": bool(
            preds["mean_mil"]["outer_fold"].tolist() == preds["attention_mil"]["outer_fold"].tolist()),
        "identical_labels": bool(
            preds["mean_mil"]["true_label"].tolist() == preds["attention_mil"]["true_label"].tolist()),
    }

    with open(os.path.join(res_dir, "oof_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)
    plot_curves(cfg, preds, thr)

    for m in MODELS:
        o = summary["models"][m]["pooled_oof"]
        fm = summary["models"][m]["across_folds_mean_sd"]
        print("%-14s pooled OOF: ROC-AUC %.4f  PR-AUC %.4f  balacc %.4f  MCC %.4f  "
              "sens %.3f spec %.3f  | fold ROC-AUC %.3f +/- %.3f"
              % (m, o["roc_auc"], o["pr_auc"], o["balanced_accuracy"], o["mcc"],
                 o["sensitivity_adc"], o["specificity"],
                 fm["roc_auc"]["mean"], fm["roc_auc"]["sd"]))
    d = summary["comparison"]["differences_attention_minus_mean_pooled_oof"]
    print("attention - mean (pooled OOF): dROC %.4f dPR %.4f dBalAcc %.4f dMCC %.4f"
          % (d["roc_auc"], d["pr_auc"], d["balanced_accuracy"], d["mcc"]))
    print("attention better in folds %s; mean better in folds %s"
          % (summary["comparison"]["folds_where_attention_better_roc_auc"],
             summary["comparison"]["folds_where_mean_better_roc_auc"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
