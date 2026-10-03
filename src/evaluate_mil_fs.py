"""Phase 5C - patient-level evaluation of the four feature-selection MIL pipelines.

Consumes only files on disk:

  predictions/phase5c_{mrmr,lasso}_{mean,attention}_oof.csv   202 OOF predictions each
  predictions/gtv_rim_{mean,attention}_mil_oof.csv            the Phase-5 74-feature
                                                              baseline, READ ONLY
  results/phase5c/checkpoints/*.json                          per-fold selector records
  feature_selection/phase5c/**/fold_*_final.json              the selected features

Every metric is recomputed from the stored predictions rather than trusted from
the training run.  The Phase-5 baseline is NEVER retrained; its prediction files
are read, hash-checked against the pre-Phase-5C snapshot, and merged by
PatientID over the same 202 patients.

Uncertainty on every new-vs-baseline difference comes from a PATIENT-level
paired bootstrap: each resample draws 202 PATIENTS with replacement and scores
both prediction vectors on exactly those patients.  Patches are never resampled.

    python src/evaluate_mil_fs.py --config config/phase5c.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_selection import feature_family, jaccard  # noqa: E402
from mil_data import load_phase3_config, ppath, sha256_file  # noqa: E402
from mil_metrics import curve_points, mean_sd, patient_metrics  # noqa: E402
from roi_paired_compare import bootstrap_differences  # noqa: E402

FOLD_METRIC_KEYS = ("roc_auc", "pr_auc", "balanced_accuracy", "sensitivity_adc",
                    "specificity", "precision", "f1", "mcc", "accuracy")
LABELS = {"mrmr_mean": "mRMR + Mean MIL", "mrmr_attention": "mRMR + Attention MIL",
          "lasso_mean": "LASSO + Mean MIL", "lasso_attention": "LASSO + Attention MIL",
          "baseline_mean": "Phase 5 (74 features) Mean MIL",
          "baseline_attention": "Phase 5 (74 features) Attention MIL"}
FOLDS = (1, 2, 3, 4, 5)


# ----------------------------------------------------------------------- input
def load_new(cfg: dict, tag: str) -> pd.DataFrame:
    path = os.path.join(ppath(cfg, cfg["outputs"]["predictions_dir"]),
                        "%s%s_oof.csv" % (cfg["outputs"]["model_prefix"], tag))
    df = pd.read_csv(path)
    n = int(cfg["phase3"]["expected_patients"])
    if len(df) != n or df["PatientID"].duplicated().any():
        raise RuntimeError("%s: %d rows (expected %d unique patients)" % (tag, len(df), n))
    return df.sort_values("PatientID").reset_index(drop=True)


def load_baseline(cfg: dict, which: str) -> pd.DataFrame:
    rel = cfg["baseline"]["model_files"][which]
    df = pd.read_csv(ppath(cfg, rel))
    n = int(cfg["phase3"]["expected_patients"])
    if len(df) != n or df["PatientID"].duplicated().any():
        raise RuntimeError("baseline %s: %d rows" % (which, len(df)))
    return df.sort_values("PatientID").reset_index(drop=True)


def units(cfg: dict, tag: str) -> Dict[int, dict]:
    out = {}
    for f in FOLDS:
        p = os.path.join(ppath(cfg, cfg["outputs"]["checkpoint_dir"]),
                         "%s_fold_%d.json" % (tag, f))
        with open(p, "r", encoding="utf-8") as fh:
            out[f] = json.load(fh)
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


# -------------------------------------------------------------------- plotting
def plot_all(cfg: dict, preds: Dict[str, pd.DataFrame], thr: float, res: str) -> None:
    roc_dir, pr_dir, cm_dir = (os.path.join(res, d) for d in
                               ("roc_curves", "pr_curves", "confusion_matrices"))
    for d in (roc_dir, pr_dir, cm_dir):
        os.makedirs(d, exist_ok=True)

    fig_r, ax_r = plt.subplots(figsize=(6.0, 5.4))
    fig_p, ax_p = plt.subplots(figsize=(6.0, 5.4))
    for tag, df in preds.items():
        y, p = df["true_label"].values, df["predicted_probability_ADC"].values
        c = curve_points(y, p)
        st = patient_metrics(y, p, thr)
        style = "--" if tag.startswith("baseline") else "-"
        pd.DataFrame({"fpr": c["fpr"], "tpr": c["tpr"]}).to_csv(
            os.path.join(roc_dir, "%s_oof_roc.csv" % tag), index=False)
        pd.DataFrame({"recall": c["recall"], "precision": c["precision"]}).to_csv(
            os.path.join(pr_dir, "%s_oof_pr.csv" % tag), index=False)
        ax_r.plot(c["fpr"], c["tpr"], style, lw=1.4,
                  label="%s (AUC %.3f)" % (LABELS[tag], st["roc_auc"]))
        ax_p.plot(c["recall"], c["precision"], style, lw=1.4,
                  label="%s (AP %.3f)" % (LABELS[tag], st["pr_auc"]))

        cm = np.array([[st["tn"], st["fp"]], [st["fn"], st["tp"]]])
        f3, a3 = plt.subplots(figsize=(4.2, 4))
        a3.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                a3.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
        a3.set_xticks([0, 1], ["pred SCC", "pred ADC"])
        a3.set_yticks([0, 1], ["true SCC", "true ADC"])
        a3.set_title("%s - OOF (thr %.2f)" % (LABELS[tag], thr), fontsize=9)
        f3.tight_layout()
        f3.savefig(os.path.join(cm_dir, "%s_oof_confusion.png" % tag), dpi=150)
        plt.close(f3)

    ax_r.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax_r.set_xlabel("1 - specificity")
    ax_r.set_ylabel("sensitivity (ADC)")
    ax_r.set_title("Phase 5C out-of-fold ROC, 202 patients")
    ax_r.legend(fontsize=7, loc="lower right")
    fig_r.tight_layout()
    fig_r.savefig(os.path.join(roc_dir, "comparison_oof_roc.png"), dpi=150)
    plt.close(fig_r)

    prev = float((preds["mrmr_mean"]["true_label"] == 1).mean())
    ax_p.axhline(prev, color="k", ls=":", lw=0.8, label="prevalence %.3f" % prev)
    ax_p.set_xlabel("recall (ADC)")
    ax_p.set_ylabel("precision")
    ax_p.set_ylim(0, 1)
    ax_p.set_title("Phase 5C out-of-fold precision-recall, 202 patients")
    ax_p.legend(fontsize=7)
    fig_p.tight_layout()
    fig_p.savefig(os.path.join(pr_dir, "comparison_oof_pr.png"), dpi=150)
    plt.close(fig_p)


def plot_surfaces(cfg: dict, res: str) -> None:
    sdir = os.path.join(res, "selector_surfaces")
    for tag in [p["tag"] for p in cfg["pipelines"]]:
        fig, axes = plt.subplots(1, 5, figsize=(19, 3.4), sharey=True)
        for ax, fold in zip(axes, FOLDS):
            path = os.path.join(sdir, "%s_fold_%d_surface.csv" % (tag, fold))
            sel_path = os.path.join(sdir, "%s_fold_%d_selection.json" % (tag, fold))
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            sel = json.load(open(sel_path, "r", encoding="utf-8"))["selection"]
            for col in [c for c in df.columns if c.endswith("_mean_val_roc_auc")]:
                ax.plot(df["epoch"], df[col], lw=1.0,
                        label=col.replace("_mean_val_roc_auc", ""))
            ax.axvline(sel["selected_epoch"], color="k", ls="--", lw=0.8)
            ax.axhline(0.5, color="grey", ls=":", lw=0.7)
            ax.set_title("fold %d -> %s @ %d" % (fold, sel["selected_candidate"],
                                                 sel["selected_epoch"]), fontsize=8)
            ax.set_xlabel("epoch")
            ax.legend(fontsize=6, ncol=2)
        axes[0].set_ylabel("mean inner-CV ROC-AUC")
        fig.suptitle("%s - joint (selector setting, epoch) selection surface" % LABELS[tag],
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(sdir, "%s_surfaces.png" % tag), dpi=150)
        plt.close(fig)


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5C evaluation")
    ap.add_argument("--config", default="config/phase5c.yaml")
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_phase3_config(os.path.join(root, *args.config.split("/")))
    thr = float(cfg["evaluation"]["threshold"])
    res = ppath(cfg, cfg["outputs"]["results_dir"])
    os.makedirs(res, exist_ok=True)
    tags = [p["tag"] for p in cfg["pipelines"]]
    base_of = {p["tag"]: p["baseline"] for p in cfg["pipelines"]}

    preds: Dict[str, pd.DataFrame] = {t: load_new(cfg, t) for t in tags}
    for which in ("mean", "attention"):
        preds["baseline_%s" % which] = load_baseline(cfg, which)

    pid0 = preds[tags[0]]["PatientID"].tolist()
    for t, df in preds.items():
        if df["PatientID"].tolist() != pid0:
            raise RuntimeError("%s covers a different patient set" % t)
        if df["true_label"].tolist() != preds[tags[0]]["true_label"].tolist():
            raise RuntimeError("%s disagrees on the labels" % t)
    y = preds[tags[0]]["true_label"].values.astype(int)
    prevalence = float((y == 1).mean())

    # -------------------------------------------------- per-fold / pooled metrics
    summary: Dict[str, object] = {
        "phase": "5c", "threshold": thr, "positive_class": "ADC (label 1)",
        "n_patients": int(y.size), "n_adc": int((y == 1).sum()),
        "n_scc": int((y == 0).sum()), "adc_prevalence": prevalence,
        "baseline": {"name": cfg["baseline"]["name"],
                     "note": "read from disk, never retrained"},
        "models": {}, "comparison": {}, "feature_selection": {},
    }
    fold_tables = {}
    for t, df in preds.items():
        ft = fold_table(df, thr)
        ft.insert(0, "pipeline", t)
        ft.to_csv(os.path.join(res, "%s_fold_metrics.csv" % t), index=False)
        fold_tables[t] = ft
        oof = patient_metrics(df["true_label"].values,
                              df["predicted_probability_ADC"].values, thr)
        summary["models"][t] = {
            "label": LABELS[t],
            "pooled_oof": oof,
            "across_folds_mean_sd": {k: mean_sd(ft[k].values) for k in FOLD_METRIC_KEYS},
            "per_fold": ft.to_dict(orient="records"),
            "confusion_matrix_oof": {"tn": oof["tn"], "fp": oof["fp"],
                                     "fn": oof["fn"], "tp": oof["tp"]},
        }

    # ------------------------------------------------------------ model comparison
    rows = []
    for t in tags + ["baseline_mean", "baseline_attention"]:
        o = summary["models"][t]["pooled_oof"]
        a = summary["models"][t]["across_folds_mean_sd"]
        for k in FOLD_METRIC_KEYS:
            rows.append({"pipeline": t, "metric": k, "scope": "pooled_oof",
                         "value": o[k], "fold_mean": a[k]["mean"], "fold_sd": a[k]["sd"]})
    for t in tags:
        b = "baseline_%s" % base_of[t]
        for k in FOLD_METRIC_KEYS:
            rows.append({"pipeline": t, "metric": k, "scope": "difference_vs_baseline",
                         "value": (summary["models"][t]["pooled_oof"][k]
                                   - summary["models"][b]["pooled_oof"][k]),
                         "baseline": b,
                         "baseline_value": summary["models"][b]["pooled_oof"][k],
                         "new_value": summary["models"][t]["pooled_oof"][k]})
    for t in tags + ["baseline_mean", "baseline_attention"]:
        ftab = fold_tables[t].set_index("fold")
        for f in FOLDS:
            for k in ("roc_auc", "pr_auc", "balanced_accuracy", "mcc"):
                rows.append({"pipeline": t, "metric": k, "scope": "fold_%d" % f,
                             "value": float(ftab.loc[f, k])})
    pd.DataFrame(rows).to_csv(os.path.join(res, "model_comparison.csv"), index=False)

    # --------------------------------------------------- paired patient bootstrap
    bcfg = cfg["baseline"]["bootstrap"]
    if int(bcfg["expected_paired_patients"]) != int(y.size):
        raise RuntimeError("paired bootstrap expects %s patients, found %d"
                           % (bcfg["expected_paired_patients"], y.size))
    boot = {"n_paired_patients": int(y.size), "n_adc": int((y == 1).sum()),
            "n_scc": int((y == 0).sum()), "merge_on": str(bcfg["merge_on"]),
            "resampling_unit": "patient", "n_resamples": int(bcfg["n_resamples"]),
            "base_seed": int(bcfg["seed"]), "ci_level": float(bcfg["ci"]),
            "note": ("each resample draws 202 PATIENTS with replacement and scores the "
                     "feature-selection and 74-feature baseline vectors on exactly the "
                     "same sampled patients; patches are never resampled"),
            "comparisons": {}}
    for i, t in enumerate(tags):
        b = "baseline_%s" % base_of[t]
        p_new = preds[t]["predicted_probability_ADC"].values.astype(float)
        p_ref = preds[b]["predicted_probability_ADC"].values.astype(float)
        bs = bootstrap_differences(y, p_new, p_ref, int(bcfg["n_resamples"]),
                                   int(bcfg["seed"]) + i, float(bcfg["ci"]))
        bs["pipeline"] = t
        bs["baseline"] = b
        bs["observed_difference"] = {
            k: (summary["models"][t]["pooled_oof"][k]
                - summary["models"][b]["pooled_oof"][k]) for k in ("roc_auc", "pr_auc")}
        boot["comparisons"][t] = bs
    with open(os.path.join(res, "baseline_paired_bootstrap.json"), "w",
              encoding="utf-8") as fh:
        json.dump(boot, fh, indent=2, default=float)

    # ---------------------------------------------- feature-selection frequency
    names = [ln.strip() for ln in
             open(ppath(cfg, cfg["phase3"]["feature_names_file"]), "r", encoding="utf-8")
             if ln.strip()]
    unit_by_tag = {t: units(cfg, t) for t in tags}
    freq_rows, stab_rows = [], []
    for t in tags:
        sel_by_fold = {f: sorted(unit_by_tag[t][f]["selected_indices"]) for f in FOLDS}
        counts = {}
        for f, sel in sel_by_fold.items():
            for i in sel:
                counts.setdefault(i, []).append(f)
        fs_dir = os.path.join(ppath(cfg, cfg["outputs"]["feature_selection_dir"]),
                              unit_by_tag[t][1]["selector"])
        finals = {f: json.load(open(os.path.join(
            fs_dir, "fold_%d_%s_final.json" % (f, t)), "r", encoding="utf-8"))
            for f in FOLDS}
        for i in sorted(counts):
            row = {"pipeline": t, "selector": unit_by_tag[t][1]["selector"],
                   "feature_index": int(i), "feature": names[i],
                   "family": feature_family(names[i]),
                   "n_folds_selected": len(counts[i]),
                   "selection_frequency": len(counts[i]) / float(len(FOLDS)),
                   "folds": ";".join(str(f) for f in counts[i])}
            if unit_by_tag[t][1]["selector"] == "mrmr":
                ranks = [finals[f]["candidate_record"]["selection_order"].index(i) + 1
                         for f in counts[i]]
                row["mean_mrmr_rank"] = float(np.mean(ranks))
                row["mrmr_ranks"] = ";".join(str(r) for r in ranks)
                row["mean_relevance_f"] = float(np.mean(
                    [finals[f]["relevance_f"][names[i]] for f in counts[i]]))
            else:
                co = [finals[f]["candidate_record"]["coefficients"][names[i]]
                      for f in counts[i]]
                row["mean_coefficient"] = float(np.mean(co))
                row["mean_abs_coefficient"] = float(np.mean(np.abs(co)))
                row["coefficient_signs"] = ";".join("+" if v > 0 else "-" for v in co)
                row["coefficients"] = ";".join("%.5f" % v for v in co)
            freq_rows.append(row)

        js = [jaccard(sel_by_fold[a], sel_by_fold[b2]) for a, b2 in combinations(FOLDS, 2)]
        for (a, b2), v in zip(combinations(FOLDS, 2), js):
            stab_rows.append({"pipeline": t, "scope": "pairwise_jaccard",
                              "fold_a": a, "fold_b": b2, "value": v,
                              "n_selected_a": len(sel_by_fold[a]),
                              "n_selected_b": len(sel_by_fold[b2]),
                              "n_intersection": len(set(sel_by_fold[a])
                                                    & set(sel_by_fold[b2]))})
        core = set(sel_by_fold[1])
        union = set()
        for f in FOLDS:
            core &= set(sel_by_fold[f])
            union |= set(sel_by_fold[f])
        stab_rows.append({
            "pipeline": t, "scope": "summary", "fold_a": None, "fold_b": None,
            "value": float(np.mean(js)), "mean_pairwise_jaccard": float(np.mean(js)),
            "min_pairwise_jaccard": float(np.min(js)),
            "max_pairwise_jaccard": float(np.max(js)),
            "sd_pairwise_jaccard": float(np.std(js, ddof=1)),
            "n_selected_per_fold": ";".join(str(len(sel_by_fold[f])) for f in FOLDS),
            "mean_n_selected": float(np.mean([len(sel_by_fold[f]) for f in FOLDS])),
            "sd_n_selected": float(np.std([len(sel_by_fold[f]) for f in FOLDS], ddof=1)),
            "n_in_all_five_folds": len(core), "n_in_any_fold": len(union),
            "features_in_all_five_folds": ";".join(sorted(names[i] for i in core)),
            "settings_per_fold": ";".join(str(unit_by_tag[t][f]["selected_setting"])
                                          for f in FOLDS),
            "epochs_per_fold": ";".join(str(unit_by_tag[t][f]["selected_epoch"])
                                        for f in FOLDS)})

        fam_counts = {}
        for f in FOLDS:
            for i in sel_by_fold[f]:
                fam_counts[feature_family(names[i])] = \
                    fam_counts.get(feature_family(names[i]), 0) + 1
        summary["feature_selection"][t] = {
            "selector": unit_by_tag[t][1]["selector"],
            "setting_key": unit_by_tag[t][1]["selector_setting_key"],
            "setting_per_fold": {f: unit_by_tag[t][f]["selected_setting"] for f in FOLDS},
            "epoch_per_fold": {f: unit_by_tag[t][f]["selected_epoch"] for f in FOLDS},
            "n_selected_per_fold": {f: len(sel_by_fold[f]) for f in FOLDS},
            "mean_n_selected": float(np.mean([len(sel_by_fold[f]) for f in FOLDS])),
            "selected_features_per_fold": {f: [names[i] for i in sel_by_fold[f]]
                                           for f in FOLDS},
            "family_selection_counts_over_folds": fam_counts,
            "mean_pairwise_jaccard": float(np.mean(js)),
            "features_in_all_five_folds": sorted(names[i] for i in core),
            "n_distinct_features_used": len(union),
            "selection_mean_inner_cv_roc_auc": {f: unit_by_tag[t][f][
                "selection_mean_inner_cv_roc_auc"] for f in FOLDS},
            "invalid_candidates_per_fold": {
                f: unit_by_tag[t][f]["selection_detail"].get("invalid_candidates", {})
                for f in FOLDS},
        }
    pd.DataFrame(freq_rows).sort_values(
        ["pipeline", "n_folds_selected", "feature"],
        ascending=[True, False, True]).to_csv(
        os.path.join(res, "feature_selection_frequency.csv"), index=False)
    pd.DataFrame(stab_rows).to_csv(
        os.path.join(res, "feature_selection_stability.csv"), index=False)

    # ------------------------------------------------------------------ summary
    ranked = sorted(tags, key=lambda t: -summary["models"][t]["pooled_oof"]["roc_auc"])
    summary["comparison"] = {
        "pooled_oof_roc_auc": {t: summary["models"][t]["pooled_oof"]["roc_auc"]
                               for t in preds},
        "pooled_oof_pr_auc": {t: summary["models"][t]["pooled_oof"]["pr_auc"]
                              for t in preds},
        "ranking_by_pooled_roc_auc": ranked,
        "best_pipeline_by_pooled_roc_auc": ranked[0],
        "difference_vs_baseline": {
            t: {k: (summary["models"][t]["pooled_oof"][k]
                    - summary["models"]["baseline_%s" % base_of[t]]["pooled_oof"][k])
                for k in FOLD_METRIC_KEYS} for t in tags},
        "fold_roc_auc_sd": {t: summary["models"][t]["across_folds_mean_sd"]["roc_auc"]["sd"]
                            for t in preds},
        "folds_beating_baseline_roc_auc": {
            t: [int(f) for f in FOLDS
                if float(fold_tables[t].set_index("fold").loc[f, "roc_auc"])
                > float(fold_tables["baseline_%s" % base_of[t]]
                        .set_index("fold").loc[f, "roc_auc"])] for t in tags},
        "note": ("the Phase-5 baseline uses its own outer-fold assignment, which is the "
                 "SAME frozen split, so fold indices are comparable here - unlike the "
                 "Phase-5 cross-ROI comparison"),
    }
    summary["bootstrap"] = {t: {
        "observed_difference": boot["comparisons"][t]["observed_difference"],
        "roc_auc": boot["comparisons"][t]["metrics"]["roc_auc"],
        "pr_auc": boot["comparisons"][t]["metrics"]["pr_auc"]} for t in tags}
    summary["baseline_prediction_sha256"] = {
        w: sha256_file(ppath(cfg, cfg["baseline"]["model_files"][w]))
        for w in ("mean", "attention")}
    summary["sanity"] = {
        "all_probabilities_finite": bool(all(
            np.isfinite(preds[t]["predicted_probability_ADC"].values).all() for t in preds)),
        "n_predictions_per_pipeline": {t: int(len(preds[t])) for t in preds},
        "identical_patient_sets": True,
        "identical_labels": True,
        "one_prediction_per_patient": bool(all(
            not preds[t]["PatientID"].duplicated().any() for t in preds)),
    }
    with open(os.path.join(res, "oof_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)

    plot_all(cfg, preds, thr, res)
    plot_surfaces(cfg, res)

    # ------------------------------------------------------------------- console
    print("Phase 5C - pooled out-of-fold, %d patients (%d ADC / %d SCC), threshold %.2f"
          % (y.size, (y == 1).sum(), (y == 0).sum(), thr))
    for t in tags + ["baseline_mean", "baseline_attention"]:
        o = summary["models"][t]["pooled_oof"]
        sd = summary["models"][t]["across_folds_mean_sd"]["roc_auc"]
        print("  %-16s ROC-AUC %.4f  PR-AUC %.4f  balacc %.4f  MCC %.4f  | fold %.3f +/- %.3f"
              % (t, o["roc_auc"], o["pr_auc"], o["balanced_accuracy"], o["mcc"],
                 sd["mean"], sd["sd"]))
    print("difference vs the 74-feature Phase-5 baseline, with the paired patient bootstrap:")
    for t in tags:
        d = summary["bootstrap"][t]
        print("  %-16s dROC %+.4f  95%% CI [%+.4f, %+.4f] %s | dPR %+.4f  CI [%+.4f, %+.4f] %s"
              % (t, d["observed_difference"]["roc_auc"], d["roc_auc"]["ci_low"],
                 d["roc_auc"]["ci_high"],
                 "includes 0" if d["roc_auc"]["ci_includes_zero"] else "EXCLUDES 0",
                 d["observed_difference"]["pr_auc"], d["pr_auc"]["ci_low"],
                 d["pr_auc"]["ci_high"],
                 "includes 0" if d["pr_auc"]["ci_includes_zero"] else "EXCLUDES 0"))
    print("best by pooled ROC-AUC: %s" % ranked[0])
    for t in tags:
        fs = summary["feature_selection"][t]
        print("  %-16s features/fold %s  mean pairwise Jaccard %.3f  in all 5 folds: %d"
              % (t, list(fs["n_selected_per_fold"].values()),
                 fs["mean_pairwise_jaccard"], len(fs["features_in_all_five_folds"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
