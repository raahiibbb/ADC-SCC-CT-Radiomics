"""Phase 6C - patient-level evaluation, paired comparison and diagnostics.

Consumes only files on disk:

  * `predictions/phase6c_clam_style_attention_oof.csv`  - the 202 Phase-6C OOF rows
  * `predictions/phase6b_cnn_attention_oof.csv`        - the FROZEN Phase-6B baseline,
    read and verified byte-identical against `reports/phase6c_pre_snapshot.json`.
    **Nothing on the baseline side is retrained.**
  * `attention/gtv_rim_resnet18_phase6{b,c}/<PID>.npz` - out-of-fold attention weights
  * `results/phase6c/lambda_epoch_selection/*`         - the joint selection surfaces

Every metric is recomputed from the stored predictions rather than trusted from
the training run.  The only comparison made is Phase 6C vs Phase 6B CNN Attention
MIL: same 202 patients, same frozen outer split, same frozen embeddings, same
encoder dimensions, same gated attention, same optimiser, same bag loss, same
preprocessing, same evaluation.  The instance-level auxiliary loss is the only
changed variable.

    python src/evaluate_phase6c.py --config config/phase6c.yaml
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
from clam_mil import MODEL_NAME, bag_k, k_rule_from_config  # noqa: E402
from mil_data import load_cohort, load_phase3_config, model_tag, ppath, sha256_file  # noqa: E402
from mil_metrics import curve_points, mean_sd, patient_metrics  # noqa: E402
from repr_paired_compare import paired_bootstrap, read_oof  # noqa: E402

FOLD_METRIC_KEYS = ("roc_auc", "pr_auc", "balanced_accuracy", "sensitivity_adc",
                    "specificity", "precision", "f1", "mcc", "accuracy")
SNAPSHOT = "reports/phase6c_pre_snapshot.json"
NEW_LABEL = "Phase-6C CLAM-style Attention MIL"
REF_LABEL = "Phase-6B CNN Attention MIL"


# ------------------------------------------------------------------------ loading
def verify_frozen(root: str, rel: str) -> str:
    with open(os.path.join(root, *SNAPSHOT.split("/")), "r", encoding="utf-8") as fh:
        snap = json.load(fh)["files"]
    got = sha256_file(os.path.join(root, *rel.split("/")))
    want = snap.get(rel)
    if want is None:
        raise RuntimeError("%s is not in the pre-Phase-6C snapshot" % rel)
    if got != want:
        raise RuntimeError("frozen Phase-6B artefact %s CHANGED" % rel)
    return got


def fold_table(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    rows = []
    for fold, sub in df.groupby("outer_fold"):
        m = patient_metrics(sub["true_label"].values,
                            sub["predicted_probability_ADC"].values, thr)
        m["fold"] = int(fold)
        if "selected_epoch" in sub.columns:
            m["selected_epoch"] = int(sub["selected_epoch"].iloc[0])
        if "selected_lambda_instance" in sub.columns:
            m["selected_lambda_instance"] = float(sub["selected_lambda_instance"].iloc[0])
        rows.append(m)
    lead = ["fold"]
    if "selected_lambda_instance" in rows[0]:
        lead.append("selected_lambda_instance")
    if "selected_epoch" in rows[0]:
        lead.append("selected_epoch")
    cols = lead + ["n_patients", "n_adc", "n_scc"] + list(FOLD_METRIC_KEYS) + \
        ["tp", "fp", "tn", "fn", "threshold"]
    return pd.DataFrame(rows)[cols].sort_values("fold").reset_index(drop=True)


# ------------------------------------------------------------------------- curves
def plot_curves(res_dir: str, y: np.ndarray, p_new: np.ndarray, p_ref: np.ndarray,
                thr: float) -> None:
    roc_dir, pr_dir, cm_dir = (os.path.join(res_dir, d) for d in
                               ("roc_curves", "pr_curves", "confusion_matrices"))
    for d in (roc_dir, pr_dir, cm_dir):
        os.makedirs(d, exist_ok=True)

    fig_r, ax_r = plt.subplots(figsize=(5.2, 5))
    fig_p, ax_p = plt.subplots(figsize=(5.2, 5))
    prev = float((y == 1).mean())
    for tag, label, p, colour in (("phase6c_clam_attention", NEW_LABEL, p_new, "C0"),
                                  ("phase6b_cnn_attention", REF_LABEL, p_ref, "C3")):
        c = curve_points(y, p)
        s = patient_metrics(y, p, thr)
        pd.DataFrame({"fpr": c["fpr"], "tpr": c["tpr"]}).to_csv(
            os.path.join(roc_dir, "%s_oof_roc.csv" % tag), index=False)
        pd.DataFrame({"recall": c["recall"], "precision": c["precision"]}).to_csv(
            os.path.join(pr_dir, "%s_oof_pr.csv" % tag), index=False)
        ax_r.plot(c["fpr"], c["tpr"], color=colour,
                  label="%s (AUC %.3f)" % (label, s["roc_auc"]))
        ax_p.plot(c["recall"], c["precision"], color=colour,
                  label="%s (AP %.3f)" % (label, s["pr_auc"]))

        f, a = plt.subplots(figsize=(4.4, 4.2))
        cm = np.array([[s["tn"], s["fp"]], [s["fn"], s["tp"]]])
        a.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                a.text(j, i, str(cm[i, j]), ha="center", va="center",
                       color="black", fontsize=13)
        a.set_xticks([0, 1], ["pred SCC", "pred ADC"])
        a.set_yticks([0, 1], ["true SCC", "true ADC"])
        a.set_title("%s\nthreshold %.2f" % (label, thr), fontsize=9)
        f.tight_layout()
        f.savefig(os.path.join(cm_dir, "%s_oof_confusion.png" % tag), dpi=150)
        plt.close(f)
        pd.DataFrame(cm, index=["true_SCC", "true_ADC"],
                     columns=["pred_SCC", "pred_ADC"]).to_csv(
            os.path.join(cm_dir, "%s_oof_confusion.csv" % tag))

    ax_r.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax_r.set_xlabel("1 - specificity")
    ax_r.set_ylabel("sensitivity (ADC)")
    ax_r.set_title("Phase 6C vs Phase 6B - pooled OOF ROC (202 patients)")
    ax_r.legend(fontsize=8)
    fig_r.tight_layout()
    fig_r.savefig(os.path.join(roc_dir, "phase6c_vs_phase6b_oof_roc.png"), dpi=150)
    plt.close(fig_r)

    ax_p.axhline(prev, color="k", ls="--", lw=0.8, label="prevalence %.3f" % prev)
    ax_p.set_xlabel("recall (ADC)")
    ax_p.set_ylabel("precision")
    ax_p.set_title("Phase 6C vs Phase 6B - pooled OOF PR (202 patients)")
    ax_p.legend(fontsize=8)
    fig_p.tight_layout()
    fig_p.savefig(os.path.join(pr_dir, "phase6c_vs_phase6b_oof_pr.png"), dpi=150)
    plt.close(fig_p)


def plot_selection(cfg: dict) -> None:
    sel_dir = ppath(cfg, cfg["outputs"]["selection_dir"])
    if not os.path.isdir(sel_dir):
        return
    grid = sorted(float(v) for v in cfg["instance_supervision"]["lambda_instance_grid"])
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.6), sharey=True)
    for fold, ax in zip(range(1, 6), axes):
        path = os.path.join(sel_dir, "fold_%d_surface.csv" % fold)
        if not os.path.exists(path):
            continue
        s = pd.read_csv(path)
        for lam in grid:
            sub = s[np.isclose(s["lambda_instance"], lam)].sort_values("epoch")
            ax.plot(sub["epoch"], sub["mean_inner_cv_roc_auc"], lw=1.1,
                    label="lambda %.2f" % lam)
        sel = s[s["selected"]]
        if len(sel):
            ax.plot(sel["epoch"], sel["mean_inner_cv_roc_auc"], "k*", ms=11,
                    label="selected")
        ax.axhline(0.5, color="k", ls="--", lw=0.7)
        ax.set_title("outer fold %d" % fold, fontsize=10)
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("mean inner-CV ROC-AUC")
    axes[0].legend(fontsize=8)
    fig.suptitle("Phase 6C - joint (lambda_instance x epoch) selection, 3-fold inner CV",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(sel_dir, "lambda_epoch_surfaces.png"), dpi=150)
    plt.close(fig)


# -------------------------------------------------------------- attention analysis
def bag_stats(alpha: np.ndarray) -> dict:
    n = int(alpha.shape[0])
    top = max(1, n // 10)
    return {"n_patches": n,
            "max_over_uniform": float(alpha.max() * n),
            "max_over_min": float(alpha.max() / alpha.min()),
            "normalized_entropy": float(-(alpha * np.log(alpha)).sum() / np.log(n)),
            "top10pct_mass": float(np.sort(alpha)[::-1][:top].sum())}


def collect_attention(root: str, rel_dir: str) -> pd.DataFrame:
    d = os.path.join(root, *rel_dir.split("/"))
    rows = []
    for fn in sorted(f for f in os.listdir(d) if f.endswith(".npz")):
        with np.load(os.path.join(d, fn), allow_pickle=True) as z:
            a = np.asarray(z["attention"], dtype=np.float64)
            if not np.all(np.isfinite(a)):
                raise RuntimeError("non-finite attention in %s" % fn)
            row = {"PatientID": str(z["patient_id"]), "histology": str(z["histology"]),
                   "true_label": int(z["true_label"]), "outer_fold": int(z["outer_fold"]),
                   "predicted_probability_ADC": float(z["predicted_probability_adc"]),
                   "attention_sum": float(a.sum())}
            row.update(bag_stats(a))
            rows.append(row)
    return pd.DataFrame(rows).sort_values("PatientID").reset_index(drop=True)


KEYS = ("max_over_uniform", "max_over_min", "normalized_entropy", "top10pct_mass")


def attention_report(root: str, cfg: dict, res_dir: str, fold_new: pd.DataFrame) -> dict:
    new = collect_attention(root, cfg["outputs"]["attention_dir"])
    ref = collect_attention(root, str(cfg["comparison"]["primary"]["reference_attention_dir"]))
    new.to_csv(os.path.join(res_dir, "attention_concentration.csv"), index=False)
    merged = new.merge(ref, on="PatientID", suffixes=("_phase6c", "_phase6b"))
    merged.to_csv(os.path.join(res_dir, "attention_concentration_comparison.csv"),
                  index=False)

    sel_epoch = {int(r["fold"]): int(r["selected_epoch"]) for _, r in fold_new.iterrows()}
    sel_lam = {int(r["fold"]): float(r["selected_lambda_instance"])
               for _, r in fold_new.iterrows()}
    test_auc = {int(r["fold"]): float(r["roc_auc"]) for _, r in fold_new.iterrows()}

    by_fold = []
    for f, sub in new.groupby("outer_fold"):
        sub_b = ref[ref["outer_fold"] == f]
        row = {"fold": int(f), "n_bags": int(len(sub)),
               "selected_lambda_instance": sel_lam.get(int(f)),
               "selected_epoch": sel_epoch.get(int(f)),
               "outer_test_roc_auc": test_auc.get(int(f))}
        for k in KEYS:
            row["phase6c_median_%s" % k] = float(sub[k].median())
            row["phase6b_median_%s" % k] = float(sub_b[k].median())
        by_fold.append(row)
    bf = pd.DataFrame(by_fold).sort_values("fold")
    bf.to_csv(os.path.join(res_dir, "attention_concentration_by_fold.csv"), index=False)

    def sp(a, b):
        return float(pd.Series(a).corr(pd.Series(b), method="spearman"))

    return {
        "note": ("Descriptive only.  Sharper attention is NOT assumed to be better; "
                 "Phases 3B, 5 and 6B all found concentration tracking the selected "
                 "epoch rather than outer-test performance.  No spatial or biological "
                 "interpretation is made or implied."),
        "n_bags": int(len(new)),
        "phase6c": {k: {"mean": float(new[k].mean()), "median": float(new[k].median()),
                        "min": float(new[k].min()), "max": float(new[k].max())}
                    for k in KEYS},
        "phase6b": {k: {"mean": float(ref[k].mean()), "median": float(ref[k].median()),
                        "min": float(ref[k].min()), "max": float(ref[k].max())}
                    for k in KEYS},
        "by_fold": by_fold,
        "does_concentration_track_epoch": {
            "spearman_selected_epoch_vs_median_entropy":
                sp(bf["selected_epoch"], bf["phase6c_median_normalized_entropy"]),
            "spearman_selected_epoch_vs_median_top10pct_mass":
                sp(bf["selected_epoch"], bf["phase6c_median_top10pct_mass"]),
            "spearman_median_entropy_vs_outer_test_roc_auc":
                sp(bf["phase6c_median_normalized_entropy"], bf["outer_test_roc_auc"]),
            "spearman_median_top10pct_mass_vs_outer_test_roc_auc":
                sp(bf["phase6c_median_top10pct_mass"], bf["outer_test_roc_auc"]),
            "caveat": "five points; these correlations carry essentially no information",
        },
        "spearman_per_bag_entropy_phase6c_vs_phase6b":
            sp(merged["normalized_entropy_phase6c"], merged["normalized_entropy_phase6b"]),
        "max_abs_attention_sum_minus_one": float(np.abs(new["attention_sum"] - 1.0).max()),
    }


# ------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 6C evaluation and paired comparison")
    ap.add_argument("--config", default="config/phase6c.yaml")
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_phase3_config(os.path.join(root, *args.config.split("/")))
    thr = float(cfg["evaluation"]["threshold"])
    comp = cfg["comparison"]
    prim = comp["primary"]
    expected = int(comp["expected_paired_patients"])
    res_dir = ppath(cfg, cfg["outputs"]["results_dir"])
    os.makedirs(res_dir, exist_ok=True)

    tag = model_tag(cfg, MODEL_NAME)
    new_path = os.path.join(ppath(cfg, cfg["outputs"]["predictions_dir"]),
                            "%s_oof.csv" % tag)
    ref_rel = str(prim["reference_predictions"])
    ref_sha = verify_frozen(root, ref_rel)

    new = read_oof(new_path, expected)
    ref = read_oof(os.path.join(root, *ref_rel.split("/")), expected)
    pids = sorted(set(new.index) & set(ref.index))
    if len(pids) != expected:
        raise RuntimeError("%d paired patients, expected %d" % (len(pids), expected))
    y = new.loc[pids, "true_label"].values.astype(int)
    if not np.array_equal(y, ref.loc[pids, "true_label"].values.astype(int)):
        raise RuntimeError("label disagreement between Phase 6C and Phase 6B")
    p_new = new.loc[pids, "predicted_probability_ADC"].values.astype(float)
    p_ref = ref.loc[pids, "predicted_probability_ADC"].values.astype(float)

    # ---- per fold
    new_r = new.reset_index()
    ref_r = ref.reset_index()
    ft_new = fold_table(new_r, thr)
    ft_ref = fold_table(ref_r, thr)
    ft_new.to_csv(os.path.join(res_dir, "fold_metrics.csv"), index=False)

    fold_cmp = ft_new[["fold", "selected_lambda_instance", "selected_epoch",
                       "roc_auc", "pr_auc", "balanced_accuracy", "mcc"]].merge(
        ft_ref[["fold", "selected_epoch", "roc_auc", "pr_auc",
                "balanced_accuracy", "mcc"]],
        on="fold", suffixes=("_phase6c", "_phase6b"))
    for k in ("roc_auc", "pr_auc", "balanced_accuracy", "mcc"):
        fold_cmp["d_%s" % k] = fold_cmp["%s_phase6c" % k] - fold_cmp["%s_phase6b" % k]
    fold_cmp.to_csv(os.path.join(res_dir, "fold_comparison.csv"), index=False)
    n_folds_better = int((fold_cmp["d_roc_auc"] > 0).sum())

    # ---- pooled
    m_new = patient_metrics(y, p_new, thr)
    m_ref = patient_metrics(y, p_ref, thr)

    # ---- curves + selection figure
    plot_curves(res_dir, y, p_new, p_ref, thr)
    plot_selection(cfg)

    # ---- paired bootstrap
    bs = paired_bootstrap(y, p_new, p_ref, pids, int(comp["bootstrap"]["n_resamples"]),
                          int(prim["seed"]), float(comp["bootstrap"]["ci"]))
    boot_out = {
        "phase": "6c",
        "label": str(prim["label"]),
        "pre_specified": True,
        "changed_variable": "instance-level auxiliary loss only",
        "held_fixed": ["202 patients", "frozen outer split", "frozen 512-D CNN embeddings",
                       "encoder dimensions 512 -> 64 -> 32", "gated attention module",
                       "Adam lr 1e-3 wd 1e-4", "weighted BCE bag loss from patient counts",
                       "variance screen + z-score from training patches only",
                       "threshold 0.5", "3-fold inner CV seed 3000 + fold"],
        "nothing_retrained_on_the_baseline_side": True,
        "frozen_inputs": {ref_rel: ref_sha, "verified_against": SNAPSHOT},
        "cohort": {"n_paired_patients": len(pids), "n_adc": int((y == 1).sum()),
                   "n_scc": int((y == 0).sum()), "prevalence_adc": float((y == 1).mean())},
        "new": m_new, "reference": m_ref,
        "difference_phase6c_minus_phase6b": {k: m_new[k] - m_ref[k]
                                             for k in FOLD_METRIC_KEYS},
        "spearman_probability_agreement": float(
            pd.Series(p_new).corr(pd.Series(p_ref), method="spearman")),
        "same_label_at_threshold_fraction": float(
            ((p_new >= thr).astype(int) == (p_ref >= thr).astype(int)).mean()),
        "n_folds_phase6c_better_roc_auc": n_folds_better,
        "per_fold": fold_cmp.to_dict("records"),
        "bootstrap": bs,
    }
    with open(os.path.join(root, *str(prim["output"]).split("/")), "w",
              encoding="utf-8") as fh:
        json.dump(boot_out, fh, indent=2)

    # ---- attention diagnostics
    att = attention_report(root, cfg, res_dir, ft_new)

    # ---- dimension retention under the unchanged near-zero-variance screen
    prep = ppath(cfg, cfg["outputs"]["preprocessing_dir"])
    rows = []
    for fold in range(1, 6):
        for j in [None] + [1, 2, 3]:
            name = ("fold_%d_final_scaler.npz" % fold if j is None
                    else "fold_%d_inner_%d_scaler.npz" % (fold, j))
            z = np.load(os.path.join(prep, name), allow_pickle=True)
            rows.append({"fold": fold, "scaler": "final" if j is None else "inner_%d" % j,
                         "dimensions_offered": int(cfg["phase3"]["expected_features"]),
                         "dimensions_retained": int(len(z["kept_features"])),
                         "dimensions_dropped": int(len(z["dropped_features"])),
                         "mean_min": float(np.min(z["mean"])),
                         "mean_median": float(np.median(z["mean"])),
                         "mean_max": float(np.max(z["mean"])),
                         "sd_min": float(np.min(z["std"])),
                         "sd_median": float(np.median(z["std"])),
                         "sd_max": float(np.max(z["std"]))})
    ret = pd.DataFrame(rows)
    ret.to_csv(os.path.join(res_dir, "dimension_retention.csv"), index=False)

    # ---- small bags, reported separately and never excluded
    cohort = load_cohort(cfg)
    krule = k_rule_from_config(cfg)
    limit = int(cfg["evaluation"]["small_bag_threshold"])
    small = []
    for pid in pids:
        n = cohort.bags[pid].n_patches
        if n >= limit:
            continue
        k = bag_k(n, krule)
        pn, pr = float(new.loc[pid, "predicted_probability_ADC"]), \
            float(ref.loc[pid, "predicted_probability_ADC"])
        t = int(new.loc[pid, "true_label"])
        small.append({
            "PatientID": pid, "histology": cohort.bags[pid].histology, "true_label": t,
            "n_instances": n, "adaptive_k": k, "n_pseudo_instances": 2 * k,
            "two_k_le_n": bool(2 * k <= n), "top_bottom_disjoint": bool(2 * k <= n),
            "outer_fold": int(new.loc[pid, "outer_fold"]),
            "selected_lambda_instance": float(new.loc[pid, "selected_lambda_instance"]),
            "phase6b_probability_ADC": pr,
            "phase6c_probability_ADC": pn,
            "phase6b_correct": bool(int(pr >= thr) == t),
            "phase6c_correct": bool(int(pn >= thr) == t),
        })
    small_df = pd.DataFrame(small).sort_values("n_instances")
    small_df.to_csv(os.path.join(res_dir, "small_bag_predictions.csv"), index=False)

    # ---- pooled metrics file
    oof = {
        "phase": "6c", "model": MODEL_NAME,
        "n_patients": len(pids), "n_adc": int((y == 1).sum()),
        "n_scc": int((y == 0).sum()), "threshold": thr,
        "positive_class": "ADC (label 1)",
        "pooled": {"phase6c": m_new, "phase6b_reference": m_ref,
                   "difference": {k: m_new[k] - m_ref[k] for k in FOLD_METRIC_KEYS}},
        "fold_mean_sd": {
            "phase6c": {k: mean_sd(ft_new[k].values) for k in FOLD_METRIC_KEYS},
            "phase6b_reference": {k: mean_sd(ft_ref[k].values) for k in FOLD_METRIC_KEYS}},
        "per_fold": ft_new.to_dict("records"),
        "per_fold_comparison": fold_cmp.to_dict("records"),
        "n_folds_phase6c_better_roc_auc": n_folds_better,
        "selected_lambda_instance": {int(r["fold"]): float(r["selected_lambda_instance"])
                                     for _, r in ft_new.iterrows()},
        "selected_epoch": {int(r["fold"]): int(r["selected_epoch"])
                           for _, r in ft_new.iterrows()},
        "bootstrap_summary": {k: {"difference_mean": bs["metrics"][k]["difference_mean"],
                                  "ci_low": bs["metrics"][k]["ci_low"],
                                  "ci_high": bs["metrics"][k]["ci_high"],
                                  "ci_includes_zero": bs["metrics"][k]["ci_includes_zero"],
                                  "fraction_favouring_phase6c":
                                      bs["metrics"][k]["fraction_of_resamples_favouring_new"]}
                              for k in ("roc_auc", "pr_auc")},
        "attention": att,
        "dimension_retention": {
            "n_scalers": int(len(ret)),
            "dimensions_offered": int(cfg["phase3"]["expected_features"]),
            "retained_unique": sorted(set(int(v) for v in ret["dimensions_retained"])),
            "dropped_unique": sorted(set(int(v) for v in ret["dimensions_dropped"])),
            "variance_rule": cfg["preprocessing"]["variance"]},
        "small_bags": {"threshold": limit, "n": int(len(small_df)),
                       "phase6c_correct": int(small_df["phase6c_correct"].sum()),
                       "phase6b_correct": int(small_df["phase6b_correct"].sum()),
                       "all_top_bottom_disjoint": bool(small_df["top_bottom_disjoint"].all())},
    }
    with open(os.path.join(res_dir, "oof_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(oof, fh, indent=2)

    d = bs["metrics"]
    pct = "%"
    print("Phase 6C pooled OOF: ROC-AUC %.4f  PR-AUC %.4f  balacc %.4f  MCC %.4f"
          % (m_new["roc_auc"], m_new["pr_auc"], m_new["balanced_accuracy"], m_new["mcc"]))
    print("Phase 6B reference : ROC-AUC %.4f  PR-AUC %.4f  balacc %.4f  MCC %.4f"
          % (m_ref["roc_auc"], m_ref["pr_auc"], m_ref["balanced_accuracy"], m_ref["mcc"]))
    print("difference         : ROC-AUC %+.4f  PR-AUC %+.4f   folds better %d/5"
          % (m_new["roc_auc"] - m_ref["roc_auc"], m_new["pr_auc"] - m_ref["pr_auc"],
             n_folds_better))
    for k in ("roc_auc", "pr_auc"):
        print("  %-8s 95%s CI [%+.4f, %+.4f]  includes 0: %s  favouring 6C %.1f%s"
              % (k, pct, d[k]["ci_low"], d[k]["ci_high"], d[k]["ci_includes_zero"],
                 100.0 * d[k]["fraction_of_resamples_favouring_new"], pct))
    print("fold ROC-AUC mean +- SD: %.3f +- %.3f"
          % (oof["fold_mean_sd"]["phase6c"]["roc_auc"]["mean"],
             oof["fold_mean_sd"]["phase6c"]["roc_auc"]["sd"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
