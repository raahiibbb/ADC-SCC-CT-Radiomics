"""Phase 6B - representation-specific diagnostics.

Descriptive only.  Nothing here tunes, selects or re-fits anything: every number
is re-derived from artefacts already written by the Phase-6B training run and
from the frozen Phase-5 baseline on disk.

Reports, in the order the phase brief asks for them:

  1. retained embedding dimensions per fold (final refit and each inner split);
  2. dropped near-zero-variance dimensions;
  3. training-only scaler distributions;
  4. selected epochs;
  5. fold-by-fold outer-test ROC-AUC;
  6. whether the previously difficult fold 2 improves;
  7. whether any improvement is spread over folds or driven by one fold;
  8. whether attention is less uniform than in Phase 5;
  9. whether sharper attention corresponds to better outer-fold discrimination.

    python src/phase6b_diagnostics.py --config config/phase6b.yaml
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
from attention_stats import bag_stats  # noqa: E402
from mil_data import load_phase3_config, model_tag, ppath  # noqa: E402
from mil_metrics import mean_sd, patient_metrics  # noqa: E402

MODELS = ("mean_mil", "attention_mil")
CONC_KEYS = ("max_over_uniform", "max_over_min", "normalized_entropy", "top10pct_mass")
PHASE5_ATTENTION_DIR = "attention/gtv_rim"
PHASE5_PREDICTIONS = {"mean_mil": "predictions/gtv_rim_mean_mil_oof.csv",
                      "attention_mil": "predictions/gtv_rim_attention_mil_oof.csv"}


def fold_roc(df: pd.DataFrame, thr: float) -> Dict[int, float]:
    out = {}
    for f, sub in df.groupby("outer_fold"):
        out[int(f)] = patient_metrics(sub["true_label"].values,
                                      sub["predicted_probability_ADC"].values,
                                      thr)["roc_auc"]
    return out


def scaler_rows(cfg: dict) -> pd.DataFrame:
    prep = ppath(cfg, cfg["outputs"]["preprocessing_dir"])
    n_total = int(cfg["phase3"]["expected_features"])
    rows: List[dict] = []
    for fold in range(1, 6):
        meta_path = os.path.join(prep, "fold_%d_scalers.json" % fold)
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        for scope, fname in [("outer_train_final", "fold_%d_final_scaler.npz" % fold)] + \
                [("inner_%d" % j, "fold_%d_inner_%d_scaler.npz" % (fold, j))
                 for j in sorted(int(k) for k in meta["inner"])]:
            with np.load(os.path.join(prep, fname), allow_pickle=True) as z:
                keep = np.asarray(z["keep_idx"], dtype=int)
                mean = np.asarray(z["mean"], dtype=float)
                std = np.asarray(z["std"], dtype=float)
                dropped = [str(x) for x in z["dropped_features"]]
                pw = float(z["pos_weight"])
            rows.append({
                "fold": fold, "scope": scope,
                "n_dimensions_offered": n_total,
                "n_dimensions_retained": int(keep.size),
                "n_dimensions_dropped": int(n_total - keep.size),
                "dropped_dimensions": ";".join(dropped),
                "pos_weight": pw,
                "scaler_mean_min": float(mean.min()), "scaler_mean_median": float(np.median(mean)),
                "scaler_mean_max": float(mean.max()), "scaler_mean_mean": float(mean.mean()),
                "scaler_std_min": float(std.min()), "scaler_std_median": float(np.median(std)),
                "scaler_std_max": float(std.max()), "scaler_std_mean": float(std.mean()),
                "scaling_source": ("all_outer_train_patches_only" if scope == "outer_train_final"
                                   else "inner_train_patches_only"),
            })
    return pd.DataFrame(rows)


def attention_table(root: str, rel_dir: str) -> pd.DataFrame:
    d = os.path.join(root, *rel_dir.split("/"))
    rows = []
    for fn in sorted(f for f in os.listdir(d) if f.endswith(".npz")):
        with np.load(os.path.join(d, fn), allow_pickle=True) as z:
            a = np.asarray(z["attention"], dtype=np.float64)
            row = {"PatientID": str(z["patient_id"]),
                   "true_label": int(z["true_label"]),
                   "outer_fold": int(z["outer_fold"]),
                   "predicted_probability_ADC": float(z["predicted_probability_adc"]),
                   "attention_sum": float(a.sum())}
            row.update(bag_stats(a))
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 6B representation diagnostics")
    ap.add_argument("--config", default="config/phase6b.yaml")
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_phase3_config(os.path.join(root, *args.config.split("/")))
    thr = float(cfg["evaluation"]["threshold"])
    res_dir = ppath(cfg, cfg["outputs"]["results_dir"])
    os.makedirs(res_dir, exist_ok=True)

    with open(ppath(cfg, cfg["outputs"]["run_json"]), "r", encoding="utf-8") as fh:
        run = json.load(fh)
    selected = {m: {int(k): int(v) for k, v in run["selected_epochs"][m].items()}
                for m in MODELS}

    pred_dir = ppath(cfg, cfg["outputs"]["predictions_dir"])
    new = {m: pd.read_csv(os.path.join(pred_dir, "%s_oof.csv" % model_tag(cfg, m)))
           for m in MODELS}
    ref = {m: pd.read_csv(os.path.join(root, *PHASE5_PREDICTIONS[m].split("/")))
           for m in MODELS}

    # ---------------------------------------------------- 1-3 dimension retention
    sc = scaler_rows(cfg)
    sc.to_csv(os.path.join(res_dir, "dimension_retention.csv"), index=False)

    # --------------------------------------------------------- 4-7 folds, epochs
    folds: Dict[str, dict] = {}
    for m in MODELS:
        fr_new = fold_roc(new[m], thr)
        fr_ref = fold_roc(ref[m], thr)
        folds[m] = {
            "selected_epoch_by_fold": selected[m],
            "phase6b_fold_roc_auc": fr_new,
            "phase5_fold_roc_auc": fr_ref,
            "difference_by_fold": {f: fr_new[f] - fr_ref[f] for f in sorted(fr_new)},
            "phase6b_fold_mean_sd": mean_sd(list(fr_new.values())),
            "phase5_fold_mean_sd": mean_sd(list(fr_ref.values())),
            "n_folds_phase6b_better": int(sum(fr_new[f] > fr_ref[f] for f in fr_new)),
            "fold2_phase5": fr_ref[2], "fold2_phase6b": fr_new[2],
            "fold2_improves": bool(fr_new[2] > fr_ref[2]),
            "fold2_above_chance_phase6b": bool(fr_new[2] > 0.5),
        }
        d = np.array([fr_new[f] - fr_ref[f] for f in sorted(fr_new)], dtype=float)
        pooled_new = patient_metrics(new[m]["true_label"].values,
                                     new[m]["predicted_probability_ADC"].values, thr)
        pooled_ref = patient_metrics(ref[m]["true_label"].values,
                                     ref[m]["predicted_probability_ADC"].values, thr)
        # is any pooled gain spread over folds or carried by a single fold?
        largest = int(np.argmax(np.abs(d))) + 1
        folds[m]["improvement_concentration"] = {
            "fold_differences": {int(f): float(v) for f, v in
                                 zip(sorted(fr_new), d)},
            "largest_absolute_difference_fold": largest,
            "largest_absolute_difference": float(np.abs(d).max()),
            "sum_of_differences": float(d.sum()),
            "share_of_summed_difference_from_largest_fold":
                (float(d[largest - 1] / d.sum()) if abs(d.sum()) > 1e-12 else None),
            "n_folds_improved": int((d > 0).sum()),
            "n_folds_worsened": int((d < 0).sum()),
        }
        folds[m]["pooled_oof"] = {"phase6b": pooled_new["roc_auc"],
                                  "phase5": pooled_ref["roc_auc"],
                                  "difference": pooled_new["roc_auc"] - pooled_ref["roc_auc"]}

    # ------------------------------------------------------------- 8-9 attention
    att_new = attention_table(root, cfg["outputs"]["attention_dir"])
    att_ref = attention_table(root, PHASE5_ATTENTION_DIR)
    att_new.to_csv(os.path.join(res_dir, "attention_concentration_phase6b_detail.csv"),
                   index=False)

    def summarise(df: pd.DataFrame) -> dict:
        return {k: {"mean": float(df[k].mean()), "median": float(df[k].median()),
                    "min": float(df[k].min()), "max": float(df[k].max())}
                for k in CONC_KEYS}

    fr_att_new = fold_roc(new["attention_mil"], thr)
    by_fold = {}
    for f, sub in att_new.groupby("outer_fold"):
        sub_ref = att_ref[att_ref["outer_fold"] == f]
        by_fold[int(f)] = {
            "n_bags": int(len(sub)),
            "selected_epoch_attention_mil": selected["attention_mil"][int(f)],
            "outer_test_roc_auc": fr_att_new[int(f)],
            "phase6b_median": {k: float(sub[k].median()) for k in CONC_KEYS},
            "phase5_median_same_fold_index": {k: float(sub_ref[k].median())
                                              for k in CONC_KEYS},
        }
    # rank correlation between concentration and outer-fold discrimination (5 points)
    med_entropy = np.array([by_fold[f]["phase6b_median"]["normalized_entropy"]
                            for f in sorted(by_fold)])
    med_top10 = np.array([by_fold[f]["phase6b_median"]["top10pct_mass"]
                          for f in sorted(by_fold)])
    auc_by_fold = np.array([by_fold[f]["outer_test_roc_auc"] for f in sorted(by_fold)])
    spearman = lambda a, b: float(pd.Series(a).corr(pd.Series(b), method="spearman"))

    attention = {
        "n_bags_phase6b": int(len(att_new)), "n_bags_phase5": int(len(att_ref)),
        "phase6b_overall": summarise(att_new),
        "phase5_overall": summarise(att_ref),
        "more_selective_than_phase5": {
            "median_normalized_entropy_phase6b": float(att_new["normalized_entropy"].median()),
            "median_normalized_entropy_phase5": float(att_ref["normalized_entropy"].median()),
            "lower_entropy_means_more_selective": True,
            "phase6b_more_selective_by_median_entropy": bool(
                att_new["normalized_entropy"].median() < att_ref["normalized_entropy"].median()),
            "median_max_over_uniform_phase6b": float(att_new["max_over_uniform"].median()),
            "median_max_over_uniform_phase5": float(att_ref["max_over_uniform"].median()),
            "median_top10pct_mass_phase6b": float(att_new["top10pct_mass"].median()),
            "median_top10pct_mass_phase5": float(att_ref["top10pct_mass"].median()),
            "n_bags_below_entropy_0_99_phase6b": int((att_new["normalized_entropy"] < 0.99).sum()),
            "n_bags_below_entropy_0_99_phase5": int((att_ref["normalized_entropy"] < 0.99).sum()),
        },
        "by_fold": by_fold,
        "concentration_vs_discrimination": {
            "n_points": int(len(auc_by_fold)),
            "spearman_median_entropy_vs_outer_test_roc_auc": spearman(med_entropy, auc_by_fold),
            "spearman_median_top10pct_mass_vs_outer_test_roc_auc":
                spearman(med_top10, auc_by_fold),
            "caveat": ("five folds only; a rank correlation on five points carries "
                       "essentially no information and is reported descriptively"),
        },
        "sanity": {
            "all_attention_finite": bool(np.isfinite(att_new[list(CONC_KEYS)].values).all()),
            "max_abs_attention_sum_minus_one": float(
                np.max(np.abs(att_new["attention_sum"].values - 1.0))),
        },
    }

    out = {
        "phase": "6b",
        "note": ("descriptive diagnostics only; nothing in this file was used to "
                 "tune, select or re-fit anything"),
        "dimension_retention": {
            "n_dimensions_offered": int(cfg["phase3"]["expected_features"]),
            "variance_rule": cfg["preprocessing"]["variance"],
            "retained_by_fold_final": {int(r["fold"]): int(r["n_dimensions_retained"])
                                       for _, r in sc[sc["scope"] == "outer_train_final"].iterrows()},
            "dropped_by_fold_final": {int(r["fold"]): int(r["n_dimensions_dropped"])
                                      for _, r in sc[sc["scope"] == "outer_train_final"].iterrows()},
            "retained_min_over_all_scalers": int(sc["n_dimensions_retained"].min()),
            "retained_max_over_all_scalers": int(sc["n_dimensions_retained"].max()),
            "n_scalers": int(len(sc)),
            "rows": sc.to_dict(orient="records"),
        },
        "folds": folds,
        "attention": attention,
        "runtime_seconds": run.get("total_runtime_seconds"),
        "device": run.get("environment", {}).get("device"),
    }
    with open(os.path.join(res_dir, "representation_diagnostics.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)

    print("dimensions retained (final scalers): %s"
          % out["dimension_retention"]["retained_by_fold_final"])
    for m in MODELS:
        f = folds[m]
        print("%-14s selected epochs %s | fold ROC-AUC %s | pooled %.4f vs Phase-5 %.4f"
              % (m, f["selected_epoch_by_fold"],
                 {k: round(v, 4) for k, v in f["phase6b_fold_roc_auc"].items()},
                 f["pooled_oof"]["phase6b"], f["pooled_oof"]["phase5"]))
        print("               fold 2: %.4f -> %.4f (improves: %s); folds better than Phase 5: %d/5"
              % (f["fold2_phase5"], f["fold2_phase6b"], f["fold2_improves"],
                 f["n_folds_phase6b_better"]))
    a = attention["more_selective_than_phase5"]
    print("attention median normalised entropy: Phase 6B %.4f vs Phase 5 %.4f (more selective: %s)"
          % (a["median_normalized_entropy_phase6b"], a["median_normalized_entropy_phase5"],
             a["phase6b_more_selective_by_median_entropy"]))
    print("wrote %s" % os.path.join(cfg["outputs"]["results_dir"],
                                    "representation_diagnostics.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
