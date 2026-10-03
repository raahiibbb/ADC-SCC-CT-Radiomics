"""Phase-4 descriptive comparison of two extracted ROI experiments.

Two products, both DESCRIPTIVE ONLY -- no classifier is trained, fitted or
evaluated anywhere in this module:

  1. cohort overlap report -- which patients are shared between the reference
     experiment (Gradient, 194 patients) and the new one (GTV+Rim, 202), which
     are new, which were lost, and whether their labels agree.  Written so that
     a later PAIRED comparison on the shared subset is possible.
  2. bag-statistics comparison -- patient count, total patches, median patches
     per patient, patch-count range and ADC/SCC patch counts for each ROI, both
     over the full cohort of each experiment and over the shared subset.

    python src/roi_compare.py --config config/gtv_rim_experiment.yaml
"""
from __future__ import annotations

import csv
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cohort import load_cohort                                   # noqa: E402
from config_io import (DEFAULT_CONFIG, extraction_signature,     # noqa: E402
                       load_config, project_path)

OVERLAP_FIELDS = ["PatientID", "Histology", "label", "in_reference", "in_new",
                  "membership", "reference_patches", "new_patches",
                  "reference_fold", "new_fold", "labels_agree"]


def bag_index(cfg):
    """{PatientID: n_patches} for every bag of an experiment, plus the labels."""
    d = project_path(cfg, *cfg["paths"]["features_dir"].split("/"))
    out = {}
    if not os.path.isdir(d):
        return out
    for fn in sorted(f for f in os.listdir(d) if f.endswith(".npz")):
        with np.load(os.path.join(d, fn), allow_pickle=True) as z:
            out[str(z["patient_id"])] = {
                "n_patches": int(z["features"].shape[0]),
                "label": int(z["label"]),
                "histology": str(z["histology"]),
                "roi": str(z["roi_name"]),
            }
    return out


def fold_index(cfg):
    """{PatientID: outer test fold} from the frozen split of an experiment."""
    p = project_path(cfg, *cfg["paths"]["splits_file"].split("/"))
    if not os.path.isfile(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {r["PatientID"]: int(r["fold"]) for r in rows if r["split"] == "test"}


def describe(bags, pids=None):
    sel = [b for p, b in bags.items() if pids is None or p in pids]
    if not sel:
        return None
    n = [b["n_patches"] for b in sel]
    adc = [b["n_patches"] for b in sel if b["label"] == 1]
    scc = [b["n_patches"] for b in sel if b["label"] == 0]
    return {
        "n_patients": len(sel),
        "n_adc": len(adc), "n_scc": len(scc),
        "total_patches": int(sum(n)),
        "total_patches_adc": int(sum(adc)),
        "total_patches_scc": int(sum(scc)),
        "patches_min": min(n), "patches_max": max(n),
        "patches_median": float(statistics.median(n)),
        "patches_mean": round(statistics.mean(n), 1),
        "patches_sd": round(statistics.pstdev(n), 1),
        "patches_median_adc": float(statistics.median(adc)) if adc else None,
        "patches_median_scc": float(statistics.median(scc)) if scc else None,
        "bags_under_20_patches": sum(1 for v in n if v < 20),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--reference-config", default="config/experiment.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ref = load_config(project_path(cfg, *args.reference_config.split("/")))
    new_roi = cfg["experiment"]["roi"]
    ref_roi = ref["experiment"]["roi"]

    new_bags, ref_bags = bag_index(cfg), bag_index(ref)
    new_folds, ref_folds = fold_index(cfg), fold_index(ref)
    new_cohort = {r["PatientID"]: r for r in load_cohort(cfg) if r["eligible"]}
    ref_cohort = {r["PatientID"]: r for r in load_cohort(ref) if r["eligible"]}

    shared = sorted(set(ref_cohort) & set(new_cohort))
    only_new = sorted(set(new_cohort) - set(ref_cohort))
    only_ref = sorted(set(ref_cohort) - set(new_cohort))
    all_ids = sorted(set(ref_cohort) | set(new_cohort))

    rows, disagree = [], []
    for pid in all_ids:
        r, w = ref_cohort.get(pid), new_cohort.get(pid)
        src = w or r
        member = ("shared" if (r and w) else "%s_only" % new_roi if w
                  else "%s_only" % ref_roi)
        agree = (r is None or w is None or int(r["label"]) == int(w["label"]))
        if not agree:
            disagree.append(pid)
        rows.append({
            "PatientID": pid, "Histology": src["Histology"],
            "label": int(src["label"]),
            "in_reference": r is not None, "in_new": w is not None,
            "membership": member,
            "reference_patches": (ref_bags.get(pid, {}).get("n_patches")
                                  if r is not None else None),
            "new_patches": (new_bags.get(pid, {}).get("n_patches")
                            if w is not None else None),
            "reference_fold": ref_folds.get(pid),
            "new_fold": new_folds.get(pid),
            "labels_agree": agree,
        })

    overlap_csv = project_path(cfg, "reports", "gradient_vs_gtv_rim_overlap.csv")
    with open(overlap_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OVERLAP_FIELDS)
        w.writeheader()
        w.writerows(rows)

    def counts(ids, src):
        return {"n": len(ids),
                "adc": sum(1 for p in ids if int(src[p]["label"]) == 1),
                "scc": sum(1 for p in ids if int(src[p]["label"]) == 0)}

    # how many outer-fold assignments happen to coincide -- purely informational;
    # the two splits are independent because cohort membership differs
    same_fold = sum(1 for p in shared
                    if p in ref_folds and p in new_folds
                    and ref_folds[p] == new_folds[p])

    overlap = {
        "reference_roi": ref_roi,
        "reference_config_signature": extraction_signature(ref),
        "new_roi": new_roi,
        "new_config_signature": extraction_signature(cfg),
        "reference_cohort": counts(sorted(ref_cohort), ref_cohort),
        "new_cohort": counts(sorted(new_cohort), new_cohort),
        "shared": counts(shared, new_cohort),
        "%s_only" % new_roi: counts(only_new, new_cohort),
        "%s_only" % ref_roi: counts(only_ref, ref_cohort),
        "%s_only_ids" % new_roi: only_new,
        "%s_only_ids" % ref_roi: only_ref,
        "label_disagreements": disagree,
        "shared_patients_same_outer_fold": same_fold,
        "note": ("The two outer 5-fold splits are independent: cohort membership "
                 "differs (%d vs %d patients), so the folds were generated "
                 "separately with StratifiedKFold(5, shuffle=True, "
                 "random_state=42) on each cohort. Use the %d shared patients "
                 "for any later PAIRED comparison, and compare only "
                 "out-of-fold predictions, never fold indices."
                 % (len(ref_cohort), len(new_cohort), len(shared))),
        "shared_patient_ids": shared,
    }
    overlap_json = project_path(cfg, "reports", "gradient_vs_gtv_rim_overlap.json")
    with open(overlap_json, "w", encoding="utf-8") as fh:
        json.dump(overlap, fh, indent=2)

    # ---------------------------------------------------------- bag statistics
    stats = {
        "descriptive_only": True,
        "note": "No classification was performed. Bag sizes only.",
        ref_roi: {"full_cohort": describe(ref_bags),
                  "shared_subset": describe(ref_bags, set(shared))},
        new_roi: {"full_cohort": describe(new_bags),
                  "shared_subset": describe(new_bags, set(shared))},
    }
    a, b = stats[ref_roi]["shared_subset"], stats[new_roi]["shared_subset"]
    if a and b:
        pairs = [(ref_bags[p]["n_patches"], new_bags[p]["n_patches"])
                 for p in shared if p in ref_bags and p in new_bags]
        ratios = [n / float(r) for r, n in pairs if r]
        stats["shared_subset_paired"] = {
            "n_pairs": len(pairs),
            "patch_ratio_new_over_reference": {
                "min": round(min(ratios), 3), "median": round(
                    statistics.median(ratios), 3), "max": round(max(ratios), 3),
                "mean": round(statistics.mean(ratios), 3),
            },
            "n_pairs_new_ge_reference": sum(1 for r, n in pairs if n >= r),
        }
    stats_json = project_path(cfg, "reports",
                              "gradient_vs_gtv_rim_bag_stats.json")
    with open(stats_json, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    keys = ["n_patients", "n_adc", "n_scc", "total_patches", "total_patches_adc",
            "total_patches_scc", "patches_min", "patches_median", "patches_mean",
            "patches_max", "patches_sd", "patches_median_adc",
            "patches_median_scc", "bags_under_20_patches"]
    stats_csv = project_path(cfg, "reports", "gradient_vs_gtv_rim_bag_stats.csv")
    with open(stats_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["roi", "subset"] + keys)
        for roi in (ref_roi, new_roi):
            for sub in ("full_cohort", "shared_subset"):
                d = stats[roi][sub]
                if d:
                    w.writerow([roi, sub] + [d[k] for k in keys])

    print("cohort overlap  : %d shared, %d %s-only, %d %s-only"
          % (len(shared), len(only_new), new_roi, len(only_ref), ref_roi))
    print("label conflicts : %d" % len(disagree))
    for roi in (ref_roi, new_roi):
        d = stats[roi]["full_cohort"]
        if d:
            print("%-9s: %d patients, %d patches, median %g, range %d-%d"
                  % (roi, d["n_patients"], d["total_patches"],
                     d["patches_median"], d["patches_min"], d["patches_max"]))
    print("written: %s" % overlap_csv)
    print("written: %s" % overlap_json)
    print("written: %s" % stats_csv)
    print("written: %s" % stats_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
