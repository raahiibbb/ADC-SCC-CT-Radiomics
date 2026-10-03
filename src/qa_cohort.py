"""Cohort-level QA over a full local-radiomics extraction (ROI from --config).

Read-only with respect to the extraction: it re-opens every .npz bag and its QA
sidecar and re-checks the cohort as a whole. It never re-runs PyRadiomics and
never rewrites a bag.

    python src/qa_cohort.py
    python src/qa_cohort.py --config config/gtv_rim_experiment.yaml

Checks (the ten acceptance criteria, plus split coverage). Expected counts come
from the configuration, so the same script validates Gradient (194 / 49 / 145)
and GTV+Rim (202 / 51 / 151):
  1  exactly `cohort.expected_eligible` valid patient bags exist
  2  the expected ADC / SCC split
  3  every bag has the expected features, identical names in identical order
  4  no missing and no duplicate patients
  5  no non-finite values anywhere in the cohort feature matrix
  6  extraction failures and rejection counts
  7  patch-count distribution and unusually small bags
  8  saved labels agree with cohort metadata
  9  config signatures identical across every bag and sidecar
 10  external Dataset / Vuong5ROI trees unchanged
 11  the frozen 5-fold split still covers exactly the extracted patients
"""
from __future__ import annotations

import csv
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cohort import load_cohort                                  # noqa: E402
from config_io import (DEFAULT_CONFIG, cfg_path,                # noqa: E402
                       extraction_signature, load_config, project_path)
from extract_patient import validate_existing                   # noqa: E402
import readonly_guard                                           # noqa: E402

SMALL_BAG_THRESHOLD = 20        # flag bags below this many patches
# a patch dropped by retention criterion 3 is expected behaviour, but a large
# share would indicate a systemic problem rather than isolated dense patches
FEATURE_REJECT_TOLERANCE = 0.005
REASONS = ["center_outside_roi", "roi_voxels_below_min", "pyradiomics_error",
           "nonfinite_features", "feature_count_mismatch"]


def check(results, name, ok, detail):
    results.append({"check": name, "pass": bool(ok), "detail": detail})
    print("[%s] %-30s %s" % ("PASS" if ok else "FAIL", name, detail))
    return ok


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    args = ap.parse_args()

    cfg = load_config(args.config)
    sig = extraction_signature(cfg)
    n_expected = int(cfg["cohort"]["expected_eligible"])
    hmap = cfg["cohort"]["histology_map"]
    pos_name = [k for k, v in hmap.items() if int(v) == 1][0]
    neg_name = [k for k, v in hmap.items() if int(v) == 0][0]
    feat_dir = project_path(cfg, *cfg["paths"]["features_dir"].split("/"))
    reports = project_path(cfg, "reports")

    cohort = {r["PatientID"]: r for r in load_cohort(cfg)}
    eligible = sorted([r for r in cohort.values() if r["eligible"]],
                      key=lambda r: r["PatientID"])
    eligible_ids = [r["PatientID"] for r in eligible]

    fn_rel = cfg["paths"].get("feature_names_file", "config/feature_names.txt")
    with open(project_path(cfg, *fn_rel.split("/")), encoding="utf-8") as fh:
        expected_names = [l.strip() for l in fh if l.strip()]

    R = []
    per_patient = []
    bad_signature, bad_label, nonfinite, invalid = [], [], [], []
    name_mismatch, feature_count_bad = [], []
    missing_npz, missing_qa, qa_count_mismatch = [], [], []
    feature_rejection_detail = []
    rej_total = dict((k, 0) for k in REASONS)

    print("config signature : %s" % sig)
    print("eligible cohort  : %d patients" % len(eligible_ids))
    print("-" * 72)

    for rec in eligible:
        pid = rec["PatientID"]
        npz = os.path.join(feat_dir, "%s.npz" % pid)
        qaf = os.path.join(feat_dir, "%s.qa.json" % pid)
        if not os.path.isfile(npz):
            missing_npz.append(pid)
            continue
        if not os.path.isfile(qaf):
            missing_qa.append(pid)

        ok, why = validate_existing(npz, pid, sig, rec["label"])
        if not ok:
            invalid.append({"patient_id": pid, "reason": why})

        with np.load(npz, allow_pickle=True) as z:
            f = z["features"]
            names = [str(x) for x in z["feature_names"]]
            n = int(f.shape[0])
            row = {
                "patient_id": pid,
                "histology": str(z["histology"]),
                "label_saved": int(z["label"]),
                "label_cohort": int(rec["label"]),
                "histology_cohort": rec["Histology"],
                "roi_name": str(z["roi_name"]),
                "n_patches": n,
                "n_features": int(f.shape[1]),
                "config_signature": str(z["config_signature"]),
                "npz_version": int(z["npz_version"]),
                "n_nonfinite": int(np.sum(~np.isfinite(f))),
                "feature_min": float(np.min(f)) if f.size else None,
                "feature_max": float(np.max(f)) if f.size else None,
                "roi_voxels_min": int(np.min(z["roi_voxels"])) if n else None,
                "roi_voxels_max": int(np.max(z["roi_voxels"])) if n else None,
                "roi_fraction_min": (round(float(np.min(z["roi_fraction"])), 4)
                                     if n else None),
                "coords_ok": bool(z["coords_index"].shape == (n, 3)
                                  and z["coords_world"].shape == (n, 3)),
                "npz_bytes": os.path.getsize(npz),
                "bag_valid": bool(ok),
                "bag_invalid_reason": "" if ok else why,
            }
            if row["config_signature"] != sig:
                bad_signature.append(pid)
            if row["label_saved"] != row["label_cohort"]:
                bad_label.append(pid)
            if row["n_nonfinite"]:
                nonfinite.append(pid)
            if row["n_features"] != len(expected_names):
                feature_count_bad.append(pid)
            if names != expected_names:
                name_mismatch.append(pid)

        if os.path.isfile(qaf):
            with open(qaf, encoding="utf-8") as fh:
                qa = json.load(fh)
            row["qa_signature"] = qa.get("config_signature")
            row["n_candidate_patches"] = qa["n_candidate_patches"]
            row["n_retained_patches"] = qa["n_retained_patches"]
            row["n_rejected_patches"] = qa["n_rejected_patches"]
            for k in REASONS:
                row["rej_" + k] = qa["rejected_by_reason"][k]
                rej_total[k] += qa["rejected_by_reason"][k]
            row["seconds_total"] = qa["timing_seconds"]["total"]
            row["roi_volume_mm3_resampled"] = round(qa["roi_volume_mm3_resampled"], 1)
            if qa["n_retained_patches"] != row["n_patches"]:
                qa_count_mismatch.append(pid)
            n_fr = (qa["rejected_by_reason"]["pyradiomics_error"]
                    + qa["rejected_by_reason"]["nonfinite_features"]
                    + qa["rejected_by_reason"]["feature_count_mismatch"])
            if n_fr:
                feature_rejection_detail.append({
                    "patient_id": pid,
                    "n_feature_rejections": n_fr,
                    "n_retained_patches": qa["n_retained_patches"],
                    "reasons": dict((k, qa["rejected_by_reason"][k])
                                    for k in ("pyradiomics_error",
                                              "nonfinite_features",
                                              "feature_count_mismatch")),
                    "examples": [e for e in qa.get("rejection_examples", [])
                                 if e.get("reason") != "center_outside_roi"][:5],
                })
            if qa.get("config_signature") != sig and pid not in bad_signature:
                bad_signature.append(pid)
        per_patient.append(row)

    n_bags = len(per_patient)
    adc = [r for r in per_patient if r["label_saved"] == 1]
    scc = [r for r in per_patient if r["label_saved"] == 0]
    patches = [r["n_patches"] for r in per_patient]
    small = sorted([r for r in per_patient if r["n_patches"] < SMALL_BAG_THRESHOLD],
                   key=lambda r: r["n_patches"])
    ids = [r["patient_id"] for r in per_patient]
    dupes = sorted(set(i for i in ids if ids.count(i) > 1))
    on_disk = set(os.path.splitext(f)[0] for f in os.listdir(feat_dir)
                  if f.endswith(".npz"))
    extras = sorted(on_disk - set(eligible_ids))

    # ---- the ten checks -------------------------------------------------
    n_adc_exp = sum(1 for r in eligible if int(r["label"]) == 1)
    n_scc_exp = sum(1 for r in eligible if int(r["label"]) == 0)
    check(R, "1_bag_count_%d" % n_expected,
          n_bags == n_expected and not missing_npz and not invalid,
          "%d valid bags, %d missing, %d invalid"
          % (n_bags, len(missing_npz), len(invalid)))
    check(R, "2_class_balance_%d_%d" % (n_adc_exp, n_scc_exp),
          len(adc) == n_adc_exp and len(scc) == n_scc_exp,
          "%d ADC / %d SCC (%s / %s)" % (len(adc), len(scc), pos_name, neg_name))
    check(R, "3_all_%d_features" % len(expected_names),
          not feature_count_bad and not name_mismatch,
          "%d wrong count, %d name/order mismatch, expected %d names"
          % (len(feature_count_bad), len(name_mismatch), len(expected_names)))
    check(R, "4_no_missing_or_duplicate", not missing_npz and not dupes and not extras,
          "missing %d, duplicate %d, unexpected npz %d"
          % (len(missing_npz), len(dupes), len(extras)))
    check(R, "5_all_finite", not nonfinite,
          "%d patients with non-finite values in %d total patches"
          % (len(nonfinite), sum(patches)))
    # 6 splits into two: a patient-level extraction failure is a defect and must be
    # zero, whereas a patch dropped by retention criterion 3 (PyRadiomics could not
    # return 74 finite features) is the retention rule working as specified. The
    # latter is reported and bounded, not required to be zero.
    run_failures = []
    run_summary = cfg_path(cfg, "extraction_summary_json", tag="all")
    if os.path.isfile(run_summary):
        with open(run_summary, encoding="utf-8") as fh:
            run_failures = json.load(fh).get("failures", [])
    check(R, "6a_no_patient_level_failures",
          not run_failures and not missing_qa and not qa_count_mismatch,
          "%d driver failures, %d missing sidecars, %d count mismatches"
          % (len(run_failures), len(missing_qa), len(qa_count_mismatch)))

    feat_rej = (rej_total["pyradiomics_error"] + rej_total["nonfinite_features"]
                + rej_total["feature_count_mismatch"])
    n_kept = sum(r["n_patches"] for r in per_patient)
    share = feat_rej / float(n_kept + feat_rej) if (n_kept + feat_rej) else 0.0
    check(R, "6b_feature_rejections_bounded", share < FEATURE_REJECT_TOLERANCE,
          "%d feature-level rejections (%.4f%% of extractable patches; "
          "pyradiomics_error %d, nonfinite %d, count_mismatch %d)"
          % (feat_rej, 100.0 * share, rej_total["pyradiomics_error"],
             rej_total["nonfinite_features"], rej_total["feature_count_mismatch"]))
    check(R, "7_patch_distribution", min(patches) > 0,
          "min %d, median %d, max %d; %d bags < %d patches"
          % (min(patches), int(statistics.median(patches)), max(patches),
             len(small), SMALL_BAG_THRESHOLD))
    check(R, "8_labels_match_cohort", not bad_label,
          "%d label disagreements" % len(bad_label))
    check(R, "9_signatures_identical", not bad_signature,
          "%d mismatches; all bags and sidecars == %s" % (len(bad_signature), sig))

    guard_ok, guard = readonly_guard.verify(cfg)
    check(R, "10_external_trees_unchanged", guard_ok,
          "%d files: +%d -%d ~%d" % (guard["n_now"], guard["n_added"],
                                     guard["n_removed"], guard["n_changed"]))

    split_path = project_path(cfg, *cfg["paths"]["splits_file"].split("/"))
    with open(split_path, encoding="utf-8") as fh:
        split_rows = list(csv.DictReader(fh))
    split_ids = sorted(set(r["PatientID"] for r in split_rows))
    check(R, "11_split_covers_all_bags", split_ids == sorted(ids),
          "%d patients in the frozen split, %d bags" % (len(split_ids), len(ids)))

    # ---- write outputs --------------------------------------------------
    fields = sorted(set(k for r in per_patient for k in r))
    order = ["patient_id", "histology", "label_saved", "label_cohort",
             "roi_name", "n_patches", "n_features", "n_candidate_patches",
             "n_retained_patches", "n_rejected_patches"]
    order += [f for f in fields if f.startswith("rej_")]
    order += [f for f in fields if f not in order]
    qa_csv = cfg_path(cfg, "cohort_qa_csv")
    with open(qa_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=order)
        w.writeheader()
        w.writerows(sorted(per_patient, key=lambda r: r["patient_id"]))

    q = statistics.quantiles(patches, n=4)
    total_cand = sum(r.get("n_candidate_patches", 0) for r in per_patient)
    out = {
        "roi": cfg["experiment"]["roi"],
        "roi_filename": cfg["experiment"]["roi_filename"],
        "config_signature": sig,
        "n_bags": n_bags,
        "n_adc": len(adc), "n_scc": len(scc),
        "n_patches_total": sum(patches),
        "n_candidate_patches_total": total_cand,
        "patches": {
            "min": min(patches), "q1": q[0], "median": q[1], "q3": q[2],
            "max": max(patches), "mean": round(statistics.mean(patches), 1),
            "std": round(statistics.pstdev(patches), 1),
            "total_adc": sum(r["n_patches"] for r in adc),
            "total_scc": sum(r["n_patches"] for r in scc),
            "mean_adc": round(statistics.mean([r["n_patches"] for r in adc]), 1),
            "mean_scc": round(statistics.mean([r["n_patches"] for r in scc]), 1),
            "median_adc": statistics.median([r["n_patches"] for r in adc]),
            "median_scc": statistics.median([r["n_patches"] for r in scc]),
        },
        "small_bags": [{"patient_id": r["patient_id"], "label": r["label_saved"],
                        "histology": r["histology"], "n_patches": r["n_patches"]}
                       for r in small],
        "largest_bags": [{"patient_id": r["patient_id"], "label": r["label_saved"],
                          "n_patches": r["n_patches"]}
                         for r in sorted(per_patient, key=lambda x: -x["n_patches"])[:5]],
        "feature_level_rejections": {
            "total": sum(rej_total[k] for k in ("pyradiomics_error",
                                                "nonfinite_features",
                                                "feature_count_mismatch")),
            "tolerance_share": FEATURE_REJECT_TOLERANCE,
            "by_patient": feature_rejection_detail,
        },
        "patient_level_failures": run_failures,
        "rejections": rej_total,
        "rejections_total": sum(rej_total.values()),
        "rejection_share_of_candidates": dict(
            (k, (round(v / float(total_cand), 6) if total_cand else None))
            for k, v in rej_total.items()),
        "problems": {
            "missing_npz": missing_npz, "missing_qa": missing_qa,
            "invalid_bags": invalid, "duplicate_ids": dupes,
            "unexpected_npz": extras, "bad_signature": bad_signature,
            "bad_label": bad_label, "nonfinite": nonfinite,
            "feature_count_bad": feature_count_bad,
            "name_mismatch": name_mismatch,
            "qa_count_mismatch": qa_count_mismatch,
        },
        "feature_value_range": {
            "min": min(r["feature_min"] for r in per_patient),
            "max": max(r["feature_max"] for r in per_patient),
        },
        "roi_voxels_per_patch": {
            "min": min(r["roi_voxels_min"] for r in per_patient),
            "max": max(r["roi_voxels_max"] for r in per_patient),
        },
        "readonly_guard": guard,
        "storage_bytes": sum(r["npz_bytes"] for r in per_patient),
        "checks": R,
        "n_checks": len(R),
        "n_passed": sum(1 for r in R if r["pass"]),
        "all_passed": all(r["pass"] for r in R),
    }
    sfile = cfg_path(cfg, "cohort_qa_json")
    with open(sfile, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("-" * 72)
    print("%d/%d checks passed" % (out["n_passed"], out["n_checks"]))
    print("patches: total %d, min %d, median %d, max %d"
          % (out["n_patches_total"], out["patches"]["min"],
             out["patches"]["median"], out["patches"]["max"]))
    print("per-patient QA : %s" % qa_csv)
    print("summary        : %s" % sfile)
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
