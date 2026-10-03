"""Resumable driver for local-radiomics extraction (ROI chosen by --config).

Usage
-----
    python src/run_extraction.py --patients LUNG1-001 LUNG1-002
    python src/run_extraction.py --pilot
    python src/run_extraction.py --all                 # phase 2, not phase 1
    python src/run_extraction.py --all --force         # ignore existing bags
    python src/run_extraction.py --config config/gtv_rim_experiment.yaml --all

Already-completed patients are skipped unless their bag fails validation or was
produced by a different configuration signature.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cohort import load_cohort                       # noqa: E402
from config_io import (DEFAULT_CONFIG, cfg_path,              # noqa: E402
                       extraction_signature, load_config, project_path)
from extract_patient import (build_extractor, extract_patient,  # noqa: E402
                             validate_existing)

QA_FIELDS = ["patient_id", "histology", "label", "status", "n_candidate_patches",
             "n_retained_patches", "n_rejected_patches", "rej_center_outside_roi",
             "rej_roi_voxels_below_min", "rej_pyradiomics_error",
             "rej_nonfinite_features", "rej_feature_count_mismatch", "n_features",
             "roi_voxels_resampled", "roi_volume_mm3_resampled",
             "roi_voxels_per_patch_min", "roi_voxels_per_patch_max",
             "nonfinite_in_saved_matrix", "seconds_total", "seconds_features"]


def qa_row(qa, status="extracted"):
    rej = qa["rejected_by_reason"]
    mm = qa.get("roi_voxel_count_min_max") or [None, None]
    return {
        "patient_id": qa["patient_id"], "histology": qa["histology"],
        "label": qa["label"], "status": status,
        "n_candidate_patches": qa["n_candidate_patches"],
        "n_retained_patches": qa["n_retained_patches"],
        "n_rejected_patches": qa["n_rejected_patches"],
        "rej_center_outside_roi": rej["center_outside_roi"],
        "rej_roi_voxels_below_min": rej["roi_voxels_below_min"],
        "rej_pyradiomics_error": rej["pyradiomics_error"],
        "rej_nonfinite_features": rej["nonfinite_features"],
        "rej_feature_count_mismatch": rej["feature_count_mismatch"],
        "n_features": qa["n_features"],
        "roi_voxels_resampled": qa["roi_voxels_resampled"],
        "roi_volume_mm3_resampled": round(qa["roi_volume_mm3_resampled"], 1),
        "roi_voxels_per_patch_min": mm[0], "roi_voxels_per_patch_max": mm[1],
        "nonfinite_in_saved_matrix": qa["nonfinite_in_saved_matrix"],
        "seconds_total": qa["timing_seconds"]["total"],
        "seconds_features": qa["timing_seconds"]["features"],
    }


def environment_report():
    import numpy, scipy, sklearn, pandas, SimpleITK, radiomics, yaml
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": {
            "numpy": numpy.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "pandas": pandas.__version__,
            "SimpleITK": SimpleITK.__version__, "pyradiomics": radiomics.__version__,
            "PyYAML": yaml.__version__,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--pilot", action="store_true",
                    help="run the frozen pilot list from config/pilot_patients.txt")
    ap.add_argument("--all", action="store_true", help="all eligible patients")
    ap.add_argument("--force", action="store_true", help="ignore existing bags")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None, help="suffix for the QA summary file")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sig = extraction_signature(cfg)
    cohort = {r["PatientID"]: r for r in load_cohort(cfg)}
    eligible = [r for r in cohort.values() if r["eligible"]]

    if args.patients:
        selected = [cohort[p] for p in args.patients]
    elif args.pilot:
        pf = cfg_path(cfg, "pilot_file")
        ids = [l.split("#")[0].strip() for l in open(pf, encoding="utf-8")]
        ids = [i for i in ids if i]
        selected = [cohort[p] for p in ids]
    elif args.all:
        selected = sorted(eligible, key=lambda r: r["PatientID"])
    else:
        ap.error("one of --patients / --pilot / --all is required")
    for r in selected:
        if not r["eligible"]:
            raise SystemExit("%s is not in the eligible %s cohort"
                             % (r["PatientID"], cfg["experiment"]["roi"]))
    if args.limit:
        selected = selected[:args.limit]

    out_dir = project_path(cfg, *cfg["paths"]["features_dir"].split("/"))
    os.makedirs(out_dir, exist_ok=True)
    tag = args.tag or ("pilot" if args.pilot else "all" if args.all else "custom")
    reports_dir = project_path(cfg, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    qa_csv = cfg_path(cfg, "extraction_qa_csv", tag=tag)
    sfile = cfg_path(cfg, "extraction_summary_json", tag=tag)

    print("roi              : %s (%s)"
          % (cfg["experiment"]["roi"], cfg["experiment"]["roi_filename"]))
    print("config signature : %s" % sig)
    print("patients         : %d" % len(selected))
    print("output           : %s" % out_dir)

    extractor = build_extractor(cfg)
    rows, failures = [], []
    t_run = time.time()
    for n, rec in enumerate(selected, 1):
        pid = rec["PatientID"]
        npz = os.path.join(out_dir, "%s.npz" % pid)
        qa_json = os.path.join(out_dir, "%s.qa.json" % pid)
        if not args.force and os.path.isfile(npz):
            ok, why = validate_existing(npz, pid, sig, rec["label"])
            if ok and os.path.isfile(qa_json):
                with open(qa_json, encoding="utf-8") as fh:
                    rows.append(qa_row(json.load(fh), status="skipped_valid"))
                print("[%3d/%3d] %s  SKIP (already valid)" % (n, len(selected), pid))
                continue
            print("[%3d/%3d] %s  re-extracting (%s)" % (n, len(selected), pid, why))
        try:
            qa = extract_patient(cfg, rec, extractor=extractor, out_dir=out_dir)
            rows.append(qa_row(qa))
            print("[%3d/%3d] %s  %4d/%-4d patches kept, %d features, %.1fs"
                  % (n, len(selected), pid, qa["n_retained_patches"],
                     qa["n_candidate_patches"], qa["n_features"],
                     qa["timing_seconds"]["total"]))
        except Exception as exc:                       # noqa: BLE001
            failures.append({"patient_id": pid, "error": str(exc)[:500]})
            print("[%3d/%3d] %s  FAILED: %s" % (n, len(selected), pid, str(exc)[:200]))

    with open(qa_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=QA_FIELDS)
        w.writeheader()
        w.writerows(rows)

    extracted = [r for r in rows if r["status"] == "extracted"]
    elapsed = time.time() - t_run
    summary = {
        "tag": tag,
        "roi": cfg["experiment"]["roi"],
        "roi_filename": cfg["experiment"]["roi_filename"],
        "config": os.path.relpath(cfg["_config_path"], cfg["_project_root"]).replace("\\", "/"),
        "config_signature": sig,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_selected": len(selected),
        "n_extracted_now": len(extracted),
        "n_skipped_valid": len([r for r in rows if r["status"] == "skipped_valid"]),
        "n_failed": len(failures),
        "failures": failures,
        "wall_seconds": round(elapsed, 1),
        "seconds_per_patient_mean": (round(sum(r["seconds_total"] for r in extracted)
                                           / len(extracted), 2) if extracted else None),
        "patches_total_retained": sum(r["n_retained_patches"] for r in rows),
        "patches_total_candidate": sum(r["n_candidate_patches"] for r in rows),
        "n_eligible_cohort": len(eligible),
        "environment": environment_report(),
    }
    if summary["seconds_per_patient_mean"]:
        summary["estimated_full_cohort_minutes"] = round(
            summary["seconds_per_patient_mean"] * len(eligible) / 60.0, 1)
    with open(sfile, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("-" * 60)
    print("extracted %d, skipped %d, failed %d in %.1f s"
          % (summary["n_extracted_now"], summary["n_skipped_valid"],
             summary["n_failed"], elapsed))
    if summary.get("estimated_full_cohort_minutes"):
        print("estimated full %d-patient run: %.1f min"
              % (len(eligible), summary["estimated_full_cohort_minutes"]))
    print("QA:      %s" % qa_csv)
    print("summary: %s" % sfile)


if __name__ == "__main__":
    main()
