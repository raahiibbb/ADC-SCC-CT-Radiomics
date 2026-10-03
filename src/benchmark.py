"""Runtime benchmark and full-cohort extrapolation from the pilot run.

The pilot is deliberately size-extreme (smallest / median / largest Gradient ROI
of each histology), so its *mean* seconds-per-patient is a poor estimator for the
whole cohort.  Instead a linear model

    seconds(patient) = a + b * n_retained_patches
    n_retained_patches ~ c * gradient_volume_cm3

is fitted on the pilot and applied to the 194 known Gradient volumes.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cohort import load_cohort                       # noqa: E402
from config_io import DEFAULT_CONFIG, load_config, project_path   # noqa: E402


def main():
    cfg = load_config(DEFAULT_CONFIG)
    qa_path = project_path(cfg, "reports", "extraction_qa_pilot.csv")
    with open(qa_path, encoding="utf-8-sig") as fh:
        pilot = list(csv.DictReader(fh))

    cohort = {r["PatientID"]: r for r in load_cohort(cfg)}
    eligible = [r for r in cohort.values() if r["eligible"]]
    vols = np.array([float(r["gradient_volume_mm3"]) / 1000.0 for r in eligible])

    n_ret = np.array([float(r["n_retained_patches"]) for r in pilot])
    n_cand = np.array([float(r["n_candidate_patches"]) for r in pilot])
    secs = np.array([float(r["seconds_total"]) for r in pilot])
    pv = np.array([float(cohort[r["patient_id"]]["gradient_volume_mm3"]) / 1000.0
                   for r in pilot])

    # seconds = a + b * retained
    b, a = np.polyfit(n_ret, secs, 1)
    resid = secs - (a + b * n_ret)
    # retained = c * volume_cm3  (through the origin)
    c = float((n_ret @ pv) / (pv @ pv))

    pred_ret = c * vols
    pred_secs = a + b * pred_ret
    total_s = float(pred_secs.sum())

    out = {
        "pilot_patients": [r["patient_id"] for r in pilot],
        "pilot_measured": [
            {"patient_id": r["patient_id"], "histology": r["histology"],
             "gradient_volume_cm3": round(float(
                 cohort[r["patient_id"]]["gradient_volume_mm3"]) / 1000.0, 1),
             "candidate_patches": int(r["n_candidate_patches"]),
             "retained_patches": int(r["n_retained_patches"]),
             "seconds": float(r["seconds_total"]),
             "ms_per_retained_patch": round(1000.0 * float(r["seconds_total"])
                                            / max(1, int(r["n_retained_patches"])), 1)}
            for r in pilot],
        "pilot_totals": {
            "n_patients": len(pilot),
            "candidate_patches": int(n_cand.sum()),
            "retained_patches": int(n_ret.sum()),
            "wall_seconds": round(float(secs.sum()), 1),
            "mean_seconds_per_patient": round(float(secs.mean()), 2),
        },
        "timing_model": {
            "form": "seconds = a + b * n_retained_patches",
            "a_seconds": round(float(a), 4),
            "b_seconds_per_patch": round(float(b), 5),
            "max_abs_residual_seconds": round(float(np.abs(resid).max()), 3),
            "retained_per_cm3": round(c, 4),
        },
        "full_cohort_projection": {
            "n_eligible": len(eligible),
            "gradient_volume_cm3_total": round(float(vols.sum()), 1),
            "gradient_volume_cm3_median": round(float(np.median(vols)), 1),
            "predicted_retained_patches_total": int(round(pred_ret.sum())),
            "predicted_retained_patches_median_per_patient": int(round(np.median(pred_ret))),
            "predicted_retained_patches_min_per_patient": int(round(pred_ret.min())),
            "predicted_retained_patches_max_per_patient": int(round(pred_ret.max())),
            "estimated_seconds_total": round(total_s, 1),
            "estimated_minutes_total": round(total_s / 60.0, 1),
            "naive_estimate_minutes_from_pilot_mean": round(
                float(secs.mean()) * len(eligible) / 60.0, 1),
            "estimated_feature_bytes_float32": int(
                round(pred_ret.sum()) * 74 * 4),
        },
    }
    path = project_path(cfg, "reports", "runtime_benchmark.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out["timing_model"], indent=2))
    print(json.dumps(out["full_cohort_projection"], indent=2))
    print("written:", path)


if __name__ == "__main__":
    main()
