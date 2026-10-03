"""Build and validate an ROI cohort from the read-only metadata sources.

The ROI is chosen entirely by configuration (`experiment.roi_filename` plus the
`cohort.*_column` / `cohort.*_field` keys), so the same code builds the Gradient
cohort and the GTV+Rim cohort. Defaults reproduce the Gradient behaviour.

Two independent sources are cross-checked:
  * Vuong5ROI  cohort_summary.csv       (mask availability + exclusion flags)
  * Dataset    dataset_metadata_master.csv (histology of record)

Nothing outside the project directory is ever written.
"""
from __future__ import annotations

import csv
import os

from config_io import (cohort_setting, declared_exclusions, load_config,
                       project_path)

TRUE = {"true", "1", "yes", "t"}


def _read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _is_true(v) -> bool:
    return str(v).strip().lower() in TRUE


def build_cohort(cfg: dict) -> dict:
    """Return dict with the eligible cohort and a full consistency audit."""
    summary = _read_csv(cfg["paths"]["cohort_summary"])
    master = _read_csv(cfg["paths"]["dataset_metadata"])
    master_by_id = {r["PatientID"]: r for r in master}

    vuong_root = cfg["paths"]["vuong_root"]
    ct_name = cfg["paths"]["ct_filename"]
    roi_name = cfg["experiment"]["roi_filename"]
    hmap = cfg["cohort"]["histology_map"]
    declared = declared_exclusions(cfg)
    excl_col = cohort_setting(cfg, "exclusion_column")
    excl_field = cohort_setting(cfg, "exclusion_field")
    vol_col = cohort_setting(cfg, "roi_volume_column")
    vol_field = cohort_setting(cfg, "roi_volume_field")

    audit = {
        "n_summary_rows": len(summary),
        "n_master_rows": len(master),
        "histology_mismatch": [],
        "missing_in_master": [],
        "flagged_excluded": [],
        "roi_file_missing": [],
        "roi_file_present_but_excluded": [],
        "ct_file_missing": [],
        "unmapped_histology": [],
    }

    rows = []
    for r in summary:
        pid = r["PatientID"]
        m = master_by_id.get(pid)
        if m is None:
            audit["missing_in_master"].append(pid)
            continue
        if (m["Histology"] or "").strip() != (r["Histology"] or "").strip():
            audit["histology_mismatch"].append(
                {"PatientID": pid, "summary": r["Histology"], "master": m["Histology"]})
        hist = (m["Histology"] or "").strip()
        if hist not in hmap:
            audit["unmapped_histology"].append({"PatientID": pid, "Histology": hist})
            continue

        pdir = os.path.join(vuong_root, pid)
        ct_path = os.path.join(pdir, ct_name)
        roi_path = os.path.join(pdir, roi_name)
        ct_ok = os.path.isfile(ct_path)
        roi_ok = os.path.isfile(roi_path)
        if not ct_ok:
            audit["ct_file_missing"].append(pid)

        excluded_flag = _is_true(r.get(excl_col, "")) or _is_true(m.get(excl_col, ""))
        if excluded_flag:
            audit["flagged_excluded"].append(pid)
        if excluded_flag and roi_ok:
            audit["roi_file_present_but_excluded"].append(pid)
        if not excluded_flag and not roi_ok:
            audit["roi_file_missing"].append(pid)

        rows.append({
            "PatientID": pid,
            "Histology": hist,
            "label": hmap[hist],
            excl_field: excluded_flag,
            "roi_present": roi_ok,
            "ct_present": ct_ok,
            "eligible": (not excluded_flag) and roi_ok and ct_ok,
            "exclusion_reason": r.get("ROI_Exclusion_Reason", "") or m.get(
                "Vuong5ROI_Exclusion_Reason", ""),
            vol_field: r.get(vol_col, ""),
            "gtv_volume_mm3": r.get("GTV_Volume_mm3", ""),
            "flagged_for_review": r.get("Flagged_For_Review", ""),
            "flag_reason": r.get("Flag_Reason", ""),
            "ct_dir": pdir,
        })

    eligible = [r for r in rows if r["eligible"]]
    excluded = sorted({r["PatientID"] for r in rows if not r["eligible"]})

    audit["n_total"] = len(rows)
    audit["n_eligible"] = len(eligible)
    audit["excluded_ids"] = excluded
    audit["declared_exclusions"] = sorted(declared)
    audit["exclusions_match_declared"] = (set(excluded) == declared)
    audit["roi"] = cfg["experiment"]["roi"]
    audit["roi_filename"] = roi_name
    audit["exclusion_column"] = excl_col
    audit["label_counts_total"] = _counts(rows)
    audit["label_counts_eligible"] = _counts(eligible)
    audit["expected_total_ok"] = len(rows) == cfg["cohort"]["expected_total_subjects"]
    audit["expected_eligible_ok"] = len(eligible) == cfg["cohort"]["expected_eligible"]

    return {"rows": rows, "eligible": eligible, "audit": audit}


def _counts(rows):
    out = {}
    for r in rows:
        out[r["Histology"]] = out.get(r["Histology"], 0) + 1
    return out


def cohort_fields(cfg: dict):
    """Column order of the cohort CSV; two columns are ROI-named."""
    return ["PatientID", "Histology", "label", "eligible",
            cohort_setting(cfg, "exclusion_field"), "roi_present", "ct_present",
            "exclusion_reason", cohort_setting(cfg, "roi_volume_field"),
            "gtv_volume_mm3", "flagged_for_review", "flag_reason"]


def write_cohort(cfg: dict, cohort: dict) -> str:
    COHORT_FIELDS = cohort_fields(cfg)
    out = project_path(cfg, *cfg["paths"]["cohort_file"].split("/"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COHORT_FIELDS)
        w.writeheader()
        for r in sorted(cohort["rows"], key=lambda x: x["PatientID"]):
            w.writerow({k: r[k] for k in COHORT_FIELDS})
    return out


def load_cohort(cfg: dict):
    path = project_path(cfg, *cfg["paths"]["cohort_file"].split("/"))
    rows = _read_csv(path)
    for r in rows:
        r["label"] = int(r["label"])
        r["eligible"] = _is_true(r["eligible"])
    return rows


if __name__ == "__main__":
    import json
    cfg = load_config()
    c = build_cohort(cfg)
    p = write_cohort(cfg, c)
    print(json.dumps(c["audit"], indent=2))
    print("written:", p)
