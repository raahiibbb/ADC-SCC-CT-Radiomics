"""Phase 8A - acquisition / domain-shift descriptors, DESCRIPTIVE ONLY.

Collects the non-label acquisition metadata of the external CT series and puts
it beside the corresponding LUNG1 distributions, which are read from the
read-only `Dataset/Lung1 clinical/dataset_metadata_master.csv` and the frozen
Phase-4 geometry QA.  Nothing here is fitted, corrected or harmonised:

  * ComBat is NOT fitted (`batch_descriptors.combat: false`);
  * no site classifier is trained;
  * the two cohorts are never combined into one modelling table;
  * no ADC/SCC label is used, and the summaries are not stratified by label.

    python src/phase8a_batch.py --config config/phase8a_external.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase8a_tcia import load_cfg, ppath                            # noqa: E402


def _num(s):
    try:
        return float(str(s).split("\\")[0])
    except (TypeError, ValueError):
        return np.nan


def _dist(series, top=12):
    s = series.dropna().astype(str)
    s = s[s != ""]
    vc = s.value_counts()
    return {"n_present": int(len(s)), "n_missing": int(len(series) - len(s)),
            "n_distinct": int(vc.size),
            "top": {str(k): int(v) for k, v in vc.head(top).items()}}


def _numeric_summary(values):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "min": float(v.min()), "p25": float(np.percentile(v, 25)),
            "median": float(np.median(v)), "p75": float(np.percentile(v, 75)),
            "max": float(v.max()), "mean": float(v.mean()), "sd": float(v.std(ddof=1))
            if v.size > 1 else 0.0,
            "n_distinct": int(np.unique(np.round(v, 6)).size)}


def external_table(cfg):
    """One row per INCLUDED external patient, from its own build record."""
    cohort = pd.read_csv(ppath(cfg, "cohort_csv"))
    inc = cohort[cohort["included"].astype(str).str.lower() == "true"]
    rows = []
    for pid in sorted(inc["PatientID"].astype(str)):
        rec_path = os.path.join(ppath(cfg, "masks_dir"), pid, "record.json")
        with open(rec_path, "r", encoding="utf-8") as fh:
            rec = json.load(fh)
        acq = dict(rec.get("acquisition", {}))
        h = rec["ct_header"]
        row = {"PatientID": pid, "cohort": "NSCLC-Radiogenomics",
               "ct_series_uid": rec["ct_series_uid"],
               "n_slices": h["n_slices"], "rows": h["rows"], "columns": h["columns"],
               "pixel_spacing_mm": h["pixel_spacing_mm"][0],
               "slice_spacing_mm": h["slice_spacing_mm_median"],
               "slice_spacing_uniform": abs(h["slice_spacing_mm_max"]
                                            - h["slice_spacing_mm_min"]) <= 0.01}
        row.update({k: acq.get(k, "") for k in cfg["batch_descriptors"]["dicom_tags"]})
        row["SliceThickness_mm"] = _num(acq.get("SliceThickness"))
        row["KVP_kV"] = _num(acq.get("KVP"))
        row["XRayTubeCurrent_mA"] = _num(acq.get("XRayTubeCurrent"))
        row["ExposureTime_ms"] = _num(acq.get("ExposureTime"))
        row["contrast_recorded"] = bool(str(acq.get("ContrastBolusAgent", "")).strip())
        rows.append(row)
    return pd.DataFrame(rows)


def lung1_table(cfg):
    """The LUNG1 GTV+Rim cohort's acquisition descriptors, read-only."""
    master = pd.read_csv(cfg["batch_descriptors"]["lung1_reference"]["metadata_csv"])
    keep = master[(~master["Excluded_From_GTV_Rim"].astype(bool))
                  & (master["Histology"].astype(str).str.lower()
                     .isin(["adenocarcinoma", "squamous cell carcinoma"]))].copy()
    out = pd.DataFrame({
        "PatientID": keep["PatientID"].astype(str),
        "cohort": "LUNG1",
        "n_slices": keep["CT_NumSlices"],
        "rows": keep["CT_Rows"], "columns": keep["CT_Columns"],
        "pixel_spacing_mm": keep["CT_PixelSpacing_X_mm"],
        "slice_spacing_mm": keep["CT_SliceThickness_mm"],
        "Manufacturer": keep["CT_Manufacturer"].astype(str),
        "ManufacturerModelName": keep["CT_ManufacturerModelName"].astype(str),
        "ConvolutionKernel": keep["CT_ConvolutionKernel"].astype(str),
        "SliceThickness_mm": keep["CT_SliceThickness_mm"],
        "KVP_kV": keep["CT_KVP"],
        "XRayTubeCurrent_mA": keep["CT_XRayTubeCurrent_mA"],
        "StudyDescription": keep["StudyDescription"].astype(str),
    })
    return out.reset_index(drop=True)


def summarise(df, cfg):
    cat = ["Manufacturer", "ManufacturerModelName", "ConvolutionKernel",
           "StudyDescription", "SeriesDescription", "PatientSex", "ContrastBolusAgent"]
    num = ["n_slices", "rows", "columns", "pixel_spacing_mm", "slice_spacing_mm",
           "SliceThickness_mm", "KVP_kV", "XRayTubeCurrent_mA", "ExposureTime_ms"]
    out = {"n_patients": int(len(df)), "categorical": {}, "numeric": {}}
    for c in cat:
        if c in df.columns:
            out["categorical"][c] = _dist(df[c])
    for c in num:
        if c in df.columns:
            out["numeric"][c] = _numeric_summary(pd.to_numeric(df[c], errors="coerce"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)

    if bool(cfg["batch_descriptors"]["combat"]):
        raise RuntimeError("batch_descriptors.combat must stay false in Phase 8A")

    ext = external_table(cfg)
    ext.to_csv(ppath(cfg, "acquisition_csv"), index=False)

    l1 = lung1_table(cfg)
    summary = {
        "note": ("DESCRIPTIVE ONLY.  No harmonisation, no ComBat, no site "
                 "classifier, no combined table, no label stratification."),
        "external": summarise(ext, cfg),
        "lung1_gtv_rim_adc_scc": summarise(l1, cfg),
    }
    a, b = summary["external"]["numeric"], summary["lung1_gtv_rim_adc_scc"]["numeric"]
    summary["headline_differences"] = {
        k: {"external_median": a.get(k, {}).get("median"),
            "lung1_median": b.get(k, {}).get("median")}
        for k in ("pixel_spacing_mm", "slice_spacing_mm", "SliceThickness_mm",
                  "n_slices", "KVP_kV")
    }
    with open(ppath(cfg, "acquisition_summary_json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({"external": summary["external"]["numeric"],
                      "lung1": summary["lung1_gtv_rim_adc_scc"]["numeric"]}, indent=1))
    print(json.dumps(summary["external"]["categorical"], indent=1)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
