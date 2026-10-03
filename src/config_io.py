"""Configuration loading and hashing.

Scientific parameters are never hard-coded in the pipeline modules; they are read
from config/experiment.yaml.  The extraction-relevant subset is hashed so that a
change to any of it invalidates previously written patient bags (resume safety).
"""
from __future__ import annotations

import hashlib
import json
import os

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "config", "experiment.yaml")


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = os.path.abspath(path)
    cfg["_project_root"] = PROJECT_ROOT
    return cfg


def project_path(cfg: dict, *parts: str) -> str:
    return os.path.join(cfg["_project_root"], *parts)


def load_radiomics_params(cfg: dict) -> dict:
    path = project_path(cfg, *cfg["radiomics"]["params_file"].split("/"))
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def extraction_signature(cfg: dict) -> str:
    """Stable hash over everything that changes the numeric content of a bag."""
    relevant = {
        "roi": cfg["experiment"]["roi"],
        "roi_filename": cfg["experiment"]["roi_filename"],
        "preprocessing": cfg["preprocessing"],
        "patches": cfg["patches"],
        "radiomics": {k: v for k, v in cfg["radiomics"].items() if k != "params_file"},
        "radiomics_params": load_radiomics_params(cfg),
        "histology_map": cfg["cohort"]["histology_map"],
    }
    blob = json.dumps(relevant, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# --------------------------------------------------------------------------
# ROI-generic helpers (Phase 4).
#
# The pipeline is shared between ROIs; everything ROI-specific is a config key.
# Each helper falls back to the value the Gradient configuration used before
# Phase 4, so config/experiment.yaml keeps behaving exactly as it did.
# --------------------------------------------------------------------------

_PATH_DEFAULTS = {
    "pilot_file": "config/pilot_patients.txt",
    "extraction_qa_csv": "reports/extraction_qa_{tag}.csv",
    "extraction_summary_json": "reports/extraction_summary_{tag}.json",
    "cohort_qa_csv": "reports/phase2_cohort_qa.csv",
    "cohort_qa_json": "reports/phase2_cohort_qa.json",
    "geometry_csv": "reports/geometry_check.csv",
    "geometry_json": "reports/geometry_check_summary.json",
    "qa_patches_dir": "qa/gradient_patches",
}

_COHORT_DEFAULTS = {
    # metadata column carrying the ROI-specific exclusion flag
    "exclusion_column": "Excluded_From_Gradient",
    # metadata column carrying the ROI volume, and the cohort-CSV column it feeds
    "roi_volume_column": "Gradient_Volume_mm3",
    "roi_volume_field": "gradient_volume_mm3",
    # cohort-CSV column carrying the exclusion flag
    "exclusion_field": "excluded_from_gradient",
    # config key holding the declared exclusion list
    "exclusions_key": "gradient_exclusions",
}


def cfg_path(cfg: dict, key: str, **fmt) -> str:
    """Absolute path for a named project-owned output, with a Gradient default."""
    rel = cfg.get("paths", {}).get(key, _PATH_DEFAULTS[key])
    if fmt:
        rel = rel.format(**fmt)
    return project_path(cfg, *rel.split("/"))


def cohort_setting(cfg: dict, key: str):
    return cfg.get("cohort", {}).get(key, _COHORT_DEFAULTS[key])


def declared_exclusions(cfg: dict):
    key = cohort_setting(cfg, "exclusions_key")
    return set(cfg["cohort"][key])


def roi_label(cfg: dict) -> str:
    return cfg["experiment"]["roi"]
