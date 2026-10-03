"""Phase 7A - whole-region (GLOBAL) radiomics extraction for GTV and rim.

ONE patient contributes exactly ONE feature row per region.  There is no patch
lattice here and no local radiomics: Phase 7A is the whole-ROI control the
project never ran.

    CT.nii.gz + ROI_A_GTV.nii.gz + ROI_E_GTV_Rim.nii.gz
      -> 2 mm isotropic resampling (CT linear, masks nearest, ONE shared grid)
      -> GTV, and rim = (GTV+Rim) MINUS GTV                (phase7a_regions)
      -> PyRadiomics, Original + LoG(2,3,4 mm) + Wavelet(level 1)
      -> one row per region, cached to disk

Resumable: a cached patient carrying the current extraction signature is
skipped.  Modelling NEVER re-runs PyRadiomics - it reads gtv.csv / rim.csv.

    python src/phase7a_extract.py --config config/phase7a.yaml --pilot
    python src/phase7a_extract.py --config config/phase7a.yaml --all
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import radiomics                                          # noqa: E402
from radiomics import featureextractor                    # noqa: E402

from phase7a_regions import build_regions                  # noqa: E402

radiomics.setVerbosity(logging.ERROR)
logging.getLogger("radiomics").setLevel(logging.ERROR)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CACHE_VERSION = 1


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = os.path.abspath(path)
    cfg["_project_root"] = PROJECT_ROOT
    return cfg


def ppath(cfg, key_or_rel):
    rel = cfg["paths"].get(key_or_rel, key_or_rel)
    return os.path.join(cfg["_project_root"], *rel.split("/"))


def load_pyradiomics_params(cfg):
    with open(ppath(cfg, "pyradiomics_params"), "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def extraction_signature(cfg):
    """Stable hash over everything that changes the numeric content of a row."""
    relevant = {
        "regions": cfg["regions"],
        "preprocessing": cfg["preprocessing"],
        "radiomics": {k: v for k, v in cfg["radiomics"].items() if k != "params_file"},
        "pyradiomics_params": load_pyradiomics_params(cfg),
        "cache_version": CACHE_VERSION,
    }
    blob = json.dumps(relevant, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def build_extractor(cfg, region):
    """The GTV extractor is the params file verbatim; the rim extractor is the
    SAME file with the single key featureClass.shape removed.  That one deletion
    is the entire implementation of "no rim shape features"."""
    params = copy.deepcopy(load_pyradiomics_params(cfg))
    if not bool(cfg["regions"][region]["shape_features"]):
        params["featureClass"].pop("shape", None)
    return featureextractor.RadiomicsFeatureExtractor(params)


# ---------------------------------------------------------------------------
# cohort
# ---------------------------------------------------------------------------

def load_cohort(cfg):
    df = pd.read_csv(ppath(cfg, "cohort_file"))
    keep = df[(df["eligible"]) & (~df["excluded_from_gtv_rim"])].copy()
    keep = keep.sort_values("PatientID").reset_index(drop=True)
    hm = cfg["cohort"]["histology_map"]
    keep["label"] = keep["Histology"].map(hm).astype(int)
    n, adc, scc = len(keep), int((keep.label == 1).sum()), int((keep.label == 0).sum())
    exp = cfg["cohort"]
    if (n, adc, scc) != (exp["expected_patients"], exp["expected_adc"], exp["expected_scc"]):
        raise RuntimeError("cohort is %d/%d ADC/%d SCC, expected %d/%d/%d"
                           % (n, adc, scc, exp["expected_patients"],
                              exp["expected_adc"], exp["expected_scc"]))
    excl = set(cfg["cohort"]["exclusions"])
    if excl & set(keep["PatientID"]):
        raise RuntimeError("an excluded patient is in the cohort: %s" % (excl & set(keep.PatientID)))
    return keep


def pilot_patients(cfg, cohort):
    """Deterministic and label-blind: smallest and largest GTV volume, the first
    and last PatientID, and two evenly spaced interior patients."""
    n_want = int(cfg["pilot"]["n_patients"])
    df = cohort.sort_values("gtv_volume_mm3").reset_index(drop=True)
    picks = [df.PatientID.iloc[0], df.PatientID.iloc[-1]]
    s = cohort.sort_values("PatientID").reset_index(drop=True)
    picks += [s.PatientID.iloc[0], s.PatientID.iloc[-1]]
    for frac in (1.0 / 3.0, 2.0 / 3.0):
        picks.append(s.PatientID.iloc[int(round(frac * (len(s) - 1)))])
    out = []
    for p in picks:
        if p not in out:
            out.append(p)
    i = 0
    while len(out) < n_want and i < len(s):
        if s.PatientID.iloc[i] not in out:
            out.append(s.PatientID.iloc[i])
        i += 1
    return out[:n_want]


# ---------------------------------------------------------------------------
# one patient
# ---------------------------------------------------------------------------

def _clean(result, drop_prefixes):
    """PyRadiomics output -> {feature_name: float}, diagnostics dropped."""
    out = {}
    for k, v in result.items():
        if any(k.startswith(p) for p in drop_prefixes):
            continue
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            out[str(k)] = float("nan")
    return out


def extract_patient(cfg, patient_id, extractors, signature):
    t0 = time.time()
    reg = build_regions(cfg, patient_id)
    t_regions = time.time() - t0

    drop = tuple(cfg["radiomics"]["drop_column_prefixes"])
    rec = {"schema": CACHE_VERSION, "PatientID": patient_id,
           "extraction_signature": signature, "geometry": reg.geometry,
           "regions": {}, "timing": {"regions_s": round(t_regions, 3)}}

    for region, extractor in extractors.items():
        t1 = time.time()
        res = extractor.execute(reg.ct, reg.masks[region], label=1)
        feats = _clean(res, drop)
        names = sorted(feats)
        vals = np.asarray([feats[k] for k in names], dtype=np.float64)
        rec["regions"][region] = {
            "feature_names": names,
            "values": [None if not np.isfinite(v) else float(v) for v in vals],
            "n_features": len(names),
            "n_nan": int(np.isnan(vals).sum()),
            "n_inf": int(np.isinf(vals).sum()),
            "n_mask_voxels": int(sitk.GetArrayViewFromImage(reg.masks[region]).sum()),
        }
        rec["timing"]["%s_s" % region] = round(time.time() - t1, 3)

    rec["timing"]["total_s"] = round(time.time() - t0, 3)
    return rec


def cache_path(cfg, patient_id):
    return os.path.join(ppath(cfg, "cache_dir"), "%s.json" % patient_id)


def cached_ok(cfg, patient_id, signature):
    p = cache_path(cfg, patient_id)
    if not os.path.isfile(p):
        return False
    try:
        with open(p, "r", encoding="utf-8") as fh:
            rec = json.load(fh)
    except Exception:
        return False
    if rec.get("extraction_signature") != signature or rec.get("schema") != CACHE_VERSION:
        return False
    for region in ("gtv", "rim"):
        r = rec.get("regions", {}).get(region)
        if not r or not r.get("feature_names") or len(r["feature_names"]) != len(r["values"]):
            return False
    return True


# ---------------------------------------------------------------------------
# matrices
# ---------------------------------------------------------------------------

def assemble(cfg, cohort, signature):
    """Read every cached patient and write gtv.csv / rim.csv + the manifest."""
    rows = {"gtv": [], "rim": []}
    geom_rows, manifest = [], []
    names_ref = {}
    for _, row in cohort.iterrows():
        pid = row.PatientID
        with open(cache_path(cfg, pid), "r", encoding="utf-8") as fh:
            rec = json.load(fh)
        if rec["extraction_signature"] != signature:
            raise RuntimeError("%s: stale cache signature" % pid)
        geom_rows.append(rec["geometry"])
        m = {"PatientID": pid, "Histology": row.Histology, "label": int(row.label),
             "extraction_signature": signature,
             "total_s": rec["timing"]["total_s"]}
        for region in ("gtv", "rim"):
            r = rec["regions"][region]
            names = r["feature_names"]
            if region not in names_ref:
                names_ref[region] = names
            elif names != names_ref[region]:
                raise RuntimeError("%s: %s feature names differ from the cohort reference" % (pid, region))
            vals = [np.nan if v is None else float(v) for v in r["values"]]
            d = {"PatientID": pid, "Histology": row.Histology, "label": int(row.label)}
            d.update(dict(zip(names, vals)))
            rows[region].append(d)
            m["%s_n_features" % region] = r["n_features"]
            m["%s_n_nan" % region] = r["n_nan"]
            m["%s_n_inf" % region] = r["n_inf"]
            m["%s_n_voxels" % region] = r["n_mask_voxels"]
        m["gtv_volume_mm3"] = rec["geometry"]["gtv_volume_mm3"]
        m["rim_volume_mm3"] = rec["geometry"]["rim_volume_mm3"]
        m["disjoint_ok"] = rec["geometry"]["disjoint_ok"]
        m["union_ok"] = rec["geometry"]["union_ok"]
        m["native_subset_ok"] = rec["geometry"]["native_subset_ok"]
        m["status"] = "ok"
        manifest.append(m)

    os.makedirs(ppath(cfg, "features_dir"), exist_ok=True)
    for region, path_key in (("gtv", "gtv_csv"), ("rim", "rim_csv")):
        df = pd.DataFrame(rows[region])
        cols = ["PatientID", "Histology", "label"] + names_ref[region]
        df = df[cols]
        df.to_csv(ppath(cfg, path_key), index=False)

    pd.DataFrame(geom_rows).to_csv(ppath(cfg, "geometry_csv"), index=False)
    os.makedirs(os.path.dirname(ppath(cfg, "manifest_csv")), exist_ok=True)
    pd.DataFrame(manifest).to_csv(ppath(cfg, "manifest_csv"), index=False)

    families = {}
    for region, names in names_ref.items():
        families[region] = feature_family_counts(names)
    with open(ppath(cfg, "feature_names_json"), "w", encoding="utf-8") as fh:
        json.dump({"extraction_signature": signature,
                   "gtv": names_ref["gtv"], "rim": names_ref["rim"],
                   "n_gtv": len(names_ref["gtv"]), "n_rim": len(names_ref["rim"]),
                   "family_counts": families}, fh, indent=1)
    return names_ref, families, manifest


def feature_family_counts(names):
    """{image_type: {family: count}} - the audit's view of the feature bank."""
    out = {}
    for n in names:
        parts = n.split("_")
        img = parts[0] if parts else "?"
        fam = parts[1] if len(parts) > 1 else "?"
        out.setdefault(img, {}).setdefault(fam, 0)
        out[img][fam] += 1
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(cfg, patients, signature, force=False, tag="run"):
    os.makedirs(ppath(cfg, "cache_dir"), exist_ok=True)
    extractors = {r: build_extractor(cfg, r) for r in ("gtv", "rim")}
    done, skipped, failures = 0, 0, []
    t0 = time.time()
    for i, pid in enumerate(patients, 1):
        if not force and cached_ok(cfg, pid, signature):
            skipped += 1
            print("[%3d/%3d] %-12s cached" % (i, len(patients), pid), flush=True)
            continue
        try:
            rec = extract_patient(cfg, pid, extractors, signature)
        except Exception as exc:                       # recorded, never silent
            failures.append({"PatientID": pid, "error": "%s: %s" % (type(exc).__name__, exc)})
            print("[%3d/%3d] %-12s FAILED  %s" % (i, len(patients), pid, exc), flush=True)
            continue
        tmp = cache_path(cfg, pid) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(rec, fh)
        os.replace(tmp, cache_path(cfg, pid))
        done += 1
        print("[%3d/%3d] %-12s ok  gtv=%d rim=%d feats  %.1fs"
              % (i, len(patients), pid, rec["regions"]["gtv"]["n_features"],
                 rec["regions"]["rim"]["n_features"], rec["timing"]["total_s"]), flush=True)
    return {"tag": tag, "n_requested": len(patients), "n_extracted": done,
            "n_cached_skipped": skipped, "failures": failures,
            "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/phase7a.yaml")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    sig = extraction_signature(cfg)
    cohort = load_cohort(cfg)
    print("extraction signature:", sig)
    print("cohort:", len(cohort), "patients,",
          int((cohort.label == 1).sum()), "ADC,", int((cohort.label == 0).sum()), "SCC")

    if args.patients:
        pats, tag = list(args.patients), "explicit"
    elif args.pilot:
        pats, tag = pilot_patients(cfg, cohort), "pilot"
    elif args.all:
        pats, tag = list(cohort.PatientID), "all"
    else:
        raise SystemExit("choose --pilot, --all or --patients")

    summary = run(cfg, pats, sig, force=args.force, tag=tag)
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=1))
    if summary["failures"]:
        print("FAILURES:", json.dumps(summary["failures"], indent=1))

    if tag == "all" and not summary["failures"]:
        names, families, manifest = assemble(cfg, cohort, sig)
        print("gtv features:", len(names["gtv"]), " rim features:", len(names["rim"]))
        print("family counts:", json.dumps(families, indent=1))

    out = ppath(cfg, "qa_dir")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "extraction_%s.json" % tag), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
