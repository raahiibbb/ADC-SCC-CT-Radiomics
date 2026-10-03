"""Phase 9A - third-cohort feasibility, selection and reservation.

    ./.venv/Scripts/python.exe src/phase9a_audit.py --config config/phase9a.yaml --metadata
    ./.venv/Scripts/python.exe src/phase9a_audit.py --config config/phase9a.yaml --audit

`--metadata` fetches, and caches, the official TCIA series listing for every
development cohort and every candidate collection.  `--audit` derives the
feasibility summary, the pairwise independence comparison and the candidate
table from the cached metadata alone.

Phase 9A is DATASET FEASIBILITY ONLY.  This module trains nothing, computes no
ADC/SCC classification metric, extracts no radiomic feature from any candidate,
runs no segmentation model, and makes no claim about how any model would perform
on any candidate.  See docs/PHASE9_EXTERNAL_RESERVATION_PROTOCOL.md.

No histology label is read here at all: eligibility counts reported by the
audit are literature CONTEXT carried through from the config and are labelled as
such.  Nothing in this module branches on a class label.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = os.path.abspath(path)
    cfg["_project_root"] = PROJECT_ROOT
    ac = cfg.get("assessment_config")
    if ac:
        ap = os.path.join(PROJECT_ROOT, *ac.split("/"))
        with open(ap, "r", encoding="utf-8") as fh:
            cfg["assessment"] = yaml.safe_load(fh)["assessment"]
        cfg["_assessment_path"] = os.path.abspath(ap)
    return cfg


def ppath(cfg, key_or_rel):
    rel = cfg["paths"].get(key_or_rel, key_or_rel)
    if os.path.isabs(rel):
        return rel
    return os.path.join(cfg["_project_root"], *rel.split("/"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def protocol_sha(cfg):
    return sha256_file(os.path.join(cfg["_project_root"],
                                    *cfg["protocol"]["path"].split("/")))


def verify_protocol(cfg):
    """The frozen protocol must be byte-identical to what the config records."""
    actual, expected = protocol_sha(cfg), cfg["protocol"]["sha256"]
    if actual != expected:
        raise SystemExit(
            "PROTOCOL HASH MISMATCH\n  expected %s\n  actual   %s\n"
            "The Phase-9 reservation protocol was frozen before the audit and "
            "must not change." % (expected, actual))
    return actual


def slug(name):
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in name]
    return "".join(keep).strip("_").lower()


# ---------------------------------------------------------------------------
# TCIA access - read-only, metadata only
# ---------------------------------------------------------------------------

def _get(cfg, url):
    dl = cfg["source"]["download"]
    last = None
    for attempt in range(int(dl["retries"]) + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": cfg["source"]["user_agent"]})
            with urllib.request.urlopen(req, timeout=dl["timeout_s"]) as resp:
                if resp.status != 200:
                    raise RuntimeError("HTTP %s" % resp.status)
                return resp.read()
        except Exception as exc:                                   # noqa: BLE001
            last = exc
            if attempt < int(dl["retries"]):
                time.sleep(float(dl["retry_sleep_s"]) * (attempt + 1))
    raise RuntimeError("GET failed after %d attempts: %s (%s)"
                       % (int(dl["retries"]) + 1, url, str(last)[:300]))


def api_get(cfg, endpoint, params=None):
    url = "%s/%s" % (cfg["source"]["nbia_api"].rstrip("/"), endpoint)
    if params:
        url = "%s?%s" % (url, urllib.parse.urlencode(params))
    return json.loads(_get(cfg, url).decode("utf-8"))


# ---------------------------------------------------------------------------
# metadata fetch
# ---------------------------------------------------------------------------

def all_collections(cfg):
    return sorted(c["Collection"] for c in api_get(cfg, "getCollectionValues"))


def fetch_collection(cfg, collection, audit_dir):
    """Fetch and cache the official series listing for one TCIA collection."""
    dest = os.path.join(audit_dir, slug(collection))
    os.makedirs(dest, exist_ok=True)
    series_path = os.path.join(dest, "series.json")
    if os.path.exists(series_path):
        with open(series_path, "r", encoding="utf-8") as fh:
            series = json.load(fh)
        cached = True
    else:
        series = api_get(cfg, "getSeries", {"Collection": collection})
        with open(series_path, "w", encoding="utf-8") as fh:
            json.dump(series, fh, indent=1, sort_keys=True)
        cached = False
    return series, series_path, cached


def summarise(cfg, collection, series):
    """Everything the feasibility table needs, derived from TCIA metadata only."""
    seg_mods = set(cfg["segmentation_modalities"])
    patients = sorted({s.get("PatientID") for s in series if s.get("PatientID")})
    studies = sorted({s.get("StudyInstanceUID") for s in series
                      if s.get("StudyInstanceUID")})
    modality = Counter(s.get("Modality") or "(missing)" for s in series)

    ct_patients = sorted({s["PatientID"] for s in series
                          if s.get("Modality") == "CT" and s.get("PatientID")})
    seg_patients = sorted({s["PatientID"] for s in series
                           if s.get("Modality") in seg_mods and s.get("PatientID")})
    both = sorted(set(ct_patients) & set(seg_patients))

    total_bytes = sum(int(s.get("FileSize") or 0) for s in series)
    ct_bytes = sum(int(s.get("FileSize") or 0) for s in series
                   if s.get("Modality") == "CT")

    manufacturers = Counter(s.get("Manufacturer") or "(missing)" for s in series
                            if s.get("Modality") == "CT")
    licenses = sorted({s.get("LicenseName") for s in series if s.get("LicenseName")})
    uris = sorted({s.get("CollectionURI") for s in series if s.get("CollectionURI")})

    return {
        "collection": collection,
        "n_series": len(series),
        "n_patients": len(patients),
        "patients": patients,
        "n_studies": len(studies),
        "studies": studies,
        "modality_counts": dict(sorted(modality.items(), key=lambda kv: -kv[1])),
        "n_patients_with_ct": len(ct_patients),
        "n_patients_with_segmentation": len(seg_patients),
        "n_patients_with_ct_and_segmentation": len(both),
        "segmentation_modalities_present": sorted(
            m for m in modality if m in seg_mods),
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / (1 << 30), 3),
        "ct_bytes": ct_bytes,
        "ct_gib": round(ct_bytes / (1 << 30), 3),
        "ct_manufacturers": dict(sorted(manufacturers.items(), key=lambda kv: -kv[1])),
        "licenses": licenses,
        "collection_uris": uris,
    }


def overlap(a, b):
    """Pairwise identifier overlap - protocol section 4.2 evidence ranks 1 and 2."""
    pa, pb = set(a["patients"]), set(b["patients"])
    sa, sb = set(a["studies"]), set(b["studies"])
    shared_p, shared_s = sorted(pa & pb), sorted(sa & sb)
    if not shared_p and not shared_s:
        rank, verdict = 1, "disjoint PatientIDs AND disjoint StudyInstanceUIDs"
    elif not shared_p:
        rank, verdict = 2, "disjoint PatientIDs but %d shared StudyInstanceUID(s)" % len(shared_s)
    else:
        rank, verdict = 5, "%d shared PatientID(s)" % len(shared_p)
    return {
        "a": a["collection"], "b": b["collection"],
        "n_shared_patient_ids": len(shared_p),
        "n_shared_study_uids": len(shared_s),
        "shared_patient_ids": shared_p[:50],
        "shared_study_uids": shared_s[:50],
        "evidence_rank": rank, "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------

def collections_of(entry):
    return entry.get("tcia_collections") or [entry["tcia_collection"]]


def run_metadata(cfg):
    verify_protocol(cfg)
    audit_dir = ppath(cfg, "audit_dir")
    os.makedirs(audit_dir, exist_ok=True)

    known = all_collections(cfg)
    with open(os.path.join(audit_dir, "tcia_collection_values.json"), "w",
              encoding="utf-8") as fh:
        json.dump(known, fh, indent=1)
    print("TCIA collections listed: %d" % len(known))

    wanted = []
    for e in cfg["development_cohorts"] + cfg["candidates"]:
        for c in collections_of(e):
            if c not in wanted:
                wanted.append(c)

    missing = [c for c in wanted if c not in known]
    for c in missing:
        print("  WARNING: %r is not in the official collection listing" % c)

    for c in wanted:
        if c in missing:
            continue
        series, path, cached = fetch_collection(cfg, c, audit_dir)
        print("  %-34s %5d series  %s" % (c, len(series),
                                          "(cached)" if cached else "(fetched)"))
    return missing


def run_audit(cfg):
    t0 = time.time()
    proto = verify_protocol(cfg)
    audit_dir = ppath(cfg, "audit_dir")
    known_path = os.path.join(audit_dir, "tcia_collection_values.json")
    with open(known_path, "r", encoding="utf-8") as fh:
        known = json.load(fh)

    summaries, missing = {}, []
    for e in cfg["development_cohorts"] + cfg["candidates"]:
        for c in collections_of(e):
            if c not in known:
                missing.append(c)
                continue
            if c in summaries:
                continue
            series, _, _ = fetch_collection(cfg, c, audit_dir)
            summaries[c] = summarise(cfg, c, series)

    # ---- per-entry rollup (a candidate may span >1 collection) -------------
    def rollup(entry, kind):
        cols = [c for c in collections_of(entry) if c in summaries]
        if not cols:
            return None
        pats, studs = [], []
        mod, man = Counter(), Counter()
        tb = cb = 0
        segp = ctp = both = 0
        lic, uri = set(), set()
        for c in cols:
            s = summaries[c]
            pats += s["patients"]
            studs += s["studies"]
            mod.update(s["modality_counts"])
            man.update(s["ct_manufacturers"])
            tb += s["total_bytes"]
            cb += s["ct_bytes"]
            segp += s["n_patients_with_segmentation"]
            ctp += s["n_patients_with_ct"]
            both += s["n_patients_with_ct_and_segmentation"]
            lic.update(s["licenses"])
            uri.update(s["collection_uris"])
        seg_mods = sorted(set(mod) & set(cfg["segmentation_modalities"]))
        return {
            "key": entry["key"],
            "label": entry.get("label", entry.get("tcia_collection")),
            "kind": kind,
            "tcia_collections": cols,
            "institution": entry.get("institution", ""),
            "dois": entry.get("dois") or [entry.get("doi")],
            "n_patients": len(set(pats)),
            "n_studies": len(set(studs)),
            "n_series": sum(summaries[c]["n_series"] for c in cols),
            "modality_counts": dict(sorted(mod.items(), key=lambda kv: -kv[1])),
            "n_patients_with_ct": ctp,
            "n_patients_with_segmentation": segp,
            "n_patients_with_ct_and_segmentation": both,
            "segmentation_modalities_present": seg_mods,
            "has_public_segmentation": bool(seg_mods),
            "total_gib": round(tb / (1 << 30), 3),
            "ct_gib": round(cb / (1 << 30), 3),
            "ct_manufacturers": dict(sorted(man.items(), key=lambda kv: -kv[1])),
            "licenses": sorted(lic),
            "collection_uris": sorted(uri),
            "literature_context": entry.get("literature_context", {}),
            "patients": sorted(set(pats)),
            "studies": sorted(set(studs)),
        }

    entries = []
    for e in cfg["development_cohorts"]:
        r = rollup(e, "development")
        if r:
            r["consumed_by"] = e.get("consumed_by", "")
            r["eligible_adc"] = e.get("eligible_adc")
            r["eligible_scc"] = e.get("eligible_scc")
            entries.append(r)
    for e in cfg["candidates"]:
        r = rollup(e, "candidate")
        if r:
            entries.append(r)

    # ---- independence: every candidate against both development cohorts ----
    by_key = {e["key"]: e for e in entries}
    dev_keys = [e["key"] for e in entries if e["kind"] == "development"]
    cand_keys = [e["key"] for e in entries if e["kind"] == "candidate"]

    overlaps = []
    for ck in cand_keys:
        for dk in dev_keys:
            o = overlap({"collection": ck, "patients": by_key[ck]["patients"],
                         "studies": by_key[ck]["studies"]},
                        {"collection": dk, "patients": by_key[dk]["patients"],
                         "studies": by_key[dk]["studies"]})
            overlaps.append(o)
    # candidate-vs-candidate as well: differently named is not automatically
    # independent (protocol section 4).
    for i, ck in enumerate(cand_keys):
        for dk in cand_keys[i + 1:]:
            overlaps.append(overlap(
                {"collection": ck, "patients": by_key[ck]["patients"],
                 "studies": by_key[ck]["studies"]},
                {"collection": dk, "patients": by_key[dk]["patients"],
                 "studies": by_key[dk]["studies"]}))

    # ---- same-centre flag (protocol section 4.1) --------------------------
    flags = [f.lower() for f in cfg["independence"]["development_institutions_to_flag"]]
    for e in entries:
        inst = (e["institution"] or "").lower()
        e["same_centre_as_development"] = any(f in inst for f in flags)

    # ---- size rule (protocol section 2.1), on LITERATURE context only -----
    thr = cfg["eligibility"]["primary_external_test"]
    for e in entries:
        lc = e.get("literature_context") or {}
        adc, scc = lc.get("adc"), lc.get("scc")
        if adc is None or scc is None:
            e["size_rule"] = {"evaluable": False,
                              "reason": "no ADC/SCC count in the supplied literature"}
        else:
            total, minority = adc + scc, min(adc, scc)
            e["size_rule"] = {
                "evaluable": True, "source": lc.get("source"),
                "total": total, "minority": minority,
                "meets_total": total >= thr["min_total_adc_scc"],
                "meets_minority": minority >= thr["min_minority_class"],
                "meets_rule": (total >= thr["min_total_adc_scc"]
                               and minority >= thr["min_minority_class"]),
            }

    # ---- write artefacts --------------------------------------------------
    os.makedirs(audit_dir, exist_ok=True)
    with open(os.path.join(audit_dir, "collection_summaries.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=1, sort_keys=True)
    with open(os.path.join(audit_dir, "independence_overlaps.json"), "w",
              encoding="utf-8") as fh:
        json.dump(overlaps, fh, indent=1)
    with open(os.path.join(audit_dir, "entry_rollups.json"), "w",
              encoding="utf-8") as fh:
        json.dump(entries, fh, indent=1)

    # ---- the feasibility table (protocol section 5) -----------------------
    #
    # Columns are fixed by the protocol.  No model-performance estimate may
    # appear here, and none is computed anywhere in Phase 9A.
    assess = cfg.get("assessment", {})
    cols = ["candidate", "total_adc", "total_scc", "pre_treatment_ct",
            "segmentation_availability", "segmentation_provenance",
            "institution_independence", "possible_patient_overlap",
            "acquisition_diversity", "download_size",
            "technical_processing_difficulty", "manual_expert_work_required",
            "suitable_as_primary_external_test"]
    codes = cfg["segmentation_codes"]
    rows = []
    for e in entries:
        if e["kind"] != "candidate":
            continue
        a = assess.get(e["key"], {})
        lc = e.get("literature_context") or {}
        ov = [o for o in overlaps
              if o["a"] == e["key"] and o["b"] in dev_keys]
        shared = sum(o["n_shared_patient_ids"] for o in ov)
        rank = max([o["evidence_rank"] for o in ov] or [5])
        code = a.get("segmentation_code", "")
        rows.append({
            "candidate": e["label"],
            "total_adc": lc.get("adc", ""),
            "total_scc": lc.get("scc", ""),
            "pre_treatment_ct": a.get("pre_treatment_ct", ""),
            "segmentation_availability": "%s - %s" % (code, codes.get(code, "")),
            "segmentation_provenance": a.get("segmentation_provenance", ""),
            "institution_independence": a.get("institution_independence", ""),
            "possible_patient_overlap":
                "%d shared PatientID(s) with a development cohort; "
                "identifier-evidence rank %d" % (shared, rank),
            "acquisition_diversity": a.get("acquisition_diversity", ""),
            "download_size": "%.2f GiB (%d patients, %d series)"
                             % (e["total_gib"], e["n_patients"], e["n_series"]),
            "technical_processing_difficulty":
                a.get("technical_processing_difficulty", ""),
            "manual_expert_work_required": a.get("manual_expert_work_required", ""),
            "suitable_as_primary_external_test":
                a.get("suitable_as_primary_external_test", ""),
        })
    with open(ppath(cfg, "candidate_table"), "w", encoding="utf-8",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- the STOP rule (protocol section 7) -------------------------------
    # A cohort can be the PRIMARY external test only if it satisfies the size
    # rule AND already carries code A or B segmentation.
    qualified = []
    for e in entries:
        if e["kind"] != "candidate":
            continue
        a = assess.get(e["key"], {})
        sr = e.get("size_rule", {})
        if a.get("segmentation_code") in ("A", "B") and sr.get("meets_rule"):
            qualified.append(e["key"])
    stop = {
        "qualified_primary_external_candidates": qualified,
        "stop_rule_triggered": len(qualified) == 0,
        "verbatim_finding": (
            "No adequately sized independent public ADC/SCC cohort with "
            "directly reusable primary-tumor segmentation was identified."
            if not qualified else ""),
        "pilot_downloaded": False,
        "pilot_reason": (
            "protocol section 6 permits a pilot ONLY for a candidate that "
            "already satisfies the size rule AND carries code A or B "
            "segmentation; no candidate does"
            if not qualified else ""),
    }

    run = {
        "phase": "9A",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "protocol_path": cfg["protocol"]["path"],
        "protocol_sha256": proto,
        "config_path": os.path.relpath(cfg["_config_path"], cfg["_project_root"]),
        "config_sha256": sha256_file(cfg["_config_path"]),
        "assessment_config_path": cfg.get("assessment_config"),
        "assessment_config_sha256": (sha256_file(cfg["_assessment_path"])
                                     if cfg.get("_assessment_path") else None),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "tcia_api": cfg["source"]["nbia_api"],
        "n_tcia_collections_listed": len(known),
        "collections_not_found": sorted(set(missing)),
        "n_collections_audited": len(summaries),
        "eligibility_thresholds": thr,
        "entries": [{k: v for k, v in e.items()
                     if k not in ("patients", "studies")} for e in entries],
        "independence_overlaps": overlaps,
        # Stated rather than hidden, exactly as the Phase-8B protocol stated
        # its analogous CT-hash limitation (section 4.2 of that document).
        "identifier_evidence_limitation": {
            "exact_id_disjointness_is_proof_of_independence": False,
            "statement": (
                "Every TCIA collection is de-identified into its own "
                "PatientID and StudyInstanceUID namespace, so exact-string "
                "disjointness between two collections is guaranteed a priori "
                "and carries almost no information about whether the same "
                "physical patient appears in both. It is reported as evidence "
                "rank 1 under the protocol's hierarchy, but rank 1 here means "
                "'no identifier contradiction found', not 'non-overlap "
                "verified'. The concrete counter-example found in this audit "
                "is QIN LUNG CT versus LungCT-Diagnosis: exact-string "
                "disjoint, yet 24 subjects share the same R-number under two "
                "different formats at the same institution."),
            "counter_example": {
                "collections": ["QIN LUNG CT", "LungCT-Diagnosis"],
                "institution": "H. Lee Moffitt Cancer Center",
                "n_shared_r_numbers": 24,
            },
            "consequence": (
                "For a same-centre candidate (Lung3, Interobserver1) the "
                "independence claim rests on institution-level provenance "
                "only, which the protocol records as a defect."),
        },
        "stop_rule": stop,
        "scope": {
            "models_trained": 0,
            "classification_metrics_computed": 0,
            "radiomic_features_extracted_on_candidates": 0,
            "segmentation_models_run": 0,
            "harmonisations_fitted": 0,
            "candidate_images_downloaded": 0,
        },
        "elapsed_s": round(time.time() - t0, 2),
    }
    with open(ppath(cfg, "run_record"), "w", encoding="utf-8") as fh:
        json.dump(run, fh, indent=1)

    print("\nAudited %d TCIA collections across %d entries in %.1fs"
          % (len(summaries), len(entries), time.time() - t0))
    for e in entries:
        a = assess.get(e["key"], {})
        sr = e.get("size_rule", {})
        print("  %-12s %-34s n=%4d  seg=%-13s code=%-2s size_rule=%-11s %s"
              % (e["kind"], e["label"], e["n_patients"],
                 ",".join(e["segmentation_modalities_present"]) or "none",
                 a.get("segmentation_code", "-"),
                 ("PASS" if sr.get("meets_rule") else "fail")
                 if sr.get("evaluable") else "n/a",
                 "SAME-CENTRE" if e["same_centre_as_development"] else ""))
    print("\nSTOP RULE TRIGGERED: %s" % stop["stop_rule_triggered"])
    if stop["stop_rule_triggered"]:
        print("  %s" % stop["verbatim_finding"])
    print("Pilot downloaded: %s" % stop["pilot_downloaded"])
    return run


def main():
    ap = argparse.ArgumentParser(description="Phase 9A third-cohort feasibility audit")
    ap.add_argument("--config", default="config/phase9a.yaml")
    ap.add_argument("--metadata", action="store_true",
                    help="fetch and cache official TCIA collection metadata")
    ap.add_argument("--audit", action="store_true",
                    help="derive summaries, independence and the run record")
    args = ap.parse_args()

    cfg = load_cfg(os.path.join(PROJECT_ROOT, *args.config.split("/"))
                   if not os.path.isabs(args.config) else args.config)
    if args.metadata:
        run_metadata(cfg)
    if args.audit:
        run_audit(cfg)
    if not (args.metadata or args.audit):
        ap.error("choose --metadata and/or --audit")


if __name__ == "__main__":
    main()
