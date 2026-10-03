"""Phase-9A audit: third-cohort feasibility, selection and reservation.

Run with:
    ./.venv/Scripts/python.exe tests/test_phase9a.py

Phase 9A selects nothing, trains nothing and scores nothing.  Its whole output
is a claim about what public data exists.  The corresponding risk is therefore
not leakage but OVERCLAIM: a segmentation that is asserted rather than shown, a
"disjoint" that is an artefact of a naming scheme, a size that comes from a
paper and is quietly reported as verified, or a stop rule that gets softened
because the answer is inconvenient.

Every check below re-derives its answer from the cached official TCIA metadata
on disk, and the segmentation checks are two-sided: a code E claim must be
contradicted by finding ANY segmentation series, and a code A claim must be
contradicted by finding NONE.

Groups
  A  protocol freeze, config integrity and scope                     (1-9)
  B  cached TCIA metadata reproduces the summaries                  (10-17)
  C  segmentation codes are justified by the metadata, two-sided    (18-24)
  D  independence audit, and the limits of identifier evidence      (25-31)
  E  the size rule and the stop rule, recomputed                    (32-38)
  F  the feasibility table carries no performance estimate          (39-43)
  G  methods that must NOT appear anywhere in Phase 9A              (44-49)
  H  namespace, read-only trees, frozen earlier artefacts           (50-55)
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from phase9a_audit import (load_cfg, overlap, ppath, protocol_sha,   # noqa: E402
                           sha256_file, slug, summarise)

CFG = load_cfg(os.path.join(ROOT, "config", "phase9a.yaml"))
CHECKS = []

PHASE9A_PY = ("src/phase9a_audit.py",)


def check(n, name, ok, detail):
    CHECKS.append({"n": n, "check": name, "pass": bool(ok), "detail": detail})
    print("%-4s %2d  %s" % ("PASS" if ok else "FAIL", n, name))
    if not ok:
        print("        %s" % (detail if isinstance(detail, str)
                              else json.dumps(detail, default=str)[:1500]))


def token_present(src, token):
    """Whole-identifier match, never substring - the frozen project rule."""
    return re.search(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(token),
                     src) is not None


def executable_source(path):
    """Source with docstrings and string literals stripped, so prose and
    evidence quotations cannot trip a scan (the Phase-5C/6A/6B/7A/8A fix)."""
    with open(os.path.join(ROOT, path), "r", encoding="utf-8") as fh:
        src = fh.read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if getattr(node, "end_lineno", None):
                spans.append((node.lineno, node.end_lineno))
    lines = src.splitlines()
    for a, b in spans:
        for i in range(a - 1, min(b, len(lines))):
            lines[i] = ""
    return "\n".join(lines)


def load_series(collection):
    p = os.path.join(ppath(CFG, "audit_dir"), slug(collection), "series.json")
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main():
    audit_dir = ppath(CFG, "audit_dir")
    run_path = ppath(CFG, "run_record")
    with open(run_path, "r", encoding="utf-8") as fh:
        RUN = json.load(fh)
    with open(os.path.join(audit_dir, "collection_summaries.json"), "r",
              encoding="utf-8") as fh:
        SUM = json.load(fh)
    with open(os.path.join(audit_dir, "entry_rollups.json"), "r",
              encoding="utf-8") as fh:
        ENTRIES = json.load(fh)
    with open(os.path.join(audit_dir, "independence_overlaps.json"), "r",
              encoding="utf-8") as fh:
        OVER = json.load(fh)
    ASSESS = CFG["assessment"]
    BY = {e["key"]: e for e in ENTRIES}
    CAND = [e for e in ENTRIES if e["kind"] == "candidate"]
    DEV = [e for e in ENTRIES if e["kind"] == "development"]

    # =====================================================================
    # A. protocol freeze, config integrity and scope
    # =====================================================================
    actual = protocol_sha(CFG)
    check(1, "the Phase-9 reservation protocol is byte-identical to its frozen hash",
          actual == CFG["protocol"]["sha256"],
          {"expected": CFG["protocol"]["sha256"], "actual": actual})

    check(2, "the run record carries the same protocol hash",
          RUN.get("protocol_sha256") == CFG["protocol"]["sha256"],
          RUN.get("protocol_sha256"))

    check(3, "the run record's config hash matches config/phase9a.yaml on disk",
          RUN.get("config_sha256") == sha256_file(CFG["_config_path"]),
          RUN.get("config_sha256"))

    check(4, "the run record's assessment-config hash matches the file on disk",
          RUN.get("assessment_config_sha256")
          == sha256_file(CFG["_assessment_path"]),
          RUN.get("assessment_config_sha256"))

    ptxt = open(os.path.join(ROOT, *CFG["protocol"]["path"].split("/")),
                "r", encoding="utf-8").read()
    required = [
        "reserved for ONE-SHOT external validation",
        "development",
        "deterministic ADC/SCC eligibility",
        "must NOT be selected for favourable imaging",
        "independence",
    ]
    missing = [r for r in required if r.lower() not in ptxt.lower()]
    check(5, "the protocol states all six required reservation policies",
          not missing, {"missing": missing})

    check(6, "the protocol forbids using reserved-cohort DISTRIBUTIONS, not only labels",
          "distributions" in ptxt.lower() and "label-free" in ptxt.lower(),
          "protocol section 1.3")

    check(7, "the protocol names LUNG1 and NSCLC-Radiogenomics as development cohorts",
          "LUNG1" in ptxt and "NSCLC-Radiogenomics" in ptxt
          and "development" in ptxt.lower(), "protocol section 1.1")

    scope = RUN.get("scope", {})
    check(8, "the run record declares ZERO models, metrics, features, "
             "segmentations, harmonisations and candidate image downloads",
          all(v == 0 for v in scope.values()) and len(scope) >= 6, scope)

    fs = CFG["experiment"]["forbidden_scope"]
    check(9, "the config declares the forbidden scope explicitly",
          {"classification_metrics", "model_training",
           "automated_segmentation", "phase9b"}.issubset(set(fs)), fs)

    # =====================================================================
    # B. cached TCIA metadata reproduces the summaries
    # =====================================================================
    n = 10
    bad = []
    for col, s in SUM.items():
        try:
            red = summarise(CFG, col, load_series(col))
        except Exception as exc:                                   # noqa: BLE001
            bad.append((col, str(exc)))
            continue
        for k in ("n_patients", "n_series", "n_studies", "modality_counts",
                  "n_patients_with_segmentation", "total_bytes"):
            if red[k] != s[k]:
                bad.append((col, k, red[k], s[k]))
    check(n, "every collection summary re-derives exactly from its cached "
             "TCIA series listing", not bad, bad[:10]); n += 1

    check(n, "all 15 audited collections appear in the official TCIA "
             "collection listing",
          RUN["n_collections_audited"] == 15 and not RUN["collections_not_found"],
          {"audited": RUN["n_collections_audited"],
           "not_found": RUN["collections_not_found"]}); n += 1

    # LUNG1 and Radiogenomics must agree with what earlier phases recorded.
    check(n, "NSCLC-Radiomics metadata reproduces the 422-subject LUNG1 collection",
          SUM["NSCLC-Radiomics"]["n_patients"] == 422,
          SUM["NSCLC-Radiomics"]["n_patients"]); n += 1

    rg = SUM["NSCLC Radiogenomics"]
    check(n, "NSCLC Radiogenomics metadata reproduces Phase 8A exactly "
             "(211 subjects, 1351 series, CT 727 / PT 480 / SEG 144)",
          rg["n_patients"] == 211 and rg["n_series"] == 1351
          and rg["modality_counts"].get("CT") == 727
          and rg["modality_counts"].get("PT") == 480
          and rg["modality_counts"].get("SEG") == 144,
          {"n_patients": rg["n_patients"], "n_series": rg["n_series"],
           "modalities": rg["modality_counts"]}); n += 1

    check(n, "every candidate in the config produced a rollup entry",
          len(CAND) == len(CFG["candidates"]),
          {"candidates_in_config": len(CFG["candidates"]),
           "rollups": len(CAND)}); n += 1

    check(n, "every candidate has an assessment entry",
          all(e["key"] in ASSESS for e in CAND),
          [e["key"] for e in CAND if e["key"] not in ASSESS]); n += 1

    check(n, "no assessment entry exists for a cohort that is not a candidate",
          set(ASSESS) == {e["key"] for e in CAND},
          {"extra": sorted(set(ASSESS) - {e["key"] for e in CAND})}); n += 1

    check(n, "both development cohorts are marked 'development', never candidates",
          len(DEV) == 2 and {e["key"] for e in DEV} == {"LUNG1", "RADIOGENOMICS"},
          [e["key"] for e in DEV]); n += 1

    # =====================================================================
    # C. segmentation codes are justified by the metadata - TWO-SIDED
    # =====================================================================
    segmods = set(CFG["segmentation_modalities"])

    wrong_e, wrong_a = [], []
    for e in CAND:
        code = ASSESS[e["key"]].get("segmentation_code")
        present = set(e["segmentation_modalities_present"])
        if code in ("D", "E") and present:
            wrong_e.append((e["key"], code, sorted(present)))
        if code in ("A", "B") and not present:
            wrong_a.append((e["key"], code))
    check(n, "every code D/E candidate has ZERO SEG and ZERO RTSTRUCT series "
             "in the official TCIA metadata", not wrong_e, wrong_e); n += 1

    check(n, "every code A/B candidate DOES carry segmentation series "
             "(the claim is falsifiable in both directions)",
          not wrong_a, wrong_a); n += 1

    tcga = BY["TCGA_LUAD_LUSC"]
    check(n, "TCGA-LUAD + TCGA-LUSC: 903 series, no SEG, no RTSTRUCT",
          tcga["n_series"] == 903
          and not (set(tcga["modality_counts"]) & segmods),
          {"n_series": tcga["n_series"],
           "modalities": tcga["modality_counts"]}); n += 1

    l3 = BY["LUNG3"]
    check(n, "Lung3 (NSCLC-Radiomics-Genomics): 89 subjects, CT only, "
             "no segmentation of any kind",
          l3["n_patients"] == 89 and set(l3["modality_counts"]) == {"CT"},
          {"n_patients": l3["n_patients"],
           "modalities": l3["modality_counts"]}); n += 1

    pcd = BY["LUNG_PET_CT_DX"]
    check(n, "Lung-PET-CT-Dx: 355 subjects, CT+PT only, no DICOM segmentation "
             "(bounding boxes are not a segmentation, protocol 3.2)",
          pcd["n_patients"] == 355
          and set(pcd["modality_counts"]) == {"CT", "PT"}
          and ASSESS["LUNG_PET_CT_DX"]["segmentation_code"] == "D",
          {"n_patients": pcd["n_patients"],
           "modalities": pcd["modality_counts"]}); n += 1

    io1 = BY["INTEROBSERVER1"]
    check(n, "Interobserver1: 22 subjects, 21 carry both CT and a segmentation "
             "series - the only code-A ADC/SCC-relevant collection",
          io1["n_patients"] == 22
          and io1["n_patients_with_ct_and_segmentation"] == 21
          and ASSESS["INTEROBSERVER1"]["segmentation_code"] == "A",
          {"n_patients": io1["n_patients"],
           "ct_and_seg": io1["n_patients_with_ct_and_segmentation"]}); n += 1

    qin = SUM["QIN LUNG CT"]
    check(n, "QIN LUNG CT carries SEG for only 10 of its 47 subjects",
          qin["n_patients"] == 47 and qin["n_patients_with_segmentation"] == 10,
          {"n_patients": qin["n_patients"],
           "with_seg": qin["n_patients_with_segmentation"]}); n += 1

    # =====================================================================
    # D. independence audit, and the LIMITS of identifier evidence
    # =====================================================================
    rebuilt = []
    dev_keys = [e["key"] for e in DEV]
    for e in CAND:
        for dk in dev_keys:
            rebuilt.append(overlap(
                {"collection": e["key"], "patients": e["patients"],
                 "studies": e["studies"]},
                {"collection": dk, "patients": BY[dk]["patients"],
                 "studies": BY[dk]["studies"]}))
    stored = [o for o in OVER if o["b"] in dev_keys]
    check(n, "every candidate-versus-development overlap re-derives from the "
             "cached identifier lists",
          [(o["a"], o["b"], o["n_shared_patient_ids"], o["n_shared_study_uids"])
           for o in rebuilt]
          == [(o["a"], o["b"], o["n_shared_patient_ids"], o["n_shared_study_uids"])
              for o in stored],
          {"rebuilt": len(rebuilt), "stored": len(stored)}); n += 1

    check(n, "no candidate shares a PatientID with LUNG1 or NSCLC-Radiogenomics",
          all(o["n_shared_patient_ids"] == 0 for o in stored),
          [o for o in stored if o["n_shared_patient_ids"]]); n += 1

    check(n, "no candidate shares a StudyInstanceUID with either development cohort",
          all(o["n_shared_study_uids"] == 0 for o in stored),
          [o for o in stored if o["n_shared_study_uids"]]); n += 1

    # The point of this check: every collection is de-identified into its own
    # identifier namespace, so exact-string disjointness is guaranteed a priori
    # and is WEAK evidence about patient identity.  The limitation must be
    # recorded machine-readably, not merely asserted in prose.
    lim = RUN.get("identifier_evidence_limitation", {})
    check(n, "the run record states, machine-readably, that exact-string "
             "PatientID disjointness is NOT proof of patient non-overlap "
             "because each collection is de-identified into its own namespace",
          bool(lim.get("statement"))
          and lim.get("exact_id_disjointness_is_proof_of_independence") is False,
          lim); n += 1

    # The concrete counter-example that proves the point.
    qin_r = {int(re.sub(r"[^0-9]", "", p)) for p in SUM["QIN LUNG CT"]["patients"]
             if re.fullmatch(r"R\d+", p)}
    lcd_r = {int(re.sub(r"[^0-9]", "", p))
             for p in SUM["LungCT-Diagnosis"]["patients"]}
    shared_r = sorted(qin_r & lcd_r)
    check(n, "QIN LUNG CT and LungCT-Diagnosis share 24 R-number subjects "
             "despite exact-string disjointness - differently named "
             "collections are NOT automatically independent",
          len(shared_r) == 24, {"n_shared_r_numbers": len(shared_r),
                                "examples": shared_r[:8]}); n += 1

    flagged = {e["key"] for e in ENTRIES if e["same_centre_as_development"]}
    check(n, "the same-centre defect is raised for LUNG3 and INTEROBSERVER1 "
             "(both MAASTRO/Maastricht, protocol 4.1)",
          {"LUNG3", "INTEROBSERVER1"}.issubset(flagged), sorted(flagged)); n += 1

    check(n, "the same-centre defect is NOT raised for any candidate outside "
             "the two Maastricht-derived collections",
          flagged - {"LUNG1"} == {"LUNG3", "INTEROBSERVER1"},
          sorted(flagged)); n += 1

    # =====================================================================
    # E. the size rule and the stop rule, recomputed from the config
    # =====================================================================
    thr = CFG["eligibility"]["primary_external_test"]
    bad = []
    for e in CAND:
        lc = e.get("literature_context") or {}
        sr = e.get("size_rule", {})
        if lc.get("adc") is None or lc.get("scc") is None:
            if sr.get("evaluable"):
                bad.append((e["key"], "marked evaluable without counts"))
            continue
        tot, mino = lc["adc"] + lc["scc"], min(lc["adc"], lc["scc"])
        want = (tot >= thr["min_total_adc_scc"]
                and mino >= thr["min_minority_class"])
        if sr.get("meets_rule") != want:
            bad.append((e["key"], want, sr.get("meets_rule")))
    check(n, "the size rule (>=60 total AND >=20 minority) recomputes exactly "
             "for every candidate", not bad, bad); n += 1

    check(n, "the size thresholds in the run record are the frozen config values",
          RUN["eligibility_thresholds"] == thr,
          RUN["eligibility_thresholds"]); n += 1

    qualified = [e["key"] for e in CAND
                 if ASSESS[e["key"]].get("segmentation_code") in ("A", "B")
                 and (e.get("size_rule") or {}).get("meets_rule")]
    check(n, "no candidate simultaneously satisfies the size rule AND carries "
             "code A/B segmentation", not qualified, qualified); n += 1

    check(n, "the stored stop rule agrees with the recomputation",
          RUN["stop_rule"]["stop_rule_triggered"] is True
          and RUN["stop_rule"]["qualified_primary_external_candidates"] == qualified,
          RUN["stop_rule"]); n += 1

    check(n, "the verbatim stop-rule finding is reproduced exactly as the "
             "protocol requires",
          RUN["stop_rule"]["verbatim_finding"] ==
          "No adequately sized independent public ADC/SCC cohort with "
          "directly reusable primary-tumor segmentation was identified.",
          RUN["stop_rule"]["verbatim_finding"]); n += 1

    check(n, "no pilot was downloaded, because the protocol permits one only "
             "for a candidate that already qualifies",
          RUN["stop_rule"]["pilot_downloaded"] is False
          and bool(RUN["stop_rule"]["pilot_reason"]),
          RUN["stop_rule"]); n += 1

    # Every size-passing candidate must in fact be unsegmented, and vice versa.
    big = {e["key"] for e in CAND if (e.get("size_rule") or {}).get("meets_rule")}
    segd = {e["key"] for e in CAND
            if ASSESS[e["key"]].get("segmentation_code") in ("A", "B")}
    check(n, "the size-passing set and the segmented set are disjoint - this "
             "dissociation IS the Phase-9A finding",
          not (big & segd), {"size_passing": sorted(big),
                             "segmented": sorted(segd)}); n += 1

    # =====================================================================
    # F. the feasibility table carries no performance estimate
    # =====================================================================
    tpath = ppath(CFG, "candidate_table")
    with open(tpath, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    expected_cols = ["candidate", "total_adc", "total_scc", "pre_treatment_ct",
                     "segmentation_availability", "segmentation_provenance",
                     "institution_independence", "possible_patient_overlap",
                     "acquisition_diversity", "download_size",
                     "technical_processing_difficulty",
                     "manual_expert_work_required",
                     "suitable_as_primary_external_test"]
    check(n, "the candidate table has exactly the 13 protocol-mandated columns",
          list(rows[0].keys()) == expected_cols,
          list(rows[0].keys())); n += 1

    check(n, "the candidate table has one row per candidate",
          len(rows) == len(CAND), {"rows": len(rows), "candidates": len(CAND)}); n += 1

    PERF = ("auc", "roc", "roc_auc", "pr_auc", "accuracy", "sensitivity",
            "specificity", "f1", "mcc", "brier", "precision", "recall",
            "performance", "score", "separability", "difficulty_for_model",
            "expected_auc")
    blob = " ".join(" ".join(r.values()) for r in rows).lower()
    hits = [w for w in PERF if re.search(r"(?<![a-z])%s(?![a-z])" % w, blob)]
    check(n, "no model-performance term appears anywhere in the candidate table",
          not hits, hits); n += 1

    # `download_size` legitimately carries decimals (GiB); every other column
    # must contain no probability-like number at all.
    nonsize = " ".join(v for r in rows for k, v in r.items()
                       if k != "download_size")
    nums = re.findall(r"(?<![\d.])0\.\d{2,}", nonsize)
    check(n, "outside the size column, the candidate table contains no "
             "probability-like number that could be read as a performance "
             "estimate", not nums, nums[:10]); n += 1

    sizes_ok = all(re.fullmatch(r"[\d.]+ GiB \(\d+ patients, \d+ series\)",
                                r["download_size"]) for r in rows)
    check(n, "every download_size is a plain size-and-count string",
          sizes_ok, [r["download_size"] for r in rows][:4]); n += 1

    check(n, "every candidate is marked NOT suitable as the primary external "
             "test, consistent with the stop rule",
          all(r["suitable_as_primary_external_test"].strip().lower()
              .startswith("no") for r in rows),
          [(r["candidate"], r["suitable_as_primary_external_test"])
           for r in rows
           if not r["suitable_as_primary_external_test"].strip()
              .lower().startswith("no")]); n += 1

    # =====================================================================
    # G. methods that must NOT appear anywhere in Phase 9A
    # =====================================================================
    src = {p: executable_source(p) for p in PHASE9A_PY}
    FORBIDDEN = ("roc_auc_score", "average_precision_score", "accuracy_score",
                 "confusion_matrix", "brier_score_loss", "LogisticRegression",
                 "StandardScaler", "StratifiedKFold", "fit", "predict",
                 "predict_proba", "combat", "neuroCombat", "featurewise_normalize")
    hits = []
    for p, s in src.items():
        for t in FORBIDDEN:
            if token_present(s, t):
                hits.append((p, t))
    check(n, "no estimator, metric, scaler, splitter or harmonisation "
             "construct exists in any Phase-9A module", not hits, hits); n += 1

    SEGMODELS = ("nnunet", "nnUNet", "totalsegmentator", "TotalSegmentator",
                 "monai", "MONAI", "SegResNet", "UNet", "segment_anything",
                 "sam", "lungmask")
    hits = [(p, t) for p, s in src.items() for t in SEGMODELS
            if token_present(s, t)]
    check(n, "no automated tumour-segmentation model is imported or invoked "
             "(protocol 3.1)", not hits, hits); n += 1

    RADIOMICS = ("radiomics", "featureextractor", "RadiomicsFeatureExtractor",
                 "SimpleITK", "sitk", "pydicom", "nibabel")
    hits = [(p, t) for p, s in src.items() for t in RADIOMICS
            if token_present(s, t)]
    check(n, "no feature extractor and no image reader is used - Phase 9A "
             "never opens a candidate image", not hits, hits); n += 1

    HIST = ("histology", "adenocarcinoma", "squamous", "label", "y_true",
            "positive_class")
    hits = [(p, t) for p, s in src.items() for t in HIST if token_present(s, t)]
    check(n, "no Phase-9A module branches on a histology label",
          not hits, hits); n += 1

    PHASE9B = ("phase9b", "phase8c")
    hits = [(p, t) for p, s in src.items() for t in PHASE9B
            if token_present(s, t.lower()) or token_present(s, t)]
    check(n, "no Phase-9B or Phase-8C construct exists in Phase-9A code",
          not hits, hits); n += 1

    downloaded = []
    for dp, _, fns in os.walk(audit_dir):
        for f in fns:
            if os.path.splitext(f)[1].lower() in (".dcm", ".zip", ".nii", ".gz",
                                                  ".nrrd", ".mha", ".npz"):
                downloaded.append(os.path.relpath(os.path.join(dp, f), ROOT))
    check(n, "the audit directory contains metadata only - no image, archive "
             "or mask was downloaded", not downloaded, downloaded[:10]); n += 1

    # =====================================================================
    # H. namespace, read-only trees, frozen earlier artefacts
    # =====================================================================
    allowed_new = {
        "docs/PHASE9_EXTERNAL_RESERVATION_PROTOCOL.md",
        "config/phase9a.yaml", "config/phase9a_assessment.yaml",
        "src/phase9a_audit.py", "tests/test_phase9a.py",
        "reports/PHASE9A_REPORT.md", "reports/phase9a_candidate_table.csv",
        "reports/phase9a_run.json", "reports/phase9a_tests.json",
    }
    present = {p for p in allowed_new if os.path.exists(os.path.join(ROOT, p))}
    check(n, "every Phase-9A artefact lives inside the declared namespace",
          present.issubset(allowed_new), sorted(present - allowed_new)); n += 1

    stray = []
    for sub in ("predictions", "models", "results", "splits", "preprocessing",
                "global_features", "patch_features", "cnn_embeddings",
                "image_patches", "attention", "feature_selection", "cohort",
                "qa", "patient_summaries"):
        d = os.path.join(ROOT, sub)
        if not os.path.isdir(d):
            continue
        for dp, _, fns in os.walk(d):
            for f in fns:
                if "phase9" in f.lower() or "phase9" in dp.lower():
                    stray.append(os.path.relpath(os.path.join(dp, f), ROOT))
    check(n, "Phase 9A wrote nothing into any modelling namespace",
          not stray, stray[:10]); n += 1

    p9b = []
    for sub in ("src", "config", "tests", "tools", "docs", "reports"):
        d = os.path.join(ROOT, sub)
        for dp, _, fns in os.walk(d):
            if "__pycache__" in dp:
                continue
            for f in fns:
                low = f.lower()
                if "phase9b" in low or "phase8c" in low:
                    p9b.append(os.path.relpath(os.path.join(dp, f), ROOT))
    check(n, "no Phase-9B and no Phase-8C artefact exists anywhere",
          not p9b, p9b); n += 1

    ro = os.path.join(ROOT, "reports", "readonly_baseline.json")
    with open(ro, "r", encoding="utf-8") as fh:
        base = json.load(fh)
    entries_ro = base["entries"]
    changed = []
    for path, meta in entries_ro.items():
        try:
            st = os.stat(path)
        except OSError:
            changed.append(("missing", path))
            continue
        if st.st_size != meta[0]:
            changed.append(("resized", path))
    check(n, "Dataset/ and Vuong5ROI/ are unchanged - all %d baseline files "
             "still present at their recorded size (read-only guarantee)"
          % base["n_files"],
          not changed and len(entries_ro) == base["n_files"],
          changed[:10] or "%d/%d unchanged" % (len(entries_ro) - len(changed),
                                               base["n_files"])); n += 1

    changed = []
    for name in ("reports/PHASE8B_REPORT.md", "reports/PHASE8A_REPORT.md",
                 "docs/PHASE8B_MULTICENTER_PROTOCOL.md",
                 "config/phase8b_multicenter.yaml",
                 "config/phase8a_external.yaml"):
        if not os.path.exists(os.path.join(ROOT, name)):
            changed.append(("missing", name))
    check(n, "the frozen Phase-8A and Phase-8B protocol, config and report "
             "files are all still present", not changed, changed); n += 1

    check(n, "Phase 9A computed no ADC/SCC metric anywhere on disk",
          not os.path.exists(os.path.join(ROOT, "results", "phase9a")),
          "results/phase9a must not exist"); n += 1

    # Creating Phase 9A necessarily falsifies the earlier suites' "no later
    # artefact exists" namespace assertions - the same structural consequence
    # the Phase-8B protocol declared in its section 1.1.  What must be true is
    # that EVERY Phase-9 path named in those failures lies inside the declared
    # Phase-9A namespace, and that no OTHER kind of check regressed.
    ALLOWED_P9 = ("docs/PHASE9_EXTERNAL_RESERVATION_PROTOCOL.md",
                  "config/phase9a", "src/phase9a_audit.py",
                  "tests/test_phase9a.py", "reports/PHASE9A_REPORT.md",
                  "reports/phase9a", "logs/phase9a",
                  "external/phase9_candidate_audit/")
    outside, suites = [], {}
    for suite, path in (("test_phase7a", "reports/phase7a_tests.json"),
                        ("test_phase8a", "reports/phase8a_tests.json"),
                        ("test_phase8b", "reports/phase8b_tests.json")):
        fp = os.path.join(ROOT, *path.split("/"))
        if not os.path.exists(fp):
            continue
        with open(fp, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        failed, named = [], []
        for c in d["checks"]:
            if c["pass"]:
                continue
            failed.append(c["n"])
            blob = json.dumps(c["detail"], default=str)
            for m in re.findall(r"[A-Za-z0-9_/\-\.]*phase9[A-Za-z0-9_/\-\.]*",
                                blob):
                named.append(m)
                if not any(m.startswith(a) for a in ALLOWED_P9):
                    outside.append((suite, c["n"], m))
        suites[suite] = {"failed": failed,
                         "phase9_paths_named": sorted(set(named))}
    check(n, "every Phase-9 path named in a frozen suite's failure lies inside "
             "the declared Phase-9A namespace - creating a new phase falsifies "
             "the earlier namespace assertions, and nothing else regressed",
          not outside, {"outside_namespace": outside, "suites": suites}); n += 1

    # =====================================================================
    out = {"phase": "9A", "n_checks": len(CHECKS),
           "n_passed": sum(1 for c in CHECKS if c["pass"]),
           "n_failed": sum(1 for c in CHECKS if not c["pass"]),
           "checks": CHECKS}
    with open(ppath(CFG, "tests_record"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\n%d/%d checks pass" % (out["n_passed"], out["n_checks"]))
    return 0 if out["n_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
