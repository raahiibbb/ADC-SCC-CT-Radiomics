"""Phase-8B audit: source-aware multicenter global GTV/rim radiomics.

Run with:
    ./.venv/Scripts/python.exe tests/test_phase8b.py

Phase 8B pools LUNG1 (202) and NSCLC-Radiogenomics (141) into 343 patients and
asks whether an intratumoral / peritumoral radiomic signal exists inside BOTH
cohorts and transports between them.  Because the class prevalence is reversed
between the two collections, the single largest threat is that a model scores
well by recognising the SCANNER rather than the TUMOUR.  Most of what follows is
therefore a leakage audit, not a performance check.

Every check re-derives its answer from files on disk.  Where a value could in
principle have been produced by looking at held-out data, the check refits the
step from the training rows alone and demands the stored artefact back, then
demands that including the held-out rows would have produced something DIFFERENT
(a reproduction test alone cannot distinguish "fitted on train" from "fitted on
everything" when the two happen to coincide).

Groups
  A  protocol freeze, scope and the retired external status           (1-10)
  B  cohort, globally unique keys, duplicates                        (11-18)
  C  cached features, no PyRadiomics rerun, compatibility            (19-25)
  D  the frozen source-stratified outer split and inner folds        (26-33)
  E  four-stratum source-balanced training weights                   (34-38)
  F  training-only preprocessing, selection and hyperparameters      (39-50)
  G  A1/A2 parity, capacity, and the A3 inner-OOF late fusion        (51-58)
  H  predictions and the three required evaluation views             (59-68)
  I  the two diagnostics                                             (69-73)
  J  methods that must NOT appear anywhere in Phase 8B               (74-79)
  K  bootstrap                                                       (80-82)
  L  frozen artefacts, read-only trees, namespace, Gate 8B           (83-88)
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from phase8b_data import (build_cohort, inner_split, load_cfg, load_region,   # noqa: E402
                          ppath, protocol_sha, sha256, source_weights)
from phase8b_models import (auc, candidates, choose_candidate, fit_candidate,  # noqa: E402
                            fit_stacker, make_preprocessor)
from phase8b_run import MODELS, PRED_FILE, _mdir                             # noqa: E402
from phase7a_models import PlattCalibrator, select_threshold                 # noqa: E402

CFG = load_cfg(os.path.join(ROOT, "config", "phase8b_multicenter.yaml"))
CHECKS = []

# Phase-8B source whose CONTENT the scope scans inspect.  This audit file is
# deliberately excluded: a scanner carrying its own search lists self-hits.
PHASE8B_PY = ("src/phase8b_data.py", "src/phase8b_models.py", "src/phase8b_run.py",
              "src/phase8b_evaluate.py", "src/phase8b_status.py",
              "tools/phase8b_baseline.py")
MODELLING_PY = ("src/phase8b_data.py", "src/phase8b_models.py", "src/phase8b_run.py")

COHORTS = ("LUNG1", "RADIOGENOMICS")
DIRECTIONS = ("lung1_to_radiogenomics", "radiogenomics_to_lung1")
TOL = 1e-9


def check(n, name, ok, detail):
    CHECKS.append({"n": n, "check": name, "pass": bool(ok), "detail": detail})
    print("%-4s %2d  %s" % ("PASS" if ok else "FAIL", n, name))
    if not ok:
        print("        %s" % (detail if isinstance(detail, str)
                              else json.dumps(detail, default=str)[:1200]))


def token_present(src, token):
    """Whole-identifier match, never substring - the Phase-6B fix, kept."""
    return re.search(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(token), src) is not None


def executable_source(path):
    """Source with docstrings stripped, so prose cannot trip a scan."""
    src = open(os.path.join(ROOT, path), "r", encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(tree))


def function_source(path, name):
    """The executable body of one function, docstrings and comments removed."""
    tree = ast.parse(open(os.path.join(ROOT, path), "r", encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
            return ast.unparse(ast.fix_missing_locations(node))
    raise KeyError("%s not found in %s" % (name, path))


def _no_strings(src):
    """Source with every string constant blanked - a dict key or a path is not
    a call, and must not trip an identifier scan."""
    return re.sub(r"(['\"])(?:(?!\1).)*\1", "''", src)


def rd(*parts):
    return os.path.join(ppath(CFG, "results_dir"), *parts)


def jload(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def close(a, b, tol=1e-8):
    return bool(np.all(np.abs(np.asarray(a, float) - np.asarray(b, float)) <= tol))


# ---------------------------------------------------------------------------
# shared state, re-derived from disk
# ---------------------------------------------------------------------------
RUN = jload(ppath(CFG, "run_json"))
FROZEN = RUN["frozen_before_modelling"]
COHORT = build_cohort(CFG)
MATS = {r: load_region(CFG, r, COHORT) for r in CFG["features"]["regions"]}
SPLIT = pd.read_csv(ppath(CFG, "split_csv"))
STRAT = dict(zip(COHORT.PatientKey, COHORT.cohort_histology))
Y = dict(zip(COHORT.PatientKey, COHORT.label))
COH = dict(zip(COHORT.PatientKey, COHORT.Cohort))
FOLDS = sorted(SPLIT.fold.unique())


def fold_keys(k):
    f = SPLIT[SPLIT.fold == k]
    return sorted(f[f.split == "train"].PatientKey), sorted(f[f.split == "test"].PatientKey)


def preds(tag, m):
    return pd.read_csv(os.path.join(ppath(CFG, "predictions_dir"), PRED_FILE[tag][m]))


def model_json(m, stem):
    return jload(os.path.join(ppath(CFG, "models_dir"), _mdir(m), "%s.json" % stem))


def surface_csv(stem):
    df = pd.read_csv(rd("hyperparameter_surfaces", "%s.csv" % stem))
    out = []
    for _, r in df.iterrows():
        out.append({"candidate_id": r["candidate_id"], "family": r["family"],
                    "l1_ratio": None if pd.isna(r["l1_ratio"]) else float(r["l1_ratio"]),
                    "C": float(r["C"]), "k": None if pd.isna(r["k"]) else int(r["k"]),
                    "mean_inner_roc_auc": float(r["mean_inner_roc_auc"]),
                    "mean_n_selected": float(r["mean_n_selected"]),
                    "config_order": int(r["config_order"])})
    return out


# Refits are the expensive part of this audit, so each (unit, region) is fitted
# once and reused by every check that needs it.
_REFIT = {}


def refit(unit, m):
    """Refit the CHOSEN candidate on the TRAINING rows of `unit` alone."""
    key = (unit, m)
    if key in _REFIT:
        return _REFIT[key]
    region = {"A1": "gtv", "A2": "rim"}[m]
    mat = MATS[region]
    if unit.startswith("pooled_fold_"):
        k = int(unit.rsplit("_", 1)[1])
        tr_keys, _ = fold_keys(k)
        rec = model_json(m, "pooled_fold_%d" % k)
        w, _r = source_weights(CFG, np.asarray([STRAT[p] for p in tr_keys]))
        cw = None
    else:
        d = next(x for x in CFG["validation"]["transfer"]["directions"] if x["id"] == unit)
        tr_keys = list(COHORT[COHORT.Cohort == d["train"]].PatientKey)
        rec = model_json(m, unit)
        w, cw = None, CFG["validation"]["transfer"]["class_balancing"]
    tr = np.asarray([mat["index"][p] for p in tr_keys])
    y_tr = np.asarray([Y[p] for p in tr_keys], dtype=int)
    cand = next(c for c in candidates(CFG) if c["id"] == rec["candidate_id"])
    pre, est = fit_candidate(CFG, mat["x"][tr], y_tr, w, cand, class_weight=cw)
    kept = pre.kept_indices
    out = {"pre": pre, "est": est, "rec": rec, "tr": tr, "tr_keys": tr_keys, "y_tr": y_tr,
           "names": [mat["names"][int(kept[j])] for j in est.selected], "mat": mat}
    _REFIT[key] = out
    return out


# =========================================================================
# A  protocol freeze, scope, retired external status
# =========================================================================

def a1_protocol_frozen_and_hashed():
    doc = os.path.join(ROOT, CFG["experiment"]["protocol_doc"])
    ok = os.path.isfile(doc) and sha256(doc) == FROZEN["protocol_sha256"]
    check(1, "the protocol document exists and its SHA-256 is the one recorded in "
             "reports/phase8b_run.json", ok,
          {"doc": CFG["experiment"]["protocol_doc"],
           "on_disk": sha256(doc)[:32] if os.path.isfile(doc) else None,
           "recorded": FROZEN["protocol_sha256"][:32]})


def a2_config_frozen():
    check(2, "the config SHA-256 is unchanged since the freeze",
          sha256(CFG["_config_path"]) == FROZEN["config_sha256"],
          {"on_disk": sha256(CFG["_config_path"])[:32],
           "recorded": FROZEN["config_sha256"][:32]})


def a3_frozen_before_any_model_ran():
    """The freeze must precede every modelling artefact on the clock.

    The run record's own mtime cannot be the reference: the driver appends its
    modelling block to that same file when it finishes.  The authority is the
    `frozen_at` stamp the freeze step wrote into it.
    """
    t_freeze = time.mktime(time.strptime(RUN["frozen_at"], "%Y-%m-%dT%H:%M:%S"))
    later = []
    for p in (ppath(CFG, "cohort_csv"), ppath(CFG, "split_csv"),
              ppath(CFG, "split_meta_json"), ppath(CFG, "cross_cohort_json")):
        # the freeze record is written LAST by the freeze step
        if os.path.isfile(p) and os.path.getmtime(p) > t_freeze + 5:
            later.append(os.path.relpath(p, ROOT))
    produced = []
    for tag in ("pooled",) + DIRECTIONS:
        for m in MODELS:
            p = os.path.join(ppath(CFG, "predictions_dir"), PRED_FILE[tag][m])
            if os.path.isfile(p):
                produced.append(os.path.getmtime(p))
    ok = not later and (not produced or min(produced) > t_freeze)
    check(3, "cohort, split and protocol were frozen BEFORE any label reached a "
             "Phase-8B model", ok,
          {"artefacts_modified_after_the_freeze": later,
           "n_prediction_files": len(produced),
           "earliest_prediction_after_freeze": bool(produced and min(produced) > t_freeze)})


def a4_external_status_retired():
    txt = open(os.path.join(ROOT, CFG["experiment"]["protocol_doc"]),
               encoding="utf-8").read().lower()
    need = {"phase 8a prepared it as an untouched external cohort":
                "untouched" in txt and "phase 8a" in txt,
            "phase 8b retires that status": "retire" in txt,
            "no phase-8b result is independent external validation":
                "not" in txt and "external validation" in txt,
            "a third cohort is required": "third" in txt and "cohort" in txt,
            "phases 1-8a remain frozen": "frozen" in txt}
    st = CFG["experiment"]["external_status"]
    ok = (all(need.values())
          and st["independent_external_validation_claimable"] is False
          and st["third_cohort_still_required"] is True
          and FROZEN["external_status"]["now"] == "multicenter_development_data")
    check(4, "the protocol records that Phase 8B RETIRES the untouched-external "
             "status and that a third cohort is still required", ok,
          {"protocol_statements": need, "config_status": st})


def a5_no_external_validation_claim():
    """Every occurrence of a forbidden claim phrase must sit in a negating or
    declarative context.

    The scan is over the Phase-8B documents only.  docs/PROJECT_STATUS.md is
    deliberately excluded: its roadmap predates Phase 8B and legitimately names
    a FUTURE external validation, which is not a claim about a Phase-8B number.
    Context is the sentence, not the line, because a negation frequently sits on
    the preceding line ("the result **must not** be called an / externally
    validated breakthrough").
    """
    bad = []
    neg = re.compile(r"\b(not|never|no|nor|cannot|forbidden|prohibit\w*|retire[sd]?|"
                     r"rather than|instead of|would be|may use|strongest phrase|"
                     r"claimable|false)\b", re.I)
    for rel in (CFG["experiment"]["protocol_doc"], "reports/PHASE8B_REPORT.md"):
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            continue
        lines = open(p, encoding="utf-8").read().splitlines()
        for i, line in enumerate(lines):
            ctx = " ".join(lines[max(0, i - 2):i + 2])
            for phrase in CFG["experiment"]["external_status"]["forbidden_claim_phrases"]:
                if phrase.lower() in line.lower() and not neg.search(ctx):
                    bad.append("%s:%d %s" % (rel, i + 1, line.strip()[:140]))
    check(5, "no Phase-8B document claims external validation in a non-negated "
             "sentence", not bad, {"unnegated_claims": bad[:12]})


def a6_scope_declared():
    sc = set(CFG["experiment"]["authorised_scope"])
    fb = set(CFG["experiment"]["forbidden_scope"])
    ok = (sc == {"A1", "A2", "A3", "domain_diagnostic", "prediction_shift_diagnostic"}
          and {"CNN", "CLAM", "Transformer", "ComBat", "SMOTE", "RandomForest",
               "XGBoost", "SVM", "phase8c"} <= fb)
    check(6, "the authorised scope is exactly A1/A2/A3 plus the two diagnostics, "
             "and the forbidden scope names every excluded method", ok,
          {"authorised": sorted(sc), "missing_from_forbidden":
              sorted({"CNN", "CLAM", "Transformer", "ComBat", "SMOTE", "RandomForest",
                      "XGBoost", "SVM", "phase8c"} - fb)})


def a7_primary_metric_is_raw_and_frozen():
    ok = (FROZEN["primary_probability"] == "raw"
          and CFG["evaluation"]["primary_probability"] == "raw"
          and CFG["gate_8b"]["probability"] == "raw")
    check(7, "the RAW-probability primary-metric convention was frozen before "
             "modelling, in the protocol, the config and the gate", ok,
          {"run_record": FROZEN["primary_probability"],
           "config": CFG["evaluation"]["primary_probability"],
           "gate": CFG["gate_8b"]["probability"]})


def a8_phase7a_report_not_altered():
    snap = jload(ppath(CFG, "pre_snapshot"))["files"]
    changed = []
    for rel in ("reports/PHASE7A_REPORT.md", "docs/PHASE7_PROTOCOL.md",
                "config/phase7a.yaml", "reports/phase7a_run.json"):
        p = os.path.join(ROOT, rel)
        if rel in snap and os.path.isfile(p) and sha256(p) != snap[rel]:
            changed.append(rel)
    check(8, "the Phase-7A protocol, config, run record and report are byte-identical "
             "- the historical calibrated convention was not rewritten", not changed,
          {"changed": changed})


def a9_no_scientific_parameter_hard_coded():
    """Grids, tolerances and gate thresholds live in the config, not in src/."""
    literals = ("0.25", "0.75", "0.90", "0.003", "0.65", "0.60", "0.55", "5000",
                "20260961")
    hits = []
    for rel in PHASE8B_PY:
        src = _no_strings(executable_source(rel))
        for lit in literals:
            if token_present(src, lit):
                hits.append("%s: %s" % (rel, lit))
    check(9, "no scientific parameter (grid value, gate threshold, bootstrap size, "
             "seed) is hard-coded in src/ - all come from the config", not hits,
          {"hard_coded": hits})


def a10_no_phase8c_artefact():
    stray = []
    for root, _, fs in os.walk(ROOT):
        if "__pycache__" in root or os.sep + ".venv" in root or os.sep + ".git" in root:
            continue
        for f in fs:
            if "phase8c" in f.lower():
                stray.append(os.path.relpath(os.path.join(root, f), ROOT))
    check(10, "no Phase-8C artefact of any kind exists", not stray, {"files": stray[:20]})


# =========================================================================
# B  cohort, globally unique keys, duplicates
# =========================================================================

def b11_patient_keys_unique_and_formatted():
    keys = list(COHORT.PatientKey)
    fmt_bad = [k for k, c, p in zip(keys, COHORT.Cohort, COHORT.PatientID)
               if k != "%s::%s" % ({"LUNG1": CFG["cohorts"]["lung1"]["key_prefix"],
                                    "RADIOGENOMICS": CFG["cohorts"]["radiogenomics"]["key_prefix"]}[c], p)]
    ok = len(set(keys)) == len(keys) == 343 and not fmt_bad
    check(11, "343 globally unique PatientKeys of the form <COHORT>::<PatientID>", ok,
          {"n": len(keys), "n_unique": len(set(keys)), "malformed": fmt_bad[:5]})


def b12_never_relies_on_raw_patient_id():
    """No Phase-8B modelling code may key, join or index on PatientID."""
    hits = []
    for rel in MODELLING_PY:
        tree = ast.parse(open(os.path.join(ROOT, rel), encoding="utf-8").read())
        for node in ast.walk(tree):
            # only the KEY arguments matter - carrying PatientID along as a
            # payload column is fine, keying on it is not
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr in ("merge", "join", "set_index", "groupby",
                                           "reindex", "drop_duplicates"):
                keys = [ast.unparse(k.value) for k in node.keywords
                        if k.arg in ("on", "left_on", "right_on", "by", "keys", "subset")]
                if node.func.attr in ("set_index", "groupby", "reindex") and node.args:
                    keys.append(ast.unparse(node.args[0]))
                if any("PatientID" in k for k in keys):
                    hits.append("%s: %s" % (rel, ast.unparse(node)[:110]))
    check(12, "PatientID is never used as a join, index or grouping key - only "
              "PatientKey is", not hits, {"uses": hits})


def b13_cohort_counts():
    n = COHORT.Cohort.value_counts().to_dict()
    ok = (n.get("LUNG1") == CFG["cohorts"]["lung1"]["expected_patients"]
          and n.get("RADIOGENOMICS") == CFG["cohorts"]["radiogenomics"]["expected_patients"]
          and len(COHORT) == CFG["cohorts"]["combined"]["expected_patients"])
    check(13, "cohort sizes are 202 LUNG1 + 141 Radiogenomics = 343", ok, n)


def b14_class_counts():
    tot = {"adc": int(COHORT.label.sum()), "scc": int((COHORT.label == 0).sum())}
    per = {c: {"adc": int(COHORT[(COHORT.Cohort == c) & (COHORT.label == 1)].shape[0]),
               "scc": int(COHORT[(COHORT.Cohort == c) & (COHORT.label == 0)].shape[0])}
           for c in COHORTS}
    ok = (tot == {"adc": 163, "scc": 180}
          and per["LUNG1"] == {"adc": 51, "scc": 151}
          and per["RADIOGENOMICS"] == {"adc": 112, "scc": 29})
    check(14, "163 ADC / 180 SCC overall, 51/151 in LUNG1 and 112/29 in "
              "Radiogenomics - the prevalence really is reversed", ok,
          {"combined": tot, "per_cohort": per})


def b15_stratum_counts():
    got = COHORT.cohort_histology.value_counts().to_dict()
    want = {k: int(v) for k, v in CFG["cohorts"]["strata"]["expected_counts"].items()}
    check(15, "the four cohort x histology strata have exactly the frozen counts",
          got == want, {"got": got, "expected": want})


def b16_no_raw_id_collision():
    a = set(COHORT[COHORT.Cohort == "LUNG1"].PatientID)
    b = set(COHORT[COHORT.Cohort == "RADIOGENOMICS"].PatientID)
    check(16, "no raw PatientID occurs in both collections", not (a & b),
          {"collisions": sorted(a & b)[:10]})


def b17_no_duplicate_feature_rows():
    dup = {}
    for region, mat in MATS.items():
        h = [hashlib.sha256(np.ascontiguousarray(r, dtype=np.float64).tobytes()).hexdigest()
             for r in mat["x"]]
        s = pd.Series(h)
        d = s[s.duplicated(keep=False)]
        dup[region] = sorted({mat["keys"][i] for i in d.index})
    check(17, "no two patients share an identical global feature vector in either "
              "region", not any(dup.values()), dup)


def b18_cross_cohort_nearest_neighbour_screen():
    scr = RUN["duplicate_screen"]["regions"]
    ok = True
    detail = {}
    for region in MATS:
        r = scr[region]
        near = float(r["cross_cohort_nn_distance_min"])
        within = float(r["within_cohort_nn_distance_p1"])
        detail[region] = {"cross_min": near, "within_p1": within,
                          "flagged": r["cross_cohort_pairs_below_threshold"]}
        ok = ok and not r["cross_cohort_pairs_below_threshold"] and near > within
    check(18, "the standardised-space nearest-neighbour screen found no cross-cohort "
              "pair closer than the within-cohort 1st percentile", ok, detail)


# =========================================================================
# C  cached features, no PyRadiomics rerun
# =========================================================================

def c19_cached_feature_files_unchanged():
    bad = []
    for region in MATS:
        f = RUN["features"][region]
        for side in ("lung1", "external"):
            p = os.path.join(ROOT, f["%s_csv" % side])
            if sha256(p) != f["%s_sha256" % side]:
                bad.append(f["%s_csv" % side])
    check(19, "every cached global-feature CSV is byte-identical to the file that "
              "was hashed at the freeze", not bad, {"changed": bad})


def c20_no_pyradiomics_rerun():
    hits = []
    for rel in PHASE8B_PY:
        # string constants are stripped: cfg["models"]["radiomics"] is a config
        # key, not an extractor call
        src = _no_strings(executable_source(rel))
        for tok in ("radiomics", "featureextractor", "RadiomicsFeatureExtractor",
                    "SimpleITK", "sitk", "imageoperations"):
            if token_present(src, tok):
                hits.append("%s: %s" % (rel, tok))
    check(20, "no Phase-8B source imports or calls PyRadiomics or SimpleITK - "
              "features were read, never re-extracted",
          not hits and CFG["features"]["rerun_pyradiomics"] is False,
          {"tokens": hits, "config_flag": CFG["features"]["rerun_pyradiomics"]})


def c21_feature_names_identical_and_ordered():
    bad = {}
    for region in MATS:
        a = jload(os.path.join(ROOT, CFG["paths"]["lung1_feature_names_json"]))
        b = jload(os.path.join(ROOT, CFG["paths"]["external_feature_names_json"]))
        na = a[region] if isinstance(a, dict) else a
        nb = b[region] if isinstance(b, dict) else b
        if list(na) != list(nb):
            first = next((i for i, (u, v) in enumerate(zip(na, nb)) if u != v), None)
            bad[region] = {"n_lung1": len(na), "n_external": len(nb),
                           "first_difference_at": first}
    check(21, "feature NAMES and ORDER are identical between the two cohorts in "
              "both regions - verified mechanically, not assumed", not bad, bad)


def c22_extraction_signature():
    sig = CFG["features"]["expected_extraction_signature"]
    got = {r: RUN["features"][r]["extraction_signature"] for r in MATS}
    check(22, "both cohorts carry the frozen Phase-7A extraction signature %s" % sig,
          all(v == sig for v in got.values()), got)


def c23_feature_counts_and_no_rim_shape():
    n = {r: MATS[r]["x"].shape[1] for r in MATS}
    shape_in_rim = [f for f in MATS["rim"]["names"] if f.split("_")[1:2] == ["shape"]
                    or "_shape_" in f]
    ok = (n["gtv"] == CFG["features"]["expected_n_gtv"]
          and n["rim"] == CFG["features"]["expected_n_rim"] and not shape_in_rim)
    check(23, "1130 GTV and 1116 rim features, and the rim bank carries no shape "
              "feature", ok, {"n": n, "rim_shape_features": shape_in_rim[:5]})


def c24_no_nan_or_inf():
    bad = {r: int((~np.isfinite(MATS[r]["x"])).sum()) for r in MATS}
    check(24, "no NaN or Inf in either 343-row feature matrix",
          all(v == 0 for v in bad.values()), bad)


def c25_feature_matrix_aligned_to_cohort():
    bad = {r: [k for k, c in zip(MATS[r]["keys"], COHORT.PatientKey) if k != c][:5]
           for r in MATS}
    ok = all(not v for v in bad.values()) and all(
        list(MATS[r]["y"]) == list(COHORT.label) for r in MATS)
    check(25, "both feature matrices are row-aligned to the frozen cohort order and "
              "carry its labels", ok, bad)


# =========================================================================
# D  the frozen outer split and inner folds
# =========================================================================

def d26_split_reproduces_from_the_frozen_seed():
    o = CFG["validation"]["outer"]
    c = COHORT.sort_values(o["sort_by"]).reset_index(drop=True)
    skf = StratifiedKFold(n_splits=int(o["n_splits"]), shuffle=bool(o["shuffle"]),
                          random_state=int(o["random_state"]))
    rebuilt = {}
    for k, (_, te) in enumerate(skf.split(np.arange(len(c)), c[o["stratify_by"]]), 1):
        for i in te:
            rebuilt[c.PatientKey[i]] = k
    frozen = {r.PatientKey: int(r.fold)
              for _, r in SPLIT[SPLIT.split == "test"].iterrows()}
    check(26, "the outer split reproduces exactly from StratifiedKFold(5, shuffle, "
              "42) on cohort_histology over PatientKey-sorted patients",
          rebuilt == frozen,
          {"n_mismatched": sum(1 for k in frozen if rebuilt.get(k) != frozen[k])})


def d27_split_file_unchanged():
    ok = (sha256(ppath(CFG, "split_csv")) == FROZEN["split_csv_sha256"]
          and sha256(ppath(CFG, "split_meta_json")) == FROZEN["split_meta_sha256"]
          and sha256(ppath(CFG, "cross_cohort_json")) == FROZEN["cross_cohort_sha256"])
    check(27, "the split, its metadata and the cross-cohort definition are unchanged "
              "since the freeze", ok, {})


def d28_every_fold_has_all_four_strata():
    want = set(CFG["cohorts"]["strata"]["levels"])
    missing = {}
    for k in FOLDS:
        tr, te = fold_keys(k)
        for part, keys in (("train", tr), ("test", te)):
            got = {STRAT[p] for p in keys}
            if got != want:
                missing["fold_%d_%s" % (k, part)] = sorted(want - got)
    check(28, "every outer training AND test partition contains all four "
              "cohort x histology strata", not missing, missing)


def d29_split_partitions_the_cohort():
    test_counts = SPLIT[SPLIT.split == "test"].PatientKey.value_counts()
    bad = []
    for k in FOLDS:
        tr, te = fold_keys(k)
        if set(tr) | set(te) != set(COHORT.PatientKey) or set(tr) & set(te):
            bad.append(k)
    ok = (len(test_counts) == 343 and test_counts.max() == 1 and not bad)
    check(29, "each patient is tested exactly once and train is always the exact "
              "complement of test", ok,
          {"n_tested_once": int((test_counts == 1).sum()), "bad_folds": bad})


def d30_stratified_on_cohort_histology_not_label():
    """Stratifying on histology alone would give a DIFFERENT split."""
    o = CFG["validation"]["outer"]
    c = COHORT.sort_values(o["sort_by"]).reset_index(drop=True)
    skf = StratifiedKFold(n_splits=int(o["n_splits"]), shuffle=bool(o["shuffle"]),
                          random_state=int(o["random_state"]))
    label_only = {}
    for k, (_, te) in enumerate(skf.split(np.arange(len(c)), c["label"]), 1):
        for i in te:
            label_only[c.PatientKey[i]] = k
    frozen = {r.PatientKey: int(r.fold)
              for _, r in SPLIT[SPLIT.split == "test"].iterrows()}
    n_diff = sum(1 for k in frozen if label_only.get(k) != frozen[k])
    check(30, "the split really is stratified on cohort x histology - a "
              "histology-only split would assign %d patients differently" % n_diff,
          n_diff > 0 and o["stratify_by"] == "cohort_histology",
          {"n_patients_that_would_move": n_diff, "stratify_by": o["stratify_by"]})


def d31_split_regenerated_not_copied():
    """The old LUNG1 split cannot have produced this one: it contains no
    Radiogenomics patient and covers 202 of the 343 rows."""
    old = os.path.join(ROOT, "splits", "gtv_rim_stratified_5fold.csv")
    covered = 0
    if os.path.isfile(old):
        o = pd.read_csv(old)
        covered = len(set(o.PatientID) & set(COHORT.PatientID))
    ok = (covered < len(COHORT) and len(SPLIT.PatientKey.unique()) == 343
          and RUN["outer_split"]["stratify_by"] == "cohort_histology"
          and RUN["outer_split"]["does_not_reuse_the_lung1_202_split"] is True)
    check(31, "the outer split was regenerated over all 343 patients under the new "
              "four-level stratification - the old 202-patient LUNG1 split could not "
              "have produced it", ok,
          {"old_file_covers_n_of_343": covered,
           "recorded_reason": RUN["outer_split"]["reason"]})


def d32_inner_folds_reproduce_and_preserve_strata():
    want = set(CFG["cohorts"]["strata"]["levels"])
    bad, mism = {}, []
    for k in FOLDS:
        tr, _ = fold_keys(k)
        seed = int(CFG["validation"]["inner"]["base_seed"]) + int(k)
        inner = inner_split(CFG, tr, [STRAT[p] for p in tr], seed)
        if len(inner) != int(CFG["validation"]["inner"]["n_splits"]):
            mism.append(k)
        for j, (a, b) in enumerate(inner, 1):
            for part, idx in (("train", a), ("val", b)):
                got = {STRAT[tr[i]] for i in idx}
                if got != want:
                    bad["fold_%d_inner_%d_%s" % (k, j, part)] = sorted(want - got)
        # the persisted inner assignment must match the regenerated one
        # (outer-test rows carry inner_fold = -1 and are not part of it)
        p = os.path.join(ppath(CFG, "inner_cv_dir"), "fold_%d.csv" % k)
        if os.path.isfile(p):
            s = pd.read_csv(p)
            s = s[s.split == "train"]
            saved = {kk: int(v) for kk, v in zip(s.PatientKey, s.inner_fold)}
            regen = {}
            for j, (_, b) in enumerate(inner):
                for i in b:
                    regen[tr[i]] = j
            if saved != regen:
                mism.append("fold_%d_persisted_mismatch" % k)
    check(32, "the 3 inner folds reproduce from the frozen seed base and every inner "
              "training and validation part carries all four strata",
          not bad and not mism, {"missing_strata": bad, "mismatch": mism})


def d33_transfer_definition_is_the_two_full_cohorts():
    cc = {r["id"]: r for r in jload(ppath(CFG, "cross_cohort_json"))["directions"]}
    bad = []
    for d in CFG["validation"]["transfer"]["directions"]:
        r = cc[d["id"]]
        n_tr = int((COHORT.Cohort == d["train"]).sum())
        n_te = int((COHORT.Cohort == d["test"]).sum())
        if r["n_train"] != n_tr or r["n_test"] != n_te:
            bad.append(d["id"])
    check(33, "each transport direction trains on one entire cohort and tests on the "
              "entire other cohort", not bad, {"bad": bad})


# =========================================================================
# E  four-stratum source-balanced weights
# =========================================================================

def e34_weights_reproduce_from_training_counts():
    bad = {}
    for k in FOLDS:
        tr, _ = fold_keys(k)
        w, rec = source_weights(CFG, np.asarray([STRAT[p] for p in tr]))
        stored = RUN["outer_training_source_weights"][str(k)]["weight_per_patient"]
        for lv, v in rec["weight_per_patient"].items():
            if abs(float(stored[lv]) - float(v)) > TOL:
                bad["fold_%d_%s" % (k, lv)] = [stored[lv], v]
    check(34, "the four-stratum weights reproduce exactly from the TRAINING counts of "
              "each outer fold", not bad, bad)


def e35_equal_total_weight_per_stratum():
    bad = {}
    for k in FOLDS:
        tr, _ = fold_keys(k)
        s = np.asarray([STRAT[p] for p in tr])
        w, _r = source_weights(CFG, s)
        tot = {lv: float(w[s == lv].sum()) for lv in np.unique(s)}
        if max(tot.values()) - min(tot.values()) > float(
                CFG["source_weights"]["equal_total_weight_tolerance"]):
            bad["fold_%d" % k] = tot
    check(35, "every stratum contributes the same TOTAL training weight in every "
              "outer fold", not bad, bad)


def e36_mean_weight_is_one():
    bad = {}
    for k in FOLDS:
        tr, _ = fold_keys(k)
        w, _r = source_weights(CFG, np.asarray([STRAT[p] for p in tr]))
        if abs(float(w.mean()) - 1.0) > float(CFG["source_weights"]["mean_weight_tolerance"]):
            bad["fold_%d" % k] = float(w.mean())
    check(36, "the weights are normalised to mean 1.0 in every outer fold", not bad, bad)


def e37_weights_never_use_outer_test_counts():
    """Weights computed over ALL 343 must differ from the stored fold weights."""
    w_all, rec_all = source_weights(CFG, COHORT.cohort_histology.to_numpy())
    same = []
    for k in FOLDS:
        stored = RUN["outer_training_source_weights"][str(k)]["weight_per_patient"]
        if all(abs(float(stored[lv]) - float(v)) <= TOL
               for lv, v in rec_all["weight_per_patient"].items()):
            same.append(k)
    check(37, "no fold's weights equal the whole-cohort weights - outer-test counts "
              "never entered the weighting", not same,
          {"folds_matching_full_cohort_weights": same,
           "full_cohort_weights": rec_all["weight_per_patient"]})


def e38_weights_not_applied_to_calibration():
    src = function_source("src/phase8b_models.py", "calibrate_and_threshold")
    ok = (CFG["calibration"]["class_weight"] is None
          and CFG["calibration"]["sample_weight"] is None
          and "platt_calibration" in CFG["source_weights"]["never_apply_to"]
          and not token_present(src, "source_weights")
          and not token_present(src, "sample_weight"))
    check(38, "the Platt calibrator is never class-weighted nor stratum-weighted", ok,
          {"config_class_weight": CFG["calibration"]["class_weight"],
           "config_sample_weight": CFG["calibration"]["sample_weight"],
           "never_apply_to": CFG["source_weights"]["never_apply_to"]})


# =========================================================================
# F  training-only preprocessing, selection, hyperparameters
# =========================================================================

def _pre_from(keys, region):
    mat = MATS[region]
    idx = np.asarray([mat["index"][p] for p in keys])
    return make_preprocessor(CFG).fit(mat["x"][idx])


def f39_imputation_fitted_on_training_rows_only():
    bad = []
    for k in FOLDS:
        tr, _ = fold_keys(k)
        for m, region in (("A1", "gtv"), ("A2", "rim")):
            p = os.path.join(ppath(CFG, "preprocessing_dir"), "pooled_%s_fold_%d.npz" % (m, k))
            if not os.path.isfile(p):
                bad.append("missing %s" % os.path.basename(p))
                continue
            saved = np.load(p)
            if not close(_pre_from(tr, region).medians, saved["medians"]):
                bad.append("fold_%d_%s_medians" % (k, m))
    check(39, "the median imputer reproduces from the outer-TRAINING rows alone",
          not bad, {"mismatch": bad})


def f40_variance_filter_fitted_on_training_rows_only():
    bad = []
    for k in FOLDS:
        tr, _ = fold_keys(k)
        for m, region in (("A1", "gtv"), ("A2", "rim")):
            saved = np.load(os.path.join(ppath(CFG, "preprocessing_dir"),
                                         "pooled_%s_fold_%d.npz" % (m, k)))
            if not np.array_equal(_pre_from(tr, region).keep.astype(np.uint8), saved["keep"]):
                bad.append("fold_%d_%s" % (k, m))
    check(40, "the near-zero-variance screen reproduces from the outer-TRAINING rows "
              "alone", not bad, {"mismatch": bad})


def f41_scaler_fitted_on_training_rows_only():
    bad = []
    for k in FOLDS:
        tr, _ = fold_keys(k)
        for m, region in (("A1", "gtv"), ("A2", "rim")):
            saved = np.load(os.path.join(ppath(CFG, "preprocessing_dir"),
                                         "pooled_%s_fold_%d.npz" % (m, k)))
            pre = _pre_from(tr, region)
            if not (close(pre.mean, saved["mean"]) and close(pre.scale, saved["scale"])):
                bad.append("fold_%d_%s" % (k, m))
    check(41, "the standard scaler reproduces from the outer-TRAINING rows alone",
          not bad, {"mismatch": bad})


def f42_including_outer_test_rows_would_have_changed_preprocessing():
    """Reproduction alone is not proof; the leaky alternative must differ."""
    identical = []
    for k in FOLDS:
        for m, region in (("A1", "gtv"), ("A2", "rim")):
            saved = np.load(os.path.join(ppath(CFG, "preprocessing_dir"),
                                         "pooled_%s_fold_%d.npz" % (m, k)))
            leaky = _pre_from(list(COHORT.PatientKey), region)
            if (close(leaky.medians, saved["medians"])
                    and close(leaky.mean, saved["mean"])
                    and close(leaky.scale, saved["scale"])):
                identical.append("fold_%d_%s" % (k, m))
    check(42, "fitting the preprocessor on all 343 patients would have produced a "
              "DIFFERENT transform in every fold - the stored one cannot have seen "
              "the outer test rows", not identical, {"indistinguishable": identical})


def f43_feature_selection_reproduces_from_training_rows_only():
    bad = {}
    for k in FOLDS:
        for m in ("A1", "A2"):
            r = refit("pooled_fold_%d" % k, m)
            if list(r["names"]) != list(r["rec"]["selected_features"]):
                bad["fold_%d_%s" % (k, m)] = {"refit": r["names"],
                                              "stored": r["rec"]["selected_features"]}
    check(43, "refitting the chosen candidate on the outer-TRAINING rows alone "
              "reproduces the stored selected features exactly", not bad, bad)


def f44_coefficients_reproduce_from_training_rows_only():
    bad = {}
    for k in FOLDS:
        for m in ("A1", "A2"):
            r = refit("pooled_fold_%d" % k, m)
            if not (close(r["est"].coefficients, r["rec"]["coefficients"])
                    and abs(float(r["est"].intercept) - float(r["rec"]["intercept"])) <= 1e-8):
                bad["fold_%d_%s" % (k, m)] = {"refit": list(r["est"].coefficients),
                                              "stored": r["rec"]["coefficients"]}
    check(44, "the stored model coefficients reproduce from the outer-TRAINING rows "
              "alone", not bad, bad)


def f45_selector_choice_reproduces_from_the_inner_surface():
    bad = {}
    for k in FOLDS:
        for m in ("A1", "A2"):
            surf = surface_csv("pooled_%s_fold_%d" % (m, k))
            chosen, _t = choose_candidate(CFG, surf)
            stored = model_json(m, "pooled_fold_%d" % k)["candidate_id"]
            if chosen["candidate_id"] != stored:
                bad["fold_%d_%s" % (k, m)] = {"recomputed": chosen["candidate_id"],
                                              "stored": stored}
    check(45, "the selector family and hyperparameters are exactly what the frozen "
              "tie-break returns from the INNER-CV surface", not bad, bad)


def f46_inner_surface_carries_no_outer_test_information():
    """The surface columns must be inner-CV quantities only."""
    allowed = {"candidate_id", "family", "l1_ratio", "C", "k", "mean_inner_roc_auc",
               "sd_inner_roc_auc", "inner_roc_auc", "mean_n_selected", "config_order"}
    bad = {}
    for stem in (["pooled_%s_fold_%d" % (m, k) for m in ("A1", "A2") for k in FOLDS]
                 + ["%s_%s" % (d, m) for d in DIRECTIONS for m in ("A1", "A2")]):
        cols = set(pd.read_csv(rd("hyperparameter_surfaces", "%s.csv" % stem)).columns)
        if cols - allowed:
            bad[stem] = sorted(cols - allowed)
    check(46, "every hyperparameter surface contains inner-CV quantities only - no "
              "outer-test or destination-cohort column exists", not bad, bad)


def f47_the_grid_is_the_frozen_grid():
    g = CFG["models"]["radiomics"]["elastic_net"]["grid"]
    expect = len(g["l1_ratio"]) * len(g["C"]) + len(CFG["models"]["radiomics"]["mrmr"]["k"])
    ids = [c["id"] for c in candidates(CFG)]
    surf_ok, bad = True, {}
    for stem in ["pooled_%s_fold_%d" % (m, k) for m in ("A1", "A2") for k in FOLDS]:
        got = list(pd.read_csv(rd("hyperparameter_surfaces", "%s.csv" % stem)).candidate_id)
        if sorted(got) != sorted(ids):
            surf_ok = False
            bad[stem] = len(got)
    check(47, "exactly %d candidates were evaluated in every fold - 32 elastic-net "
              "plus 2 mRMR, and nothing else" % expect,
          len(ids) == expect and len(set(ids)) == expect and surf_ok,
          {"n_candidates": len(ids), "expected": expect, "surfaces": bad})


def f48_only_two_selector_families():
    fams = sorted({c["family"] for c in candidates(CFG)})
    forbidden = []
    for rel in MODELLING_PY:
        src = executable_source(rel)
        for tok in ("SequentialFeatureSelector", "RFE", "RFECV", "RandomForestClassifier",
                    "XGBClassifier", "SVC", "LinearSVC", "GradientBoostingClassifier",
                    "PCA", "SelectKBest"):
            if token_present(src, tok):
                forbidden.append("%s: %s" % (rel, tok))
    check(48, "the only selector families are elastic-net and mRMR - no SFS, RF, "
              "XGBoost or SVM anywhere", fams == ["elastic_net", "mrmr"] and not forbidden,
          {"families": fams, "forbidden_estimators": forbidden})


def f49_calibration_fitted_on_training_inner_oof_only():
    bad = {}
    for k in FOLDS:
        z = pd.read_csv(os.path.join(ppath(CFG, "models_dir"), "A3_fusion",
                                     "pooled_fold_%d_training_inputs.csv" % k))
        y_tr = z.label.to_numpy(int)
        for m, col in (("A1", "P_ADC_GTV"), ("A2", "P_ADC_RIM")):
            cal = PlattCalibrator().fit(z[col].to_numpy(float), y_tr, CFG["calibration"])
            st = model_json(m, "pooled_fold_%d" % k)["calibration"]
            if abs(cal.a - st["a"]) > 1e-6 or abs(cal.b - st["b"]) > 1e-6 \
                    or st["n_inner_oof"] != len(y_tr):
                bad["fold_%d_%s" % (k, m)] = {"refit": [cal.a, cal.b],
                                              "stored": [st["a"], st["b"]]}
    check(49, "the Platt calibrator reproduces from the TRAINING inner-OOF "
              "probabilities alone, on exactly the training rows", not bad, bad)


def f50_threshold_from_calibrated_training_inner_oof_only():
    bad = {}
    for k in FOLDS:
        z = pd.read_csv(os.path.join(ppath(CFG, "models_dir"), "A3_fusion",
                                     "pooled_fold_%d_training_inputs.csv" % k))
        y_tr = z.label.to_numpy(int)
        for m, col in (("A1", "P_ADC_GTV"), ("A2", "P_ADC_RIM")):
            cal = PlattCalibrator().fit(z[col].to_numpy(float), y_tr, CFG["calibration"])
            t, _v, _s = select_threshold(cal.apply(z[col].to_numpy(float)), y_tr,
                                         CFG["threshold"]["tie_break"])
            st = model_json(m, "pooled_fold_%d" % k)["threshold"]
            if abs(t - float(st)) > 1e-9:
                bad["fold_%d_%s" % (k, m)] = {"refit": t, "stored": st}
    used_half = [float(model_json(m, "pooled_fold_%d" % k)["threshold"])
                 for k in FOLDS for m in MODELS]
    check(50, "the operating threshold reproduces from the CALIBRATED training "
              "inner-OOF alone, and 0.5 is not the primary operating point",
          not bad and CFG["threshold"]["objective"] == "balanced_accuracy"
          and not all(abs(t - 0.5) < 1e-12 for t in used_half), bad)


# =========================================================================
# G  A1/A2 parity, capacity, A3 fusion
# =========================================================================

def g51_a1_and_a2_share_one_procedure():
    a1, a2 = CFG["models"]["A1"], CFG["models"]["A2"]
    extra = set(a1) ^ set(a2)
    ok = (a1["procedure"] == a2["procedure"] == "radiomics"
          and a1["region"] == "gtv" and a2["region"] == "rim"
          and not (extra - {"name"})
          and "A1" not in CFG["models"]["radiomics"]
          and "A2" not in CFG["models"]["radiomics"])
    # and the driver calls the identical function for both
    src = function_source("src/phase8b_run.py", "run_pooled")
    calls = re.findall(r"pooled_fold\(cfg, '(\w+)'", src)
    check(51, "A2 is A1 with the region swapped - one procedure, one grid, no "
              "rim-specific hyperparameter block",
          ok and set(calls) == {"gtv", "rim"},
          {"A1": a1, "A2": a2, "regions_passed_to_the_same_function": calls})


def g52_at_most_eight_features_everywhere():
    over = {}
    for k in FOLDS:
        for m in ("A1", "A2"):
            r = model_json(m, "pooled_fold_%d" % k)
            if r["n_selected_features"] > CFG["models"]["radiomics"]["max_selected_features"]:
                over["fold_%d_%s" % (k, m)] = r["n_selected_features"]
    for d in DIRECTIONS:
        for m in ("A1", "A2"):
            r = model_json(m, d)
            if r["n_selected_features"] > CFG["models"]["radiomics"]["max_selected_features"]:
                over["%s_%s" % (d, m)] = r["n_selected_features"]
    check(52, "no model ever exceeds 8 selected dimensions, in any fold or transfer "
              "direction", not over, over)


def g53_a3_has_exactly_two_inputs():
    bad = {}
    for k in FOLDS:
        r = model_json("A3", "pooled_fold_%d" % k)
        if len(r["coefficients"]) != 2 or r["fusion_inputs"] != ["P_ADC_GTV", "P_ADC_RIM"]:
            bad["fold_%d" % k] = r["fusion_inputs"]
    for d in DIRECTIONS:
        r = model_json("A3", d)
        if len(r["coefficients"]) != 2:
            bad[d] = len(r["coefficients"])
    check(53, "the A3 stacker has exactly two inputs - the GTV and rim ADC "
              "probabilities - in every fold and direction", not bad, bad)


def g54_a3_trained_on_inner_oof_only():
    bad = {}
    for k in FOLDS:
        tr, _ = fold_keys(k)
        z = pd.read_csv(os.path.join(ppath(CFG, "models_dir"), "A3_fusion",
                                     "pooled_fold_%d_training_inputs.csv" % k))
        if set(z.PatientKey) != set(tr) or len(z) != len(tr):
            bad["fold_%d_rows" % k] = [len(z), len(tr)]
        if set(z.source) != {"inner_oof"}:
            bad["fold_%d_source" % k] = sorted(set(z.source))
    check(54, "the fusion stacker is trained on the outer-TRAINING patients' "
              "inner-OOF component probabilities and nothing else", not bad, bad)


def g55_a3_coefficients_reproduce():
    bad = {}
    for k in FOLDS:
        tr, _ = fold_keys(k)
        z = pd.read_csv(os.path.join(ppath(CFG, "models_dir"), "A3_fusion",
                                     "pooled_fold_%d_training_inputs.csv" % k))
        z = z.set_index("PatientKey").loc[tr]
        w, _r = source_weights(CFG, np.asarray([STRAT[p] for p in tr]))
        st = fit_stacker(CFG, z[["P_ADC_GTV", "P_ADC_RIM"]].to_numpy(float),
                         z.label.to_numpy(int), w)
        stored = model_json("A3", "pooled_fold_%d" % k)
        if not (close(st.coef_[0], stored["coefficients"], 1e-6)
                and abs(float(st.intercept_[0]) - float(stored["intercept"])) <= 1e-6):
            bad["fold_%d" % k] = {"refit": list(st.coef_[0]), "stored": stored["coefficients"]}
    check(55, "the A3 stacker coefficients reproduce from the stored inner-OOF inputs "
              "under the four-stratum training weights", not bad, bad)


def g56_no_in_sample_component_prediction_trains_the_stacker():
    """The stored fusion inputs must NOT be in-sample refit predictions."""
    bad = {}
    for k in FOLDS:
        tr, _ = fold_keys(k)
        z = pd.read_csv(os.path.join(ppath(CFG, "models_dir"), "A3_fusion",
                                     "pooled_fold_%d_training_inputs.csv" % k)
                        ).set_index("PatientKey").loc[tr]
        for m, col in (("A1", "P_ADC_GTV"), ("A2", "P_ADC_RIM")):
            r = refit("pooled_fold_%d" % k, m)
            in_sample = r["est"].predict_proba(r["pre"].transform(r["mat"]["x"][r["tr"]]))
            d = float(np.max(np.abs(in_sample - z[col].to_numpy(float))))
            if d < 1e-6:
                bad["fold_%d_%s" % (k, m)] = d
    check(56, "the fusion inputs are demonstrably out-of-fold, not the in-sample "
              "predictions of the refit component model", not bad,
          {"indistinguishable_from_in_sample": bad})


def g57_a3_test_predictions_are_the_stacked_component_probabilities():
    bad = {}
    p1, p2, p3 = (preds("pooled", m).set_index("PatientKey") for m in MODELS)
    for k in FOLDS:
        tr, te = fold_keys(k)
        z = pd.read_csv(os.path.join(ppath(CFG, "models_dir"), "A3_fusion",
                                     "pooled_fold_%d_training_inputs.csv" % k)
                        ).set_index("PatientKey").loc[tr]
        w, _r = source_weights(CFG, np.asarray([STRAT[p] for p in tr]))
        st = fit_stacker(CFG, z[["P_ADC_GTV", "P_ADC_RIM"]].to_numpy(float),
                         z.label.to_numpy(int), w)
        zt = np.column_stack([p1.loc[te, "p_adc_raw"], p2.loc[te, "p_adc_raw"]])
        got = st.predict_proba(zt)[:, 1]
        if not close(got, p3.loc[te, "p_adc_raw"].to_numpy(float), 1e-6):
            bad["fold_%d" % k] = float(np.max(np.abs(got - p3.loc[te, "p_adc_raw"])))
    check(57, "every A3 outer-test probability is exactly the stacker applied to the "
              "same fold's A1 and A2 outer-test probabilities", not bad, bad)


def g58_no_high_capacity_model_anywhere():
    forbidden = ("MLPClassifier", "torch", "nn", "Conv3d", "Conv2d", "ResNet",
                 "resnet18", "SMOTE", "TransformerEncoder", "MultiheadAttention")
    hits = []
    for rel in PHASE8B_PY:
        src = executable_source(rel)
        for tok in forbidden:
            if token_present(src, tok):
                hits.append("%s: %s" % (rel, tok))
    check(58, "Phase 8B added no capacity because N grew - only logistic-family "
              "estimators appear in the source", not hits, {"tokens": hits})


# =========================================================================
# H  predictions and the three views
# =========================================================================

def h59_exactly_one_pooled_prediction_per_patient_per_model():
    bad = {}
    for m in MODELS:
        d = preds("pooled", m)
        if len(d) != 343 or d.PatientKey.duplicated().any() \
                or set(d.PatientKey) != set(COHORT.PatientKey):
            bad[m] = {"n": len(d), "n_dup": int(d.PatientKey.duplicated().sum())}
    check(59, "each pooled model produced exactly one out-of-fold prediction for each "
              "of the 343 patients", not bad, bad)


def h60_prediction_folds_match_the_frozen_split():
    frozen = {r.PatientKey: int(r.fold)
              for _, r in SPLIT[SPLIT.split == "test"].iterrows()}
    bad = {}
    for m in MODELS:
        d = preds("pooled", m)
        n = sum(1 for _, r in d.iterrows() if frozen[r.PatientKey] != int(r.fold))
        if n:
            bad[m] = n
    check(60, "every pooled prediction is attributed to the outer fold in which its "
              "patient was held out", not bad, bad)


def h61_primary_metric_is_the_raw_oof_auc():
    v1 = jload(rd("pooled_metrics.json"))
    bad = {}
    for m in MODELS:
        d = preds("pooled", m)
        got = roc_auc_score(d.label, d.p_adc_raw)
        rep = float(v1["models"][m]["raw"]["roc_auc"])
        if abs(got - rep) > 1e-9:
            bad[m] = {"recomputed": got, "reported": rep}
    ok = not bad and v1.get("primary_probability", CFG["evaluation"]["primary_probability"]) == "raw"
    check(61, "the reported primary ROC-AUC is the RAW out-of-fold probability AUC, "
              "recomputed from the frozen predictions", ok, bad)


def h62_calibrated_metrics_are_reported_but_not_primary():
    v1 = jload(rd("pooled_metrics.json"))
    ok = all("calibrated" in v1["models"][m] and "raw" in v1["models"][m] for m in MODELS)
    check(62, "calibrated and thresholded metrics are reported alongside, but do not "
              "define the primary comparison", ok and CFG["evaluation"]["report_calibrated"],
          {"models_with_both_blocks": [m for m in MODELS if "calibrated" in v1["models"][m]]})


def h63_view2_recomputed_from_the_same_pooled_predictions():
    v2 = jload(rd("cohort_specific_metrics.json"))
    bad = {}
    for m in MODELS:
        d = preds("pooled", m)
        for c in COHORTS:
            s = d[d.Cohort == c]
            got = roc_auc_score(s.label, s.p_adc_raw)
            rep = float(v2["cohorts"][c][m]["raw"]["roc_auc"])
            if abs(got - rep) > 1e-9:
                bad["%s_%s" % (c, m)] = {"recomputed": got, "reported": rep}
    check(63, "View 2 is the frozen pooled OOF predictions subset by cohort - every "
              "within-cohort AUC recomputes exactly, with no retraining", not bad, bad)


def h64_view2_involved_no_retraining():
    stray = []
    for root, _, fs in os.walk(ppath(CFG, "models_dir")):
        for f in fs:
            low = f.lower()
            if ("lung1_only" in low or "radiogenomics_only" in low
                    or "cohort_specific" in low or "view2" in low):
                stray.append(f)
    src = executable_source("src/phase8b_evaluate.py")
    check(64, "no cohort-specific model was ever fitted - View 2 has no model "
              "artefacts and the evaluator fits no classifier",
          not stray and not token_present(src, "fit_candidate"),
          {"stray_models": stray})


def h65_view3_covers_each_destination_cohort_exactly_once():
    bad = {}
    for d in CFG["validation"]["transfer"]["directions"]:
        want = set(COHORT[COHORT.Cohort == d["test"]].PatientKey)
        for m in MODELS:
            p = preds(d["id"], m)
            if set(p.PatientKey) != want or p.PatientKey.duplicated().any():
                bad["%s_%s" % (d["id"], m)] = {"n": len(p), "expected": len(want)}
    check(65, "each transport direction scored the whole destination cohort exactly "
              "once", not bad, bad)


def h66_view3_preprocessing_fitted_on_the_source_cohort_only():
    bad, indistinguishable = [], []
    for d in CFG["validation"]["transfer"]["directions"]:
        for m, region in (("A1", "gtv"), ("A2", "rim")):
            p = os.path.join(ppath(CFG, "preprocessing_dir"), "%s_%s.npz" % (d["id"], m))
            saved = np.load(p)
            src_keys = list(COHORT[COHORT.Cohort == d["train"]].PatientKey)
            pre = _pre_from(src_keys, region)
            if not (close(pre.medians, saved["medians"]) and close(pre.mean, saved["mean"])
                    and close(pre.scale, saved["scale"])
                    and np.array_equal(pre.keep.astype(np.uint8), saved["keep"])):
                bad.append("%s_%s" % (d["id"], m))
            leaky = _pre_from(list(COHORT.PatientKey), region)
            if (close(leaky.medians, saved["medians"]) and close(leaky.mean, saved["mean"])
                    and close(leaky.scale, saved["scale"])):
                indistinguishable.append("%s_%s" % (d["id"], m))
    check(66, "the transfer scaler and imputer were fitted on the SOURCE cohort only "
              "- they reproduce from it and differ from any destination-aware fit",
          not bad and not indistinguishable,
          {"not_reproducible": bad, "indistinguishable_from_leaky": indistinguishable})


def h67_destination_labels_never_affected_source_training():
    bad = {}
    for d in DIRECTIONS:
        for m in ("A1", "A2"):
            r = refit(d, m)
            if list(r["names"]) != list(r["rec"]["selected_features"]) \
                    or not close(r["est"].coefficients, r["rec"]["coefficients"]):
                bad["%s_%s" % (d, m)] = {"refit": r["names"],
                                         "stored": r["rec"]["selected_features"]}
        surf = {m: choose_candidate(CFG, surface_csv("%s_%s" % (d, m)))[0]["candidate_id"]
                for m in ("A1", "A2")}
        for m in ("A1", "A2"):
            if surf[m] != model_json(m, d)["candidate_id"]:
                bad["%s_%s_selector" % (d, m)] = [surf[m], model_json(m, d)["candidate_id"]]
    check(67, "every transfer model - preprocessing, selector, hyperparameters and "
              "coefficients - reproduces from the source cohort alone; destination "
              "labels entered only the final scoring", not bad, bad)


def h68_cross_cohort_stacker_trained_on_source_inner_oof_only():
    bad = {}
    cw = CFG["validation"]["transfer"]["class_balancing"]
    for d in CFG["validation"]["transfer"]["directions"]:
        z = pd.read_csv(os.path.join(ppath(CFG, "models_dir"), "A3_fusion",
                                     "%s_training_inputs.csv" % d["id"]))
        src_keys = set(COHORT[COHORT.Cohort == d["train"]].PatientKey)
        if set(z.PatientKey) != src_keys or set(z.source) != {"inner_oof"}:
            bad["%s_rows" % d["id"]] = {"n": len(z), "expected": len(src_keys)}
            continue
        st = fit_stacker(CFG, z[["P_ADC_GTV", "P_ADC_RIM"]].to_numpy(float),
                         z.label.to_numpy(int), None, class_weight=cw)
        stored = model_json("A3", d["id"])
        if not close(st.coef_[0], stored["coefficients"], 1e-6):
            bad[d["id"]] = {"refit": list(st.coef_[0]), "stored": stored["coefficients"]}
    check(68, "the cross-cohort fusion stacker was fitted on the TRAINING cohort's "
              "own inner-OOF predictions only", not bad, bad)


# =========================================================================
# I  diagnostics
# =========================================================================

def i69_domain_classifier_target_is_cohort_identity():
    bad = {}
    for region in MATS:
        p = rd("domain_%s_oof.csv" % region)
        d = pd.read_csv(p)
        want = (COHORT.set_index("PatientKey").loc[d.PatientKey, "Cohort"]
                == "RADIOGENOMICS").astype(int).to_numpy()
        if not np.array_equal(d.domain.to_numpy(int), want):
            bad[region] = "target is not cohort identity"
    check(69, "the domain diagnostic's target is cohort identity, re-derived from the "
              "cohort table", not bad, bad)


def i70_domain_classifier_never_sees_histology():
    src = function_source("src/phase8b_run.py", "run_domain")
    hits = [t for t in ("label", "Histology", "y_map", "histology", "cohort_histology")
            if token_present(src, t)]
    cols = set(pd.read_csv(rd("domain_gtv_oof.csv")).columns)
    check(70, "the domain classifier receives ONLY cohort identity - no histology "
              "token appears in its code and no label column in its output",
          not hits and not (cols & {"label", "Histology", "cohort_histology"})
          and CFG["domain_diagnostic"]["never_uses_histology"] is True,
          {"tokens_in_run_domain": hits, "output_columns": sorted(cols)})


def i71_domain_diagnostic_reported_for_both_regions():
    dg = jload(rd("domain_shift_diagnostic.json"))
    ok = all(r in dg["regions"] for r in ("gtv", "rim"))
    check(71, "the batch-effect diagnostic is reported separately for GTV and rim",
          ok, {r: dg["regions"][r]["pooled_oof_roc_auc"] for r in dg.get("regions", {})})


def i72_domain_diagnostic_is_not_a_disease_result():
    dg = jload(rd("domain_shift_diagnostic.json"))
    gate_metrics = json.dumps(jload(rd("gate_8b.json")))
    ok = (dg.get("uses_histology") is False
          and not re.search(r"domain", gate_metrics, re.I))
    check(72, "the domain AUC is a diagnostic - it appears in no gate condition and "
              "is never presented as a disease result", ok,
          {"uses_histology": dg.get("uses_histology")})


def i73_prediction_shift_computed_from_frozen_predictions_within_each_class():
    sh = jload(rd("prediction_shift_diagnostic.json"))["models"]
    bad = {}
    for m in MODELS:
        d = preds("pooled", m)
        for cls, lab in (("ADC", 1), ("SCC", 0)):
            for c, key in (("LUNG1", "lung1"), ("RADIOGENOMICS", "radiogenomics")):
                s = d[(d.label == lab) & (d.Cohort == c)]
                got = float(np.median(s.p_adc_raw))
                try:
                    rep = float(sh[m][cls][key]["median"])
                    n = int(sh[m][cls]["n_%s" % key])
                except (KeyError, TypeError):
                    bad["%s_%s_%s" % (m, cls, c)] = "not reported"
                    continue
                if abs(got - rep) > 1e-9 or n != len(s):
                    bad["%s_%s_%s" % (m, cls, c)] = {"recomputed": got, "reported": rep,
                                                     "n": [len(s), n]}
    ok = (not bad and CFG["prediction_shift_diagnostic"]["never_used_for_tuning"] is True
          and CFG["prediction_shift_diagnostic"]["computed_after_predictions_are_frozen"] is True)
    check(73, "the prediction-shift diagnostic compares the two cohorts WITHIN true "
              "ADC and WITHIN true SCC, recomputed from the frozen predictions", ok, bad)


# =========================================================================
# J  methods that must not appear
# =========================================================================

def _absent(tokens, files=PHASE8B_PY):
    """String constants are blanked first: `tools/phase8b_baseline.py` walks the
    project's directories BY NAME, and a directory name in a snapshot walk is
    not a use of the method it belongs to."""
    hits = []
    for rel in files:
        src = _no_strings(executable_source(rel))
        for t in tokens:
            if token_present(src, t):
                hits.append("%s: %s" % (rel, t))
    return hits


def j74_no_combat():
    hits = _absent(("combat", "ComBat", "neuroCombat", "neurocombat", "harmonize",
                    "harmonise"))
    art = [f for f in os.listdir(ppath(CFG, "results_dir"))
           if "combat" in f.lower() or "harmon" in f.lower()] \
        if os.path.isdir(ppath(CFG, "results_dir")) else []
    check(74, "ComBat is neither implemented nor run, and no harmonised feature file "
              "exists", not hits and not art and CFG["harmonisation"]["combat"] is False,
          {"tokens": hits, "artefacts": art})


def j75_no_cnn_or_embeddings():
    hits = _absent(("resnet18", "ResNet", "torch", "Conv3d", "cnn_embeddings",
                    "embeddings"))
    reads = []
    for rel in MODELLING_PY:
        src = executable_source(rel)
        if token_present(src, "phase6b") or token_present(src, "phase6a"):
            reads.append(rel)
    check(75, "no CNN, no ResNet18 embedding and no Phase-6B prediction enters any "
              "Phase-8B model", not hits and not reads,
          {"tokens": hits, "modelling_files_referencing_cnn_phases": reads})


def j76_no_clam_or_transformer_or_attention():
    hits = _absent(("CLAM", "clam", "Transformer", "TransformerEncoder", "attention",
                    "Attention", "GatedAttention", "MultiheadAttention"))
    check(76, "no CLAM, no Transformer and no attention module appears in Phase 8B",
          not hits, {"tokens": hits})


def j77_no_mil_or_patch_level_modelling():
    hits = _absent(("patch_features", "bag", "instances", "mil", "MIL"))
    check(77, "Phase 8B is global-region radiomics only - no patches, bags or MIL",
          not hits, {"tokens": hits})


def j78_no_smote_or_resampling_of_patients():
    hits = _absent(("SMOTE", "RandomOverSampler", "RandomUnderSampler", "imblearn",
                    "resample"))
    check(78, "class imbalance is handled by weighting only - no synthetic or "
              "resampled patients", not hits, {"tokens": hits})


def j79_reference_numbers_read_never_retrained():
    refs = jload(rd("reference_comparisons.json"))
    src = executable_source("src/phase8b_evaluate.py")
    ok = not token_present(src, "fit_candidate") and not token_present(src, "inner_surface")
    check(79, "the Phase-7A and Phase-6B reference numbers were read from frozen "
              "prediction files, never retrained", ok,
          {"references": list(refs) if isinstance(refs, dict) else None})


# =========================================================================
# K  bootstrap
# =========================================================================

def k80_bootstrap_unit_is_the_patient():
    b = jload(rd("paired_bootstrap.json"))
    bad = {}
    for tag, r in b.items():
        if not isinstance(r, dict) or "unit" not in r:
            continue
        if "patient" not in str(r["unit"]):
            bad[tag] = r["unit"]
    check(80, "every bootstrap resamples PATIENTS, never feature rows or folds",
          not bad and CFG["evaluation"]["bootstrap"]["unit"] == "patient", bad)


def k81_pooled_bootstrap_preserves_the_source_structure():
    b = jload(rd("paired_bootstrap.json"))
    want = {k: int(v) for k, v in CFG["evaluation"]["bootstrap"]["pooled_sizes"].items()}
    bad = {}
    for tag in ("pooled_A3_minus_A1", "pooled_A3_minus_A2"):
        r = b.get(tag)
        if r is None:
            bad[tag] = "missing"
            continue
        if r.get("block_sizes") != want \
                or "source_stratified" not in str(r.get("scheme")):
            bad[tag] = {"block_sizes": r.get("block_sizes"), "scheme": r.get("scheme")}
    # and the resampler itself preserves the blocks
    rng = np.random.default_rng(int(CFG["evaluation"]["bootstrap"]["comparisons"][0]["seed"]))
    coh = COHORT.Cohort.to_numpy()
    blocks = [np.flatnonzero(coh == c) for c in COHORTS]
    idx = np.concatenate([bk[rng.integers(0, len(bk), size=len(bk))] for bk in blocks])
    got = {c: int((coh[idx] == c).sum()) for c in COHORTS}
    check(81, "each pooled resample contains exactly 202 LUNG1 and 141 Radiogenomics "
              "patients", not bad and got == want, {"reported": bad, "resampler": got})


def k82_paired_comparisons_present():
    b = jload(rd("paired_bootstrap.json"))
    want = [c["tag"] for c in CFG["evaluation"]["bootstrap"]["comparisons"]]
    missing = [t for t in want if t not in b]
    n_bad = {t: b[t].get("n_resamples") for t in want if t in b
             and b[t].get("n_resamples") != int(CFG["evaluation"]["bootstrap"]["n_resamples"])}
    need = {"observed_difference", "ci_low", "ci_high", "ci_includes_zero"}
    metrics_bad = [t for t in want if t in b and not all(
        need <= set(b[t].get(mtr, {})) for mtr in ("roc_auc", "pr_auc"))]
    check(82, "paired 5000-resample CIs exist for A3-A1 and A3-A2 in ROC-AUC and "
              "PR-AUC, pooled and in both transport directions",
          not missing and not n_bad and not metrics_bad,
          {"missing": missing, "wrong_n": n_bad, "missing_metric": metrics_bad})


# =========================================================================
# L  frozen artefacts, read-only trees, namespace, gate
# =========================================================================

AUDIT_OUTPUTS = tuple("reports/phase%s_tests.json" % p for p in
                      ("1", "2", "3", "3b", "4", "5", "5c", "6a", "6b", "6c", "7a", "8a"))
STATUS_DOC = "docs/PROJECT_STATUS.md"


def l83_phases_1_to_8a_unchanged():
    """Byte identity for every frozen artefact.

    Two classes are verified semantically instead, exactly as Phase 8A did:
    the earlier audit outputs (re-running a suite rewrites its JSON, and the two
    declared superseded checks are handled by check 86), and the status document,
    which the project's workflow discipline REQUIRES every phase to update.
    """
    snap = jload(ppath(CFG, "pre_snapshot"))["files"]
    changed, missing, semantic = [], [], {}
    for rel, h in snap.items():
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            missing.append(rel)
            continue
        if sha256(p) == h:
            continue
        if rel in AUDIT_OUTPUTS:
            d = jload(p)
            n_pass = d.get("n_pass", d.get("n_passed"))
            n_checks = d.get("n_checks")
            declared = {int(x["check"]) for x in CFG["expected_superseded_checks"]
                        if os.path.basename(x["suite"]).replace("test_phase", "")
                        .replace(".py", "") in rel}
            failed = {c["n"] for c in d.get("checks", []) if not c["pass"]}
            cascades = {c["n"] for c in d.get("checks", [])
                        if not c["pass"] and c["n"] not in declared
                        and _is_declared_cascade(c)}
            semantic[rel] = {"n_pass": n_pass, "n_checks": n_checks,
                             "failed": sorted(failed), "declared": sorted(declared),
                             "cascade_of_a_declared_failure": sorted(cascades)}
            if failed - declared - cascades:
                changed.append(rel)
            continue
        if rel == STATUS_DOC:
            txt = open(p, encoding="utf-8").read()
            semantic[rel] = {"records_phase_7a": "Phase 7A" in txt,
                             "records_phase_8a": "Phase 8A" in txt,
                             "records_phase_8b": "Phase 8B" in txt}
            if not all(semantic[rel].values()):
                changed.append(rel)
            continue
        changed.append(rel)
    check(83, "every one of the %d Phase-1..8A artefacts is byte-identical, except "
              "the earlier audit outputs and the required status document, which are "
              "verified semantically" % len(snap), not changed and not missing,
          {"changed": changed[:20], "missing": missing[:20],
           "n_changed": len(changed), "n_missing": len(missing),
           "semantic": semantic})


def l84_readonly_trees_untouched():
    import readonly_guard as RG
    from config_io import load_config
    ok, info = RG.verify(load_config(RG.DEFAULT_CONFIG))
    check(84, "Dataset/ and Vuong5ROI/ are untouched", ok,
          {"files": info["n_baseline"], "added": info["n_added"],
           "removed": info["n_removed"], "changed": info["n_changed"]})


def l85_namespace_clean():
    snap = jload(ppath(CFG, "pre_snapshot"))["files"]
    allowed = ("docs/PHASE8B", "config/phase8b", "src/phase8b_", "tests/test_phase8b.py",
               "tools/phase8b_baseline.py", "reports/PHASE8B", "reports/phase8b",
               "logs/phase8b", "cohort/phase8b", "splits/phase8b",
               "models/phase8b", "preprocessing/phase8b", "predictions/phase8b",
               "results/phase8b", "docs/PROJECT_STATUS.md")
    stray = []
    for d in ("predictions", "results", "models", "preprocessing", "splits", "qa",
              "global_features", "feature_selection", "attention", "patch_features",
              "image_patches", "cnn_embeddings", "cohort", "patient_summaries",
              "external", "config", "src", "tools", "tests", "docs", "reports", "logs"):
        root_d = os.path.join(ROOT, d)
        if not os.path.isdir(root_d):
            continue
        for root, _, fs in os.walk(root_d):
            for f in fs:
                rel = os.path.relpath(os.path.join(root, f), ROOT).replace(os.sep, "/")
                if rel in snap or "/__pycache__/" in rel or rel.endswith(".pyc"):
                    continue
                if not any(rel.startswith(a) for a in allowed):
                    stray.append(rel)
    check(85, "Phase 8B wrote only inside its own declared namespace", not stray,
          {"stray": stray[:20]})


def l86_only_the_two_declared_frozen_checks_regress():
    """The frozen Phase-7A and Phase-8A suites must fail ONLY where declared.

    Running a suite rewrites its own results JSON, which would itself break the
    byte-identity of a frozen artefact and cascade into the NEXT suite's
    frozen-artefact check.  So each suite's results file is restored to its
    pre-run bytes immediately after it is read: this audit observes the frozen
    suites without perturbing them.
    """
    declared = {os.path.basename(d["suite"]): int(d["check"])
                for d in CFG["expected_superseded_checks"]}
    detail, ok = {}, True
    for suite in sorted(declared):
        n_expected = declared[suite]
        rel = "tests/%s" % suite
        js = os.path.join(ROOT, "reports", "phase%s_tests.json"
                          % suite.replace("test_phase", "").replace(".py", ""))
        before = open(js, "rb").read() if os.path.isfile(js) else None
        try:
            subprocess.run([sys.executable, os.path.join(ROOT, rel)], cwd=ROOT,
                           capture_output=True, text=True)
            res = jload(js) if os.path.isfile(js) else None
        finally:
            if before is not None:
                with open(js, "wb") as fh:
                    fh.write(before)
        if res is None:
            detail[suite] = "no results file"
            ok = False
            continue
        failed = [c for c in res["checks"] if not c["pass"]]
        primary = [c for c in failed if c["n"] == n_expected]
        others = [c for c in failed if c["n"] != n_expected]
        # the declared failure must exist and must name only Phase-8B paths
        blames_only_8b = bool(primary) and all(
            _paths_all_phase8b(c["detail"]) for c in primary)
        # any OTHER failure is tolerated only as a pure cascade of a declared
        # one: a frozen-artefact check whose sole complaint is that an earlier
        # audit's own results file now records its declared failure
        cascade, not_cascade = [], []
        for c in others:
            (cascade if _is_declared_cascade(c) else not_cascade).append(c["n"])
        detail[suite] = {"failed_checks": sorted(c["n"] for c in failed),
                         "declared": n_expected,
                         "declared_names_only_phase8b_paths": blames_only_8b,
                         "cascade_of_a_declared_failure": sorted(cascade),
                         "unexplained_failures": sorted(not_cascade),
                         "results_file_restored": before is not None}
        ok = ok and blames_only_8b and not not_cascade
    check(86, "the frozen Phase-7A and Phase-8A suites regress only at the two "
              "declared namespace checks - each naming Phase-8B paths alone - plus "
              "the frozen-artefact check that merely observes a declared failure "
              "being recorded", ok, detail)


def _is_declared_cascade(c):
    """A frozen-artefact check that fails ONLY because an earlier audit's own
    results file now records that audit's DECLARED superseded failure."""
    d = c["detail"]
    if not isinstance(d, dict):
        return False
    changed = d.get("changed") or d.get("files_changed") or []
    if not changed or d.get("missing"):
        return False
    for rel in changed:
        if not (isinstance(rel, str) and re.fullmatch(
                r"reports/phase[0-9a-z]+_tests\.json", rel)):
            return False
        res = jload(os.path.join(ROOT, rel))
        declared = {int(x["check"]) for x in CFG["expected_superseded_checks"]
                    if os.path.basename(x["suite"]).replace("test_phase", "")
                    .replace(".py", "") in rel}
        if {ch["n"] for ch in res["checks"] if not ch["pass"]} - declared:
            return False
    return True


def _paths_all_phase8b(detail):
    """True when every PATH-LIKE string anywhere in a failure detail is a
    Phase-8B path.  Counts, booleans and prose are not evidence of blame."""
    pathish = [s for s in _leaves(detail) if isinstance(s, str)
               and ("/" in s or s.endswith((".py", ".json", ".csv", ".npz", ".md",
                                            ".yaml", ".log")))]
    return bool(pathish) and all("phase8b" in s.lower() for s in pathish)


def _leaves(o):
    if isinstance(o, dict):
        for v in o.values():
            for x in _leaves(v):
                yield x
    elif isinstance(o, (list, tuple)):
        for v in o:
            for x in _leaves(v):
                yield x
    elif o not in (None, "", [], {}):
        yield o


def l87_gate_evaluated_mechanically():
    g = jload(rd("gate_8b.json"))
    v1, v2, v3 = (jload(rd(f)) for f in ("pooled_metrics.json",
                                         "cohort_specific_metrics.json",
                                         "cross_cohort_metrics.json"))
    d3 = preds("pooled", "A3")
    got = {
        1: roc_auc_score(d3.label, d3.p_adc_raw),
        2: roc_auc_score(d3[d3.Cohort == "LUNG1"].label,
                         d3[d3.Cohort == "LUNG1"].p_adc_raw),
        3: roc_auc_score(d3[d3.Cohort == "RADIOGENOMICS"].label,
                         d3[d3.Cohort == "RADIOGENOMICS"].p_adc_raw),
    }
    bad = {}
    for cid, val in got.items():
        rep = next((c for c in g["conditions"] if int(c["id"]) == cid), None)
        if rep is None or abs(float(rep["observed"]) - val) > 1e-9:
            bad[cid] = {"recomputed": val,
                        "reported": None if rep is None else rep["observed"]}
    n_cond = len(g["conditions"])
    check(87, "all five Gate-8B conditions were evaluated on RAW probabilities and "
              "the three AUC conditions recompute exactly from the frozen "
              "predictions", not bad and n_cond == 5,
          {"n_conditions": n_cond, "mismatch": bad, "gate_passed": g.get("passed")})


def l89_lung1_fold_coincidence_is_documented():
    """A real, explainable property of the frozen split, found by this audit.

    All 202 LUNG1 patients land in the SAME outer fold as in the frozen Phase-7A
    split.  That is not reuse and not leakage: it is deterministic.  sklearn
    encodes strata by ORDER OF FIRST APPEARANCE, so in the PatientKey-sorted
    343-patient cohort class 0 is LUNG1_SCC (n=151) and class 1 is LUNG1_ADC
    (n=51) - exactly the two classes, the two sizes and the same patient order
    as the 202-patient Phase-7A split, drawn from the same seed 42 before the
    two Radiogenomics strata consume the stream.  The fold labels are therefore
    bound to be identical.  It must be stated in the report rather than left for
    a reader to notice.
    """
    old = os.path.join(ROOT, "splits", "gtv_rim_stratified_5fold.csv")
    agree, n = 0, 0
    if os.path.isfile(old):
        o = pd.read_csv(old)
        o = o[o.split == "test"][["PatientID", "fold"]].drop_duplicates()
        m = o.merge(SPLIT[(SPLIT.split == "test") & (SPLIT.Cohort == "LUNG1")]
                    [["PatientID", "fold"]], on="PatientID", suffixes=("_old", "_new"))
        n, agree = len(m), int((m.fold_old == m.fold_new).sum())
    coincides = bool(n and agree == n)
    rep = os.path.join(ROOT, "reports", "PHASE8B_REPORT.md")
    documented = False
    if os.path.isfile(rep):
        txt = open(rep, encoding="utf-8").read().lower()
        documented = ("first appearance" in txt or "order of first" in txt) and \
                     "phase 7a" in txt and "fold" in txt
    check(89, "the exact coincidence between the LUNG1 half of the Phase-8B split "
              "and the frozen Phase-7A split is either absent or explained in the "
              "report", (not coincides) or documented,
          {"n_lung1_compared": n, "n_same_fold": agree, "coincides": coincides,
           "explained_in_report": documented})


def l88_gate_verdict_consistent_with_the_report():
    g = jload(rd("gate_8b.json"))
    passed = bool(g.get("passed"))
    rep = os.path.join(ROOT, "reports", "PHASE8B_REPORT.md")
    ok, detail = True, {"gate_passed": passed}
    if os.path.isfile(rep):
        txt = open(rep, encoding="utf-8").read()
        said_pass = re.search(r"GATE\s*8B[^\n]*\bPASS", txt, re.I) is not None
        said_fail = re.search(r"GATE\s*8B[^\n]*\bFAIL", txt, re.I) is not None
        ok = (said_pass and not said_fail) if passed else (said_fail and not said_pass)
        detail.update({"report_says_pass": said_pass, "report_says_fail": said_fail})
    else:
        detail["report"] = "not written yet"
        ok = False
    check(88, "the report's Gate-8B verdict is the one the frozen results support",
          ok, detail)


def main():
    order = [a1_protocol_frozen_and_hashed, a2_config_frozen,
             a3_frozen_before_any_model_ran, a4_external_status_retired,
             a5_no_external_validation_claim, a6_scope_declared,
             a7_primary_metric_is_raw_and_frozen, a8_phase7a_report_not_altered,
             a9_no_scientific_parameter_hard_coded, a10_no_phase8c_artefact,
             b11_patient_keys_unique_and_formatted, b12_never_relies_on_raw_patient_id,
             b13_cohort_counts, b14_class_counts, b15_stratum_counts,
             b16_no_raw_id_collision, b17_no_duplicate_feature_rows,
             b18_cross_cohort_nearest_neighbour_screen,
             c19_cached_feature_files_unchanged, c20_no_pyradiomics_rerun,
             c21_feature_names_identical_and_ordered, c22_extraction_signature,
             c23_feature_counts_and_no_rim_shape, c24_no_nan_or_inf,
             c25_feature_matrix_aligned_to_cohort,
             d26_split_reproduces_from_the_frozen_seed, d27_split_file_unchanged,
             d28_every_fold_has_all_four_strata, d29_split_partitions_the_cohort,
             d30_stratified_on_cohort_histology_not_label, d31_split_regenerated_not_copied,
             d32_inner_folds_reproduce_and_preserve_strata,
             d33_transfer_definition_is_the_two_full_cohorts,
             e34_weights_reproduce_from_training_counts,
             e35_equal_total_weight_per_stratum, e36_mean_weight_is_one,
             e37_weights_never_use_outer_test_counts,
             e38_weights_not_applied_to_calibration,
             f39_imputation_fitted_on_training_rows_only,
             f40_variance_filter_fitted_on_training_rows_only,
             f41_scaler_fitted_on_training_rows_only,
             f42_including_outer_test_rows_would_have_changed_preprocessing,
             f43_feature_selection_reproduces_from_training_rows_only,
             f44_coefficients_reproduce_from_training_rows_only,
             f45_selector_choice_reproduces_from_the_inner_surface,
             f46_inner_surface_carries_no_outer_test_information,
             f47_the_grid_is_the_frozen_grid, f48_only_two_selector_families,
             f49_calibration_fitted_on_training_inner_oof_only,
             f50_threshold_from_calibrated_training_inner_oof_only,
             g51_a1_and_a2_share_one_procedure, g52_at_most_eight_features_everywhere,
             g53_a3_has_exactly_two_inputs, g54_a3_trained_on_inner_oof_only,
             g55_a3_coefficients_reproduce,
             g56_no_in_sample_component_prediction_trains_the_stacker,
             g57_a3_test_predictions_are_the_stacked_component_probabilities,
             g58_no_high_capacity_model_anywhere,
             h59_exactly_one_pooled_prediction_per_patient_per_model,
             h60_prediction_folds_match_the_frozen_split,
             h61_primary_metric_is_the_raw_oof_auc,
             h62_calibrated_metrics_are_reported_but_not_primary,
             h63_view2_recomputed_from_the_same_pooled_predictions,
             h64_view2_involved_no_retraining,
             h65_view3_covers_each_destination_cohort_exactly_once,
             h66_view3_preprocessing_fitted_on_the_source_cohort_only,
             h67_destination_labels_never_affected_source_training,
             h68_cross_cohort_stacker_trained_on_source_inner_oof_only,
             i69_domain_classifier_target_is_cohort_identity,
             i70_domain_classifier_never_sees_histology,
             i71_domain_diagnostic_reported_for_both_regions,
             i72_domain_diagnostic_is_not_a_disease_result,
             i73_prediction_shift_computed_from_frozen_predictions_within_each_class,
             j74_no_combat, j75_no_cnn_or_embeddings,
             j76_no_clam_or_transformer_or_attention,
             j77_no_mil_or_patch_level_modelling, j78_no_smote_or_resampling_of_patients,
             j79_reference_numbers_read_never_retrained,
             k80_bootstrap_unit_is_the_patient,
             k81_pooled_bootstrap_preserves_the_source_structure,
             k82_paired_comparisons_present,
             l83_phases_1_to_8a_unchanged, l84_readonly_trees_untouched,
             l85_namespace_clean, l86_only_the_two_declared_frozen_checks_regress,
             l87_gate_evaluated_mechanically, l88_gate_verdict_consistent_with_the_report,
             l89_lung1_fold_coincidence_is_documented]

    for i, fn in enumerate(order, 1):
        try:
            fn()
        except Exception as exc:                     # a raising check is a FAIL
            check(i if len(CHECKS) < i else len(CHECKS) + 1, fn.__name__, False,
                  "%s: %s" % (type(exc).__name__, exc))

    n_pass = sum(1 for c in CHECKS if c["pass"])
    out = {"phase": "8B", "n_checks": len(CHECKS), "n_pass": n_pass,
           "all_pass": n_pass == len(CHECKS),
           "protocol_sha256": protocol_sha(CFG), "checks": CHECKS}
    with open(ppath(CFG, "tests_json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\n%d/%d Phase-8B checks pass -> %s"
          % (n_pass, len(CHECKS), ppath(CFG, "tests_json")))
    sys.exit(0 if n_pass == len(CHECKS) else 1)


if __name__ == "__main__":
    main()
