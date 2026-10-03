"""Phase-6C leakage, scope, instance-supervision and non-regression checks.

Run with:
    ./.venv/Scripts/python.exe tests/test_phase6c.py

Phase 6C trains ONE model: the frozen Phase-6B gated-attention MIL on the frozen
Phase-6A ResNet18 embeddings, plus an auxiliary instance-level classifier and
instance loss, under the frozen Phase-5/6B nested-CV procedure with a joint
(lambda_instance x epoch) selection.  These checks re-derive the whole chain from
files on disk rather than trusting any recorded value.

Groups and the 34 verifications (the 30 required + 4 more)
  A  cohort, frozen split, frozen embeddings, frozen Phase-6B baseline  (1-5)
  B  leakage: splits, preprocessing, weighting, lambda and epoch choice (6-11)
  C  what is NOT in the trainable model or the training path            (12-17, 30)
  D  the instance supervision itself                                    (18-23)
  E  numerical sanity and the attention outputs                         (24-26, 33)
  F  the paired bootstraps                                              (27, 31)
  G  frozen artefacts and the read-only external trees                  (28-29)
  H  deliverables and reported preprocessing                            (32, 34)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from clam_mil import (ClamStyleGatedAttentionMIL, adaptive_k, bag_k,  # noqa: E402
                      build_clam_model, k_rule_from_config, select_pseudo_instances)
from mil_data import (fit_fold_scaler, load_cohort, load_outer_split,  # noqa: E402
                      load_phase3_config, model_tag, outer_fold_patients,
                      pos_weight_from_patients, ppath, sha256_file)
from mil_metrics import patient_metrics  # noqa: E402
from mil_models import GatedAttentionMIL, build_model  # noqa: E402
from repr_paired_compare import paired_bootstrap  # noqa: E402
from train_mil_clam import (lambda_grid, run_training,  # noqa: E402
                            select_lambda_and_epoch)
from train_mil_cv import inner_cv_assignment, make_split  # noqa: E402

CFG = load_phase3_config(os.path.join(ROOT, "config", "phase6c.yaml"))
CFG6B = load_phase3_config(os.path.join(ROOT, "config", "phase6b.yaml"))
MODEL = "clam_attention_mil"
FOLDS = (1, 2, 3, 4, 5)
N_INNER = int(CFG["inner_cv"]["n_splits"])
THR = float(CFG["evaluation"]["threshold"])
N_PATIENTS = int(CFG["phase3"]["expected_patients"])
N_INSTANCES = int(CFG["phase3"]["expected_instances"])
N_DIMS = int(CFG["phase3"]["expected_features"])
EMB_DIR = os.path.join(ROOT, *CFG["phase3"]["features_dir"].split("/"))
GRID = lambda_grid(CFG)
KRULE = k_rule_from_config(CFG)
CHECKS = []

COHORT = load_cohort(CFG)
SPLIT = load_outer_split(CFG)
RUN = json.load(open(ppath(CFG, CFG["outputs"]["run_json"]), "r", encoding="utf-8"))

# Every Phase-6C source file that trains, evaluates or diagnoses.  The audit file
# itself is deliberately EXCLUDED: the forbidden-token scans assert what the
# PIPELINE does, and a scanner carrying its own search lists would self-hit (the
# defect found in Phase 5C, again in Phase 6A and again in Phase 6B).
PHASE6C_PY = ("src/clam_mil.py", "src/train_mil_clam.py", "src/evaluate_phase6c.py",
              "src/phase6c_pooling_bootstrap.py", "src/train_mil_cv.py",
              "src/train_mil.py", "src/mil_data.py", "src/mil_models.py",
              "src/mil_metrics.py", "src/repr_paired_compare.py")

# Audit OUTPUTS, not data artefacts: they are rewritten whenever their own suite
# is re-run for non-regression and they record their own wall-clock runtime, so
# they can never be byte-identical across runs.  Both are verified SEMANTICALLY
# instead (their suites must still pass in full).
AUDIT_OUTPUT_EXCEPTIONS = ("reports/phase6a_tests.json", "reports/phase6b_tests.json")


def check(group, name):
    def deco(fn):
        CHECKS.append((group, name, fn))
        return fn
    return deco


# ------------------------------------------------------------------- helpers
def source(rel):
    with open(os.path.join(ROOT, *rel.split("/")), "r", encoding="utf-8") as fh:
        return fh.read()


def code_only(rel):
    """Source with every comment and string literal removed."""
    import io as _io
    import tokenize as _tk
    out, last = [], 1
    for tok in _tk.generate_tokens(_io.StringIO(source(rel)).readline):
        if tok.type in (_tk.COMMENT, _tk.STRING):
            continue
        if tok.start[0] != last:
            out.append("\n")
            last = tok.start[0]
        out.append(" " + tok.string)
    return "".join(out)


def imported_modules(rel):
    import ast as _ast
    mods = set()
    for node in _ast.walk(_ast.parse(source(rel))):
        if isinstance(node, _ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, _ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
            mods.update(a.name for a in node.names)
    return mods


def scan(tokens, files=PHASE6C_PY):
    """(file, token) pairs where a forbidden token appears in executable code.

    Matching is at identifier boundaries, never by substring, so the honest
    metric `confusion_matrix` cannot count as a "fusion" construct.
    """
    import re as _re
    hits = []
    for rel in files:
        src = code_only(rel).lower()
        for t in tokens:
            pat = r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % _re.escape(t.lower())
            if _re.search(pat, src):
                hits.append((rel, t))
    return hits


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot():
    return json.load(open(ppath(CFG, "reports/phase6c_pre_snapshot.json"),
                          "r", encoding="utf-8"))["files"]


def preds6c():
    return pd.read_csv(os.path.join(ppath(CFG, CFG["outputs"]["predictions_dir"]),
                                    "%s_oof.csv" % model_tag(CFG, MODEL)))


def preds(rel):
    return pd.read_csv(os.path.join(ROOT, *rel.split("/")))


def inner_df(fold):
    return pd.read_csv(os.path.join(ppath(CFG, CFG["outputs"]["inner_splits_dir"]),
                                    "fold_%d.csv" % fold))


def prep_dir():
    return ppath(CFG, CFG["outputs"]["preprocessing_dir"])


def scaler_npz(fold, inner=None):
    name = ("fold_%d_final_scaler.npz" % fold if inner is None
            else "fold_%d_inner_%d_scaler.npz" % (fold, inner))
    return np.load(os.path.join(prep_dir(), name), allow_pickle=True)


def scaler_meta(fold):
    return json.load(open(os.path.join(prep_dir(), "fold_%d_scalers.json" % fold),
                          "r", encoding="utf-8"))


def sel_dir():
    return ppath(CFG, CFG["outputs"]["selection_dir"])


def surface_df(fold):
    return pd.read_csv(os.path.join(sel_dir(), "fold_%d_surface.csv" % fold))


def per_bag_df(fold):
    return pd.read_csv(os.path.join(sel_dir(), "fold_%d_refit_per_bag.csv" % fold))


def checkpoint(fold):
    return torch.load(os.path.join(ppath(CFG, CFG["outputs"]["models_dir"]),
                                   model_tag(CFG, MODEL), "fold_%d.pt" % fold),
                      map_location="cpu", weights_only=False)


def outer_sets(fold):
    sub = SPLIT[SPLIT["fold"] == fold]
    return (set(sub.loc[sub["split"] == "train", "PatientID"]),
            set(sub.loc[sub["split"] == "test", "PatientID"]))


def att_dir():
    return ppath(CFG, CFG["outputs"]["attention_dir"])


def fold_detail(fold):
    return next(d for d in RUN["folds_detail"] if int(d["fold"]) == int(fold))


def earlier_audit_ok(rel, expected_checks):
    with open(os.path.join(ROOT, *rel.split("/")), "r", encoding="utf-8") as fh:
        d = json.load(fh)
    ok = (int(d["n_checks"]) == expected_checks and int(d["n_failed"]) == 0)
    return ok, "%s %s/%s" % (rel, d["n_passed"], d["n_checks"])


# ================================ A  cohort / frozen split / frozen baseline
@check("A", "1 exactly 202 patients, 51 ADC / 151 SCC, everywhere")
def a1():
    files = sorted(f[:-4] for f in os.listdir(EMB_DIR) if f.endswith(".npz"))
    lab = np.array([COHORT.bags[p].label for p in COHORT.patient_ids])
    counts = {"embedding files": len(files), "loaded bags": len(COHORT.bags),
              "split patients": SPLIT["PatientID"].nunique(),
              "phase6c predictions": len(preds6c()),
              "phase6c attention files": len([f for f in os.listdir(att_dir())
                                              if f.endswith(".npz")])}
    bad = {k: v for k, v in counts.items() if v != N_PATIENTS}
    if bad:
        return False, "not %d: %s" % (N_PATIENTS, bad)
    if files != sorted(COHORT.patient_ids):
        return False, "embedding file names disagree with the loaded cohort"
    if int((lab == 1).sum()) != 51 or int((lab == 0).sum()) != 151:
        return False, "%d ADC / %d SCC" % (int((lab == 1).sum()), int((lab == 0).sum()))
    n_inst = int(sum(b.n_patches for b in COHORT.bags.values()))
    if n_inst != N_INSTANCES:
        return False, "%d instances, expected %d" % (n_inst, N_INSTANCES)
    return True, ("202 in all of: embedding files, loaded bags, split, the Phase-6C "
                  "prediction file and the attention directory; 51 ADC / 151 SCC; "
                  "%d instances" % n_inst)


@check("A", "2 the frozen outer split is unchanged")
def a2():
    meta = json.load(open(ppath(CFG, CFG["phase3"]["splits_meta_file"]),
                          "r", encoding="utf-8"))
    now = sha256_file(ppath(CFG, CFG["phase3"]["splits_file"]))
    if now != meta["splits_file_sha256"]:
        return False, "split sha256 %s != its Phase-4 metadata" % now[:16]
    if now != RUN["input_manifest"]["splits_file_sha256"]:
        return False, "the split changed during the Phase-6C run"
    if now != RUN["preflight_verification"]["splits_file_sha256"]:
        return False, "the pre-flight verification saw a different split"
    if now != snapshot()["splits/gtv_rim_stratified_5fold.csv"]:
        return False, "the split differs from the pre-Phase-6C snapshot"
    # identical outer membership to Phase 6B
    for f in FOLDS:
        tr, te = outer_sets(f)
        d = inner_df(f)
        if set(d.loc[d["assignment"] == "outer_test", "PatientID"]) != te or \
           set(d.loc[d["assignment"] == "outer_train", "PatientID"]) != tr:
            return False, "fold %d membership changed" % f
        d6b = pd.read_csv(os.path.join(ppath(CFG6B, CFG6B["outputs"]["inner_splits_dir"]),
                                       "fold_%d.csv" % f))
        a = d.sort_values("PatientID")[["PatientID", "assignment", "inner_fold"]]
        b = d6b.sort_values("PatientID")[["PatientID", "assignment", "inner_fold"]]
        if not a.reset_index(drop=True).equals(b.reset_index(drop=True)):
            return False, "fold %d inner-CV assignment differs from Phase 6B" % f
    union = sorted(p for f in FOLDS for p in outer_sets(f)[1])
    if union != sorted(COHORT.patient_ids):
        return False, "the 5 test folds do not partition the 202 patients"
    return True, ("sha256 %s… matches its Phase-4 metadata, the run manifest, the "
                  "pre-flight record and the snapshot; the 5 outer folds partition all "
                  "202; every inner-CV assignment is identical to Phase 6B" % now[:16])


@check("A", "3 the frozen 512-D CNN embeddings are unchanged")
def a3():
    snap = snapshot()
    changed, missing = [], []
    for pid in COHORT.patient_ids:
        rel = "%s/%s.npz" % (CFG["phase3"]["features_dir"], pid)
        if rel not in snap:
            missing.append(rel)
        elif sha256(os.path.join(EMB_DIR, "%s.npz" % pid)) != snap[rel]:
            changed.append(rel)
    manifest = RUN["input_manifest"]["bag_sha256"]
    drift = [p for p in COHORT.patient_ids
             if manifest.get("%s.npz" % p) != sha256(os.path.join(EMB_DIR, "%s.npz" % p))]
    widths = set()
    for pid in COHORT.patient_ids:
        with np.load(os.path.join(EMB_DIR, "%s.npz" % pid), allow_pickle=True) as z:
            widths.add(int(z["embeddings"].shape[1]))
            if not bool(z["encoder_frozen"]):
                return False, "%s: encoder_frozen is False" % pid
    ok = (not changed and not missing and not drift and widths == {N_DIMS}
          and RUN["frozen_inputs_unchanged"])
    return ok, ("202 embedding files byte-identical to the pre-Phase-6C snapshot "
                "(%d changed, %d missing) and to the run's own before/after manifest "
                "(%d drifted); every file is [N, 512] and encoder_frozen"
                % (len(changed), len(missing), len(drift)))


@check("A", "4 the frozen Phase-6B predictions and attention are unchanged")
def a4():
    snap = snapshot()
    rels = ["predictions/phase6b_cnn_attention_oof.csv",
            "predictions/phase6b_cnn_mean_oof.csv",
            "results/phase6b/oof_metrics.json"]
    rels += sorted(r for r in snap if r.startswith("attention/gtv_rim_resnet18_phase6b/"))
    rels += sorted(r for r in snap if r.startswith("models/phase6b_cnn_"))
    changed = [r for r in rels
               if sha256(os.path.join(ROOT, *r.split("/"))) != snap[r]]
    if changed:
        return False, "%d frozen Phase-6B artefacts CHANGED (%s)" % (len(changed),
                                                                     changed[:3])
    df = preds("predictions/phase6b_cnn_attention_oof.csv")
    m = patient_metrics(df["true_label"].values,
                        df["predicted_probability_ADC"].values, THR)
    want_roc = float(CFG["comparison"]["primary"]["reference_pooled_roc_auc"])
    want_pr = float(CFG["comparison"]["primary"]["reference_pooled_pr_auc"])
    if abs(m["roc_auc"] - want_roc) > 5e-4 or abs(m["pr_auc"] - want_pr) > 5e-4:
        return False, "the Phase-6B baseline no longer scores %.4f / %.4f: %.4f / %.4f" \
                      % (want_roc, want_pr, m["roc_auc"], m["pr_auc"])
    if RUN["frozen_baseline"]["retrained"]:
        return False, "the run record claims the baseline was retrained"
    return True, ("%d frozen Phase-6B artefacts byte-identical (predictions, attention, "
                  "checkpoints, results) and the baseline still scores pooled OOF "
                  "ROC-AUC %.4f / PR-AUC %.4f; it was never retrained"
                  % (len(rels), m["roc_auc"], m["pr_auc"]))


@check("A", "5 exactly one Phase-6C out-of-fold prediction per patient")
def a5():
    df = preds6c()
    if df["PatientID"].duplicated().any():
        return False, "duplicate out-of-fold predictions"
    if sorted(df["PatientID"]) != sorted(COHORT.patient_ids):
        return False, "prediction patients differ from the cohort"
    if sorted(df["model"].unique()) != [MODEL]:
        return False, "unexpected model column: %s" % sorted(df["model"].unique())
    for _, r in df.iterrows():
        if r["PatientID"] not in outer_sets(int(r["outer_fold"]))[1]:
            return False, "%s scored by a fold it was not held out of" % r["PatientID"]
        if int(COHORT.bags[r["PatientID"]].label) != int(r["true_label"]):
            return False, "%s: label disagrees with the frozen bag" % r["PatientID"]
    if not np.all(np.isfinite(df["predicted_probability_ADC"].values)):
        return False, "non-finite probabilities"
    return True, ("202 unique predictions, one model (%s); every patient scored exactly "
                  "once by the model of its own held-out fold, with the frozen label"
                  % MODEL)


# ================================================================ B  leakage
@check("B", "6 no outer-test patient takes part in any inner fitting")
def b6():
    for f in FOLDS:
        tr, te = outer_sets(f)
        d = inner_df(f)
        assign = {r["PatientID"]: int(r["inner_fold"]) for _, r in d.iterrows()
                  if r["assignment"] == "outer_train"}
        if set(assign) != tr:
            return False, "fold %d: the inner assignment does not cover outer-train" % f
        if te & set(assign):
            return False, "fold %d: an outer-test patient carries an inner fold" % f
        for j in range(1, N_INNER + 1):
            itr = {p for p, k in assign.items() if k != j}
            iva = {p for p, k in assign.items() if k == j}
            if itr & iva or itr & te or iva & te:
                return False, "fold %d inner %d: overlap" % (f, j)
            if itr | iva != tr:
                return False, "fold %d inner %d: not exhaustive" % (f, j)
            yv = COHORT.labels(sorted(iva))
            if len(np.unique(yv)) < 2:
                return False, "fold %d inner %d: single-class validation" % (f, j)
        # the per-bag instance-supervision record must contain outer-train only
        pb = per_bag_df(f)
        if set(pb["PatientID"]) - tr:
            return False, "fold %d: instance supervision touched a non-training bag" % f
    return True, ("5 x 3 inner splits are disjoint, exhaustive over the outer-training "
                  "patients and two-class; no outer-test patient appears in any inner "
                  "split or in any instance-supervision record")


@check("B", "7 the variance screen and both scalers see TRAINING patches only")
def b7():
    for f in FOLDS:
        tr, te = outer_sets(f)
        d = inner_df(f)
        assign = {r["PatientID"]: int(r["inner_fold"]) for _, r in d.iterrows()
                  if r["assignment"] == "outer_train"}
        for j in range(1, N_INNER + 1):
            itr = sorted(p for p, k in assign.items() if k != j)
            iva = sorted(p for p, k in assign.items() if k == j)
            want = fit_fold_scaler(CFG, COHORT, itr)
            got = scaler_npz(f, j)
            if not (np.allclose(want.mean, got["mean"]) and np.allclose(want.std, got["std"])
                    and np.array_equal(want.keep_idx, got["keep_idx"])):
                return False, "fold %d inner %d scaler does not refit exactly" % (f, j)
            # provably different from a scaler that also saw the held-out patients
            for extra, tag in ((iva, "inner-val"), (sorted(te), "outer-test")):
                bad = fit_fold_scaler(CFG, COHORT, itr + list(extra))
                if np.allclose(want.mean, bad.mean) and np.allclose(want.std, bad.std):
                    return False, ("fold %d inner %d scaler is indistinguishable from a "
                                   "%s-contaminated one" % (f, j, tag))
        want = fit_fold_scaler(CFG, COHORT, sorted(tr))
        got = scaler_npz(f)
        if not (np.allclose(want.mean, got["mean"]) and np.allclose(want.std, got["std"])):
            return False, "fold %d final scaler does not refit exactly" % f
        bad = fit_fold_scaler(CFG, COHORT, sorted(tr) + sorted(te))
        if np.allclose(want.mean, bad.mean) and np.allclose(want.std, bad.std):
            return False, "fold %d final scaler is indistinguishable from a contaminated one" % f
        ck = checkpoint(f)
        if not (np.allclose(ck["scaler_mean"], want.mean)
                and np.allclose(ck["scaler_std"], want.std)):
            return False, "fold %d checkpoint carries a different scaler" % f
    return True, ("15 inner scalers refit exactly from inner-training patches and 5 final "
                  "scalers from ALL outer-training patches; each is provably different "
                  "from a val- or test-contaminated variant and matches its checkpoint")


@check("B", "8 pos_weight comes from PATIENT counts, never patch counts")
def b8():
    for f in FOLDS:
        tr, te = outer_sets(f)
        d = inner_df(f)
        assign = {r["PatientID"]: int(r["inner_fold"]) for _, r in d.iterrows()
                  if r["assignment"] == "outer_train"}
        want = pos_weight_from_patients(COHORT.labels(sorted(tr)))
        meta = scaler_meta(f)
        if abs(float(meta["final_pos_weight"]) - want) > 1e-12:
            return False, "fold %d final pos_weight %.6f != %.6f" \
                          % (f, meta["final_pos_weight"], want)
        if abs(float(checkpoint(f)["pos_weight"]) - want) > 1e-12:
            return False, "fold %d checkpoint pos_weight differs" % f
        # never a patch ratio
        n_adc_p = sum(COHORT.bags[p].n_patches for p in tr if COHORT.bags[p].label == 1)
        n_scc_p = sum(COHORT.bags[p].n_patches for p in tr if COHORT.bags[p].label == 0)
        if abs(want - n_scc_p / n_adc_p) < 1e-9:
            return False, "fold %d pos_weight equals the PATCH ratio" % f
        for j in range(1, N_INNER + 1):
            itr = sorted(p for p, k in assign.items() if k != j)
            wj = pos_weight_from_patients(COHORT.labels(itr))
            if abs(float(scaler_npz(f, j)["pos_weight"]) - wj) > 1e-12:
                return False, "fold %d inner %d pos_weight differs" % (f, j)
    # the instance loss must NOT reuse it
    src = code_only("src/clam_mil.py")
    if "pos_weight" in src:
        return False, "clam_mil.py references pos_weight in executable code"
    if str(CFG["model"]["instance_head"]["pos_weight"]) not in ("None", "none"):
        return False, "the config declares a pos_weight for the instance head"
    return True, ("5 final + 15 inner pos_weights equal their PATIENT ratios exactly and "
                  "none equals a patch ratio; the instance loss carries no pos_weight "
                  "(top-k and bottom-k are equal in size, so the pseudo targets are "
                  "balanced within every bag)")


@check("B", "9 lambda_instance is chosen from the inner-CV surface ONLY")
def b9():
    for f in FOLDS:
        s = surface_df(f)
        grid = sorted(s["lambda_instance"].unique())
        if [round(g, 10) for g in grid] != [round(g, 10) for g in GRID]:
            return False, "fold %d: surface covers %s, not %s" % (f, grid, GRID)
        mat = np.array([s[np.isclose(s["lambda_instance"], lam)]
                        .sort_values("epoch")["mean_inner_cv_roc_auc"].values
                        for lam in GRID])
        lam, ep, auc, _, _ = select_lambda_and_epoch(mat, GRID)
        rec = float(fold_detail(f)["selected_lambda_instance"])
        if abs(lam - rec) > 1e-12:
            return False, "fold %d: recorded lambda %.4f != surface argmax %.4f" % (f, rec, lam)
        if abs(float(checkpoint(f)["selected_lambda_instance"]) - lam) > 1e-12:
            return False, "fold %d: checkpoint lambda differs" % f
        if abs(float(preds6c().query("outer_fold == @f")["selected_lambda_instance"]
                     .iloc[0]) - lam) > 1e-12:
            return False, "fold %d: prediction rows record a different lambda" % f
        # it can never equal a choice made on the outer test set
        te_auc = float(fold_detail(f)["fold_metrics"]["roc_auc"])
        if abs(auc - te_auc) < 1e-12:
            return False, "fold %d: the selection value equals the outer-test ROC-AUC" % f
        # tie-break: no smaller lambda reaches the selected value
        li = GRID.index(lam)
        for lj in range(li):
            if mat[lj].max() >= auc - 1e-15:
                return False, "fold %d: a smaller lambda ties and was not preferred" % f
    return True, ("all 5 lambda_instance choices are the joint argmax of the stored "
                  "(lambda x epoch) mean inner-CV ROC-AUC surface, agree with the "
                  "checkpoints and the prediction rows, break ties toward the smaller "
                  "lambda, and never equal an outer-test ROC-AUC")


@check("B", "10 the epoch is chosen from the same inner-CV surface ONLY")
def b10():
    for f in FOLDS:
        s = surface_df(f)
        mat = np.array([s[np.isclose(s["lambda_instance"], lam)]
                        .sort_values("epoch")["mean_inner_cv_roc_auc"].values
                        for lam in GRID])
        if mat.shape[1] != int(CFG["training"]["max_epochs"]):
            return False, "fold %d: surface spans %d epochs" % (f, mat.shape[1])
        lam, ep, auc, li, ei = select_lambda_and_epoch(mat, GRID)
        rec = int(fold_detail(f)["selected_epoch"])
        if ep != rec:
            return False, "fold %d: recorded epoch %d != surface argmax %d" % (f, rec, ep)
        if int(checkpoint(f)["selected_epoch"]) != ep:
            return False, "fold %d: checkpoint epoch differs" % f
        if int(preds6c().query("outer_fold == @f")["selected_epoch"].iloc[0]) != ep:
            return False, "fold %d: prediction rows record a different epoch" % f
        # lower epochs at the selected lambda must be strictly worse (lowest-epoch tie-break)
        if ei > 0 and mat[li, :ei].max() >= auc - 1e-15:
            return False, "fold %d: a lower epoch ties and was not preferred" % f
        # the per-lambda inner curves must average to the surface row
        for lj, lm in enumerate(GRID):
            lam_tag = ("%.4f" % lm).rstrip("0").rstrip(".").replace(".", "p")
            c = pd.read_csv(os.path.join(sel_dir(), "fold_%d_lambda_%s.csv" % (f, lam_tag)))
            cols = [x for x in c.columns if x.endswith("_val_roc_auc")
                    and x.startswith("inner_")]
            if len(cols) != N_INNER:
                return False, "fold %d lambda %.2f: %d inner curves" % (f, lm, len(cols))
            if not np.allclose(c[cols].values.mean(axis=1), mat[lj], atol=1e-12):
                return False, "fold %d lambda %.2f: the mean curve is not the mean" % (f, lm)
    eps = {f: int(fold_detail(f)["selected_epoch"]) for f in FOLDS}
    return True, ("all 5 epochs are the joint argmax of the mean of the 3 stored inner "
                  "curves, break ties toward the lower epoch, and agree with the "
                  "checkpoints and the prediction rows: %s" % eps)


@check("B", "11 the final refit uses exactly the selected lambda and epoch")
def b11():
    n = 0
    for f in FOLDS:
        tr, te = outer_sets(f)
        split = make_split(CFG, COHORT, sorted(tr), sorted(te))
        lam = float(fold_detail(f)["selected_lambda_instance"])
        ep = int(fold_detail(f)["selected_epoch"])
        seed = int(CFG["seeds"]["final_base_seed"]) + 10 * f
        if seed != int(checkpoint(f)["seed"]):
            return False, "fold %d: refit seed differs from the checkpoint" % f
        model, _, _ = run_training(CFG, COHORT, split, seed, ep, lam,
                                   eval_every_epoch=False)
        from clam_mil import predict_clam
        prob, _ = predict_clam(model, split.tensors, split.evaluate,
                               torch.device("cpu"))
        rec = preds6c().set_index("PatientID")
        for pid, p in zip(split.evaluate, prob):
            if abs(float(rec.loc[pid, "predicted_probability_ADC"]) - float(p)) > 1e-9:
                return False, "fold %d %s: %.10f != %.10f" \
                              % (f, pid, rec.loc[pid, "predicted_probability_ADC"], p)
            n += 1
    return True, ("all %d out-of-fold predictions reproduce to < 1e-9 by retraining from "
                  "scratch on all outer-training patients with the selected "
                  "lambda_instance for exactly the selected number of epochs" % n)


# ============================================= C  what is NOT in the model
@check("C", "12 the architecture matches Phase 6B except the instance head")
def c12():
    for f in FOLDS:
        ck = checkpoint(f)
        sd = ck["model_state_dict"]
        ref = build_model("attention_mil", CFG6B, input_dim=int(ck["input_dim"]))
        rsd = ref.state_dict()
        extra = sorted(k for k in sd if k not in rsd)
        missing = sorted(k for k in rsd if k not in sd)
        if extra != ["instance_classifier.bias", "instance_classifier.weight"]:
            return False, "fold %d: unexpected extra tensors %s" % (f, extra)
        if missing:
            return False, "fold %d: missing Phase-6B tensors %s" % (f, missing)
        for k in rsd:
            if tuple(sd[k].shape) != tuple(rsd[k].shape):
                return False, "fold %d: %s has shape %s, Phase-6B has %s" \
                              % (f, k, tuple(sd[k].shape), tuple(rsd[k].shape))
        if tuple(sd["instance_classifier.weight"].shape) != (1, 32) or \
           tuple(sd["instance_classifier.bias"].shape) != (1,):
            return False, "fold %d: the instance head is not Linear(32, 1)" % f
        # loading the bag-level half into a Phase-6B model must succeed exactly
        ref.load_state_dict({k: sd[k] for k in rsd})
        # and it must be the SAME arithmetic
        m = build_clam_model(CFG, input_dim=int(ck["input_dim"]))
        m.load_state_dict(sd)
        m.eval(); ref.eval()
        x = torch.from_numpy(np.linspace(-2, 2, 40 * int(ck["input_dim"]))
                             .reshape(40, int(ck["input_dim"])).astype(np.float32))
        with torch.no_grad():
            l1, a1 = m(x)
            l2, a2 = ref(x)
        if abs(float(l1) - float(l2)) > 0 or float((a1 - a2).abs().max()) > 0:
            return False, "fold %d: the bag-level forward pass differs from Phase 6B" % f
        n_par = sum(p.numel() for p in m.parameters())
        n_ref = sum(p.numel() for p in ref.parameters())
        if n_par - n_ref != 33:
            return False, "fold %d: %d parameters vs %d, difference %d != 33" \
                          % (f, n_par, n_ref, n_par - n_ref)
    if not issubclass(ClamStyleGatedAttentionMIL, GatedAttentionMIL):
        return False, "the Phase-6C model does not inherit the Phase-6B model"
    return True, ("all 5 checkpoints carry exactly the Phase-6B tensors (512->64->32 "
                  "encoder, gated attention att_V/att_U/att_w, linear patient "
                  "classifier) plus Linear(32, 1) - 33 extra parameters, 37 123 vs "
                  "37 090 - and the bag-level forward pass is bit-identical to Phase 6B")


@check("C", "13 no CNN / ResNet parameter is inside the trained model")
def c13():
    bad = []
    for f in FOLDS:
        for k, v in checkpoint(f)["model_state_dict"].items():
            if v.dim() > 2:
                bad.append("%s fold %d dim %d" % (k, f, v.dim()))
            if any(t in k.lower() for t in ("conv", "bn", "batchnorm", "layer1", "layer2",
                                            "layer3", "layer4", "downsample", "resnet",
                                            "running_mean", "running_var")):
                bad.append("%s fold %d" % (k, f))
    if bad:
        return False, "ResNet-shaped or CNN-named tensors: %s" % bad[:5]
    tot = {f: sum(v.numel() for v in checkpoint(f)["model_state_dict"].values())
           for f in FOLDS}
    if set(tot.values()) != {37123}:
        return False, "parameter counts %s" % tot
    forbidden = scan(("resnet18", "resnet50", "torchvision", "conv2d", "conv3d",
                      "batchnorm2d", "adaptiveavgpool2d"))
    if forbidden:
        return False, "CNN constructs in executable code: %s" % forbidden[:5]
    return True, ("5 checkpoints, 37 123 parameters each, Linear layers only (no tensor "
                  "with more than 2 dimensions, no conv / norm / ResNet-named tensor); "
                  "no CNN construct anywhere in the Phase-6C training path")


@check("C", "14 no CNN fine-tuning occurred")
def c14():
    for pid in COHORT.patient_ids:
        with np.load(os.path.join(EMB_DIR, "%s.npz" % pid), allow_pickle=True) as z:
            if str(z["encoder_state_fingerprint"]) != \
               str(CFG["phase3"]["expected_encoder"]["state_fingerprint"]):
                return False, "%s: the encoder fingerprint changed" % pid
            if not bool(z["encoder_frozen"]) or str(z["encoder_head"]) != "fc -> Identity":
                return False, "%s: encoder no longer frozen / head changed" % pid
    mods = set()
    for rel in PHASE6C_PY:
        mods |= imported_modules(rel)
    if "torchvision" in mods:
        return False, "torchvision is imported by the Phase-6C training path"
    if RUN.get("torchvision_imported_during_training"):
        return False, "the run record shows torchvision imported during training"
    hits = scan(("load_state_dict", "requires_grad_", "get_model", "models", "weights"),
                files=("src/clam_mil.py", "src/train_mil_clam.py"))
    hits = [h for h in hits if h[1] in ("get_model",)]
    if hits:
        return False, "encoder construction in the Phase-6C path: %s" % hits
    return True, ("all 202 embedding files still carry fingerprint b6e132d3…, "
                  "encoder_frozen=True and head 'fc -> Identity'; torchvision is neither "
                  "imported by nor loaded during the Phase-6C training path, so no CNN "
                  "weight could have been updated")


@check("C", "15 no Transformer / self-attention-over-instances model exists")
def c15():
    hits = scan(("transformer", "transformerencoder", "transformerencoderlayer",
                 "transmil", "nystrom", "multiheadattention", "scaled_dot_product_attention",
                 "vit", "positional_encoding"))
    if hits:
        return False, "Transformer constructs: %s" % hits[:5]
    stray = []
    for d in ("src", "config", "tests", "tools"):
        for root_, _, fs in os.walk(os.path.join(ROOT, d)):
            for fn in fs:
                low = fn.lower()
                if any(t in low for t in ("transformer", "transmil", "nystrom")):
                    stray.append(os.path.join(root_, fn))
    if stray:
        return False, "Transformer source files exist: %s" % stray
    for f in FOLDS:
        keys = set(checkpoint(f)["model_state_dict"])
        if keys - {"encoder.net.0.weight", "encoder.net.0.bias", "encoder.net.3.weight",
                   "encoder.net.3.bias", "att_V.weight", "att_V.bias", "att_U.weight",
                   "att_U.bias", "att_w.weight", "att_w.bias", "classifier.weight",
                   "classifier.bias", "instance_classifier.weight",
                   "instance_classifier.bias"}:
            return False, "fold %d: unexpected tensors %s" % (f, sorted(keys))
    return True, ("no Transformer / TransMIL / Nystrom / multi-head-attention construct "
                  "in any Phase-6C module, no such source file anywhere, and the 5 "
                  "checkpoints hold only the 14 expected gated-attention tensors")


@check("C", "16 no radiomics + CNN fusion, and no radiomics extraction")
def c16():
    mods = set()
    for rel in PHASE6C_PY:
        mods |= imported_modules(rel)
    bad = mods & {"radiomics", "SimpleITK", "sitk", "nibabel", "pydicom", "featureextractor",
                  "PIL", "cv2", "skimage"}
    if bad:
        return False, "imaging/radiomics imports in the training path: %s" % sorted(bad)
    if RUN.get("radiomics_imported_during_training"):
        return False, "the run record shows radiomics/SimpleITK imported during training"
    hits = scan(("featureextractor", "radiomicsfeatureextractor", "patch_features",
                 "concatenate_features", "hstack_features"))
    if hits:
        return False, "radiomics/fusion constructs: %s" % hits[:5]
    # exactly one input directory, and every first layer is exactly its fold's width
    if str(CFG["phase3"]["features_dir"]) != "cnn_embeddings/gtv_rim_resnet18":
        return False, "unexpected features_dir %s" % CFG["phase3"]["features_dir"]
    names = [ln.strip() for ln in open(ppath(CFG, CFG["phase3"]["feature_names_file"]),
                                       "r", encoding="utf-8") if ln.strip()]
    hand = [n for n in names if any(t in n.lower() for t in
                                    ("firstorder", "glcm", "glrlm", "glszm"))]
    if hand:
        return False, "handcrafted feature names in the input space: %s" % hand[:3]
    for f in FOLDS:
        ck = checkpoint(f)
        w = ck["model_state_dict"]["encoder.net.0.weight"]
        if int(w.shape[1]) != len(ck["kept_features"]) or int(w.shape[1]) != N_DIMS:
            return False, "fold %d: first layer is %s, not [64, %d]" % (f, tuple(w.shape),
                                                                        N_DIMS)
    return True, ("one input directory (the frozen CNN embeddings), 512 embedding "
                  "dimension names with no handcrafted radiomic name among them, every "
                  "first layer exactly [64, 512], and no radiomics or fusion construct "
                  "in any Phase-6C module")


@check("C", "17 no TRUE patch-level ADC/SCC label is ever created")
def c17():
    # (a) no label-replication primitive anywhere in the Phase-6C training path
    hits = scan(("repeat", "repeat_interleave", "tile", "expand_as", "np_repeat"),
                files=("src/clam_mil.py", "src/train_mil_clam.py"))
    if hits:
        return False, "label-replication primitives: %s" % hits
    # (b) the instance targets are literal constants, never the patient label
    import ast as _ast
    tree = _ast.parse(source("src/clam_mil.py"))
    fn = next(n for n in _ast.walk(tree)
              if isinstance(n, _ast.FunctionDef) and n.name == "instance_loss_for_bag")
    if "label" in {a.arg for a in fn.args.args}:
        return False, "instance_loss_for_bag takes a label argument"
    body = _ast.dump(fn)
    if "'label'" in body or '"label"' in body:
        return False, "instance_loss_for_bag references a label"
    if not ("torch" in body and "ones" in body and "zeros" in body):
        return False, "the instance targets are not literal ones/zeros"
    # (c) every recorded class count is a PATIENT count
    for f in FOLDS:
        meta = scaler_meta(f)
        tr, _ = outer_sets(f)
        y = COHORT.labels(sorted(tr))
        n_patch = sum(COHORT.bags[p].n_patches for p in tr)
        for j in range(1, N_INNER + 1):
            info = meta["inner"][str(j)]
            if info["n_inner_train_adc"] + info["n_inner_train_scc"] == n_patch:
                return False, "fold %d inner %d records patch counts as class counts" % (f, j)
        if int((y == 1).sum()) + int((y == 0).sum()) != len(tr):
            return False, "fold %d: class counts are not patient counts" % f
    # (d) the per-bag record never carries a patch-level histology target
    pb = per_bag_df(1)
    forbidden = {"patch_label", "instance_label", "patch_histology", "true_label"}
    if forbidden & set(pb.columns):
        return False, "the per-bag record carries a patch-level label column"
    return True, ("no label-replication primitive in the Phase-6C training path; "
                  "instance_loss_for_bag takes no label and builds its targets from "
                  "literal torch.ones / torch.zeros; every recorded class count is a "
                  "PATIENT count; no patch-level label column exists anywhere")


@check("C", "30 no Phase-6D work and no second encoder has been started")
def c30():
    stray = []
    for d in ("src", "config", "tests", "tools", "notebooks", "reports", "results",
              "models", "predictions", "preprocessing", "attention", "splits"):
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for root_, _, fs in os.walk(base):
            for fn in fs:
                rel = os.path.relpath(os.path.join(root_, fn), ROOT).replace(os.sep, "/")
                segs = rel.lower().replace(".", "_").split("/")
                toks = {t for s in segs for t in s.split("_")}
                if {"phase6d", "phase7"} & toks:
                    stray.append(rel)
                if {"resnet50", "uni", "conch", "titan"} & toks:
                    stray.append(rel)
    if stray:
        return False, "Phase-6D / alternative-encoder artefacts exist: %s" % sorted(set(stray))[:5]
    hits = scan(("resnet50", "uni_v1", "conch_v1", "titan", "smote", "combat"))
    if hits:
        return False, "out-of-scope constructs: %s" % hits[:5]
    return True, ("no Phase-6D / Phase-7 file, no alternative-encoder artefact "
                  "(ResNet50 / UNI / CONCH / TITAN), and no SMOTE or ComBat construct "
                  "anywhere in the project namespaces")


# =================================================== D  instance supervision
@check("D", "18 the pseudo targets come ONLY from the current attention ranking")
def d18():
    import ast as _ast
    tree = _ast.parse(source("src/clam_mil.py"))
    sel = next(n for n in _ast.walk(tree)
               if isinstance(n, _ast.FunctionDef) and n.name == "select_pseudo_instances")
    d = _ast.dump(sel)
    if "argsort" not in d or "alpha" not in d:
        return False, "select_pseudo_instances does not rank the attention vector"
    if "label" in d:
        return False, "select_pseudo_instances references a label"
    # not precomputed: the selection is called from inside the per-bag training step
    step = _ast.dump(next(n for n in _ast.walk(tree) if isinstance(n, _ast.FunctionDef)
                          and n.name == "train_one_epoch_clam"))
    if "instance_loss_for_bag" not in step:
        return False, "the training step does not compute the instance loss per bag"
    cached = scan(("precompute", "cached_topk", "pickle", "joblib"),
                  files=("src/clam_mil.py", "src/train_mil_clam.py"))
    if cached:
        return False, "the selection looks precomputed: %s" % cached
    # empirical: the selected sets really are the extremes of the current attention
    fails = 0
    for f in FOLDS:
        ck = checkpoint(f)
        m = build_clam_model(CFG, input_dim=int(ck["input_dim"]))
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        tr, _ = outer_sets(f)
        scaler = fit_fold_scaler(CFG, COHORT, sorted(tr))
        for pid in sorted(tr)[:20]:
            x = torch.from_numpy(scaler.transform(COHORT.bags[pid].features))
            with torch.no_grad():
                _, alpha, _ = m.encode(x)
            k = bag_k(int(alpha.shape[0]), KRULE)
            top, bot = select_pseudo_instances(alpha, k)
            if float(alpha[top].min()) < float(alpha[bot].max()):
                fails += 1
    if fails:
        return False, "%d bags where a 'top' instance ranks below a 'bottom' one" % fails
    return True, ("the pseudo targets are built by a stable descending argsort of the "
                  "CURRENT forward pass's attention vector, inside the per-bag training "
                  "step (never precomputed, never cached), and reference no label; "
                  "verified empirically on 100 training bags: every top-k weight is "
                  ">= every bottom-k weight")


@check("D", "19 the adaptive-k formula is exactly the specified one")
def d19():
    rule = CFG["instance_supervision"]["k_rule"]
    if (float(rule["fraction"]), int(rule["max_k"]), int(rule["min_k"])) != (0.10, 8, 1):
        return False, "the configured rule is %s" % rule

    def want(n):
        k = min(8, max(1, int(math.floor(0.10 * n))))
        return int(math.floor(n / 2)) if 2 * k > n else k

    for n, expect in ((8, 1), (10, 1), (20, 2), (50, 5), (80, 8), (900, 8)):
        if adaptive_k(n) != expect or want(n) != expect:
            return False, "N=%d -> k=%d, expected %d" % (n, adaptive_k(n), expect)
    bad = [n for n in range(2, 1001) if adaptive_k(n) != want(n)]
    if bad:
        return False, "the implementation deviates at N = %s" % bad[:5]
    kdf = pd.read_csv(ppath(CFG, os.path.join(CFG["outputs"]["results_dir"],
                                              "adaptive_k.csv")))
    if len(kdf) != N_PATIENTS:
        return False, "adaptive_k.csv has %d rows" % len(kdf)
    wrong = kdf[kdf["adaptive_k"] != kdf["n_instances"].map(want)]
    if len(wrong):
        return False, "%d recorded k values disagree with the formula" % len(wrong)
    for f in FOLDS:
        pb = per_bag_df(f)
        w = pb[pb["k"] != pb["n_instances"].map(want)]
        if len(w):
            return False, "fold %d: %d training bags used a different k" % (f, len(w))
    dist = kdf["adaptive_k"].value_counts().sort_index().to_dict()
    return True, ("k = min(8, max(1, floor(0.10*N))) with the floor(N/2) fallback "
                  "verified for N = 2…1000 (999 values), at the six specified anchors, "
                  "for all 202 patients and for every training bag of all 5 folds; "
                  "k distribution over the cohort: %s" % dist)


@check("D", "20 2*k <= N for every training bag")
def d20():
    kdf = pd.read_csv(ppath(CFG, os.path.join(CFG["outputs"]["results_dir"],
                                              "adaptive_k.csv")))
    bad = kdf[2 * kdf["adaptive_k"] > kdf["n_instances"]]
    if len(bad):
        return False, "%d bags violate 2*k <= N" % len(bad)
    n_bags = 0
    for f in FOLDS:
        pb = per_bag_df(f)
        v = pb[2 * pb["k"] > pb["n_instances"]]
        if len(v):
            return False, "fold %d: %d training bags violate 2*k <= N" % (f, len(v))
        n_bags += len(pb)
        if not bool(fold_detail(f)["k_stats_outer_train"]["all_2k_le_n"]):
            return False, "fold %d: the run record reports a violation" % f
    smallest = int(kdf["n_instances"].min())
    return True, ("2*k <= N holds for all 202 bags (smallest bag N = %d, k = %d) and for "
                  "all %d training-bag records across the 5 folds"
                  % (smallest, int(kdf.loc[kdf['n_instances'].idxmin(), 'adaptive_k']),
                     n_bags))


@check("D", "21 the top-k and bottom-k sets never overlap")
def d21():
    for f in FOLDS:
        pb = per_bag_df(f)
        if int(pb["overlap"].abs().max()) != 0:
            return False, "fold %d: %d bags with overlapping sets" \
                          % (f, int((pb["overlap"] != 0).sum()))
    # re-derive it directly from every bag in the cohort, at the current model
    checked = 0
    for f in FOLDS:
        ck = checkpoint(f)
        m = build_clam_model(CFG, input_dim=int(ck["input_dim"]))
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        tr, _ = outer_sets(f)
        scaler = fit_fold_scaler(CFG, COHORT, sorted(tr))
        for pid in sorted(tr):
            x = torch.from_numpy(scaler.transform(COHORT.bags[pid].features))
            with torch.no_grad():
                _, alpha, _ = m.encode(x)
            k = bag_k(int(alpha.shape[0]), KRULE)
            top, bot = select_pseudo_instances(alpha, k)
            if set(top.tolist()) & set(bot.tolist()):
                return False, "fold %d %s: overlapping top/bottom sets" % (f, pid)
            checked += 1
    return True, ("every recorded training bag reports overlap = 0, and re-deriving the "
                  "two index sets from the trained models over all %d training-bag "
                  "instances yields no intersection anywhere" % checked)


@check("D", "22 each bag contributes equally many positive and negative pseudo instances")
def d22():
    for f in FOLDS:
        pb = per_bag_df(f)
        if not (pb["n_pseudo_instances"] == 2 * pb["k"]).all():
            return False, "fold %d: n_pseudo_instances != 2k" % f
        if not np.allclose(pb["fraction_participating"],
                           pb["n_pseudo_instances"] / pb["n_instances"], atol=1e-9):
            return False, "fold %d: the participating fraction is inconsistent" % f
    # the targets themselves: k ones then k zeros, by construction
    import ast as _ast
    fn = next(n for n in _ast.walk(_ast.parse(source("src/clam_mil.py")))
              if isinstance(n, _ast.FunctionDef) and n.name == "instance_loss_for_bag")
    d = _ast.dump(fn)
    if not (d.count("'ones'") >= 1 and d.count("'zeros'") >= 1):
        return False, "the targets are not one ones-block and one zeros-block"
    tot = sum(int(per_bag_df(f)["n_pseudo_instances"].sum()) for f in FOLDS)
    return True, ("every training bag supplies exactly k pseudo-positive and k "
                  "pseudo-negative instances (targets are torch.ones(k) followed by "
                  "torch.zeros(k)), so the auxiliary task is balanced inside each bag; "
                  "%d pseudo-instances in total across the 5 final refits, exactly half "
                  "of each class" % tot)


@check("D", "23 no outer-test instance influences training in any way")
def d23():
    for f in FOLDS:
        tr, te = outer_sets(f)
        pb = per_bag_df(f)
        if set(pb["PatientID"]) & te:
            return False, "fold %d: an outer-test bag received instance supervision" % f
        if set(pb["PatientID"]) != tr:
            missing = tr - set(pb["PatientID"])
            return False, "fold %d: %d outer-train bags missing from the record" \
                          % (f, len(missing))
        n_rec = int(fold_detail(f)["refit_last_epoch"]["n_bags_with_instance_loss"])
        if n_rec != len(tr):
            return False, "fold %d: %d bags supervised, %d outer-train" % (f, n_rec, len(tr))
    # the saved outer-test attention files declare no instance loss was applied
    for fn in sorted(os.listdir(att_dir())):
        with np.load(os.path.join(att_dir(), fn), allow_pickle=True) as z:
            if bool(z["instance_loss_applied_to_this_bag"]):
                return False, "%s declares an instance loss on an outer-test bag" % fn
    # and the code only ever iterates split.train
    import ast as _ast
    fn_ast = next(n for n in _ast.walk(_ast.parse(source("src/train_mil_clam.py")))
                  if isinstance(n, _ast.FunctionDef) and n.name == "run_training")
    d = _ast.dump(fn_ast)
    if "'evaluate'" in d and "collect" in d.split("'evaluate'")[0][-200:]:
        return False, "run_training may collect over the evaluation split"
    return True, ("across all 5 folds the instance-supervision record covers exactly the "
                  "outer-training bags and no outer-test bag; all 202 saved outer-test "
                  "attention files declare instance_loss_applied_to_this_bag = False; "
                  "inference runs under torch.no_grad() with no optimiser step")


# ============================================ E  numerical sanity / attention
@check("E", "24 all attention weights are finite and positive")
def e24():
    lo, hi, n = np.inf, -np.inf, 0
    for fn in sorted(os.listdir(att_dir())):
        with np.load(os.path.join(att_dir(), fn), allow_pickle=True) as z:
            a = np.asarray(z["attention"], dtype=np.float64)
            b = np.asarray(z["attention_weights"], dtype=np.float64)
            if not np.array_equal(a, b):
                return False, "%s: the attention_weights alias differs" % fn
            if not np.all(np.isfinite(a)):
                return False, "%s: non-finite attention" % fn
            if not np.all(a > 0):
                return False, "%s: non-positive attention" % fn
            lo, hi, n = min(lo, a.min()), max(hi, a.max()), n + a.size
    if n != N_INSTANCES:
        return False, "%d attention weights, expected %d" % (n, N_INSTANCES)
    return True, ("%d attention weights over 202 outer-test bags, all finite and > 0, "
                  "range [%.3e, %.4f]; the attention_weights alias is identical"
                  % (n, lo, hi))


@check("E", "25 attention sums to 1 within every bag")
def e25():
    worst = 0.0
    for fn in sorted(os.listdir(att_dir())):
        with np.load(os.path.join(att_dir(), fn), allow_pickle=True) as z:
            s = float(np.asarray(z["attention"], dtype=np.float64).sum())
            worst = max(worst, abs(s - 1.0))
    if worst > 1e-5:
        return False, "max |sum - 1| = %.3e" % worst
    return True, "max |sum - 1| over the 202 bags = %.3e" % worst


@check("E", "26 no NaN or Inf anywhere in the Phase-6C outputs")
def e26():
    df = preds6c()
    if not np.all(np.isfinite(df["predicted_probability_ADC"].values)):
        return False, "non-finite probabilities"
    n = len(df)
    for f in FOLDS:
        s = surface_df(f)
        if not np.all(np.isfinite(s["mean_inner_cv_roc_auc"].values)):
            return False, "fold %d: non-finite selection surface" % f
        n += len(s)
        r = pd.read_csv(os.path.join(sel_dir(), "fold_%d_refit_epochs.csv" % f))
        for c in ("bag_loss", "instance_loss", "total_loss"):
            if not np.all(np.isfinite(r[c].values)):
                return False, "fold %d: non-finite %s during the refit" % (f, c)
        n += len(r) * 3
        for j in [None] + list(range(1, N_INNER + 1)):
            z = scaler_npz(f, j)
            if not (np.all(np.isfinite(z["mean"])) and np.all(np.isfinite(z["std"]))
                    and np.all(z["std"] > 0)):
                return False, "fold %d scaler %s: non-finite or zero std" % (f, j)
            n += z["mean"].size + z["std"].size
    y = df.sort_values("PatientID")["true_label"].values
    p = df.sort_values("PatientID")["predicted_probability_ADC"].values
    m = patient_metrics(y, p, THR)
    rec = json.load(open(ppath(CFG, os.path.join(CFG["outputs"]["results_dir"],
                                                 "oof_metrics.json")),
                         "r", encoding="utf-8"))["pooled"]["phase6c"]
    for k in ("roc_auc", "pr_auc", "balanced_accuracy", "mcc"):
        if abs(m[k] - float(rec[k])) > 1e-12:
            return False, "%s does not recompute (%.12f vs %.12f)" % (k, m[k], rec[k])
    return True, ("%d stored values across predictions, 5 selection surfaces, 5 refit "
                  "loss curves and 20 scalers are all finite; every pooled metric "
                  "recomputes from the predictions to 1e-12" % n)


@check("E", "33 the attention files describe exactly the frozen instances")
def e33():
    for pid in COHORT.patient_ids:
        with np.load(os.path.join(att_dir(), "%s.npz" % pid), allow_pickle=True) as z:
            bag = COHORT.bags[pid]
            if int(z["attention"].shape[0]) != bag.n_patches:
                return False, "%s: %d weights for %d instances" % (
                    pid, int(z["attention"].shape[0]), bag.n_patches)
            if not np.array_equal(np.asarray(z["patch_index"]),
                                  np.arange(bag.n_patches, dtype=np.int32)):
                return False, "%s: patch_index is not 0…N-1" % pid
            if not np.array_equal(np.asarray(z["coords_index"]), bag.coords_index):
                return False, "%s: coords_index differs from the embedding file" % pid
            if not np.array_equal(np.asarray(z["coords_world"]), bag.coords_world):
                return False, "%s: coords_world differs from the embedding file" % pid
            if str(z["extraction_signature"]) != \
               str(CFG["phase3"]["expected_extraction_signature"]):
                return False, "%s: extraction signature differs" % pid
            if int(z["outer_fold"]) not in FOLDS:
                return False, "%s: bad outer fold" % pid
            if pid not in outer_sets(int(z["outer_fold"]))[1]:
                return False, "%s: attention saved by a fold it was not held out of" % pid
    return True, ("202 attention files carry exactly the frozen instance count, "
                  "patch_index 0…N-1, byte-identical coords_index and coords_world, the "
                  "GTV+Rim extraction signature 9825ed13538ee681, and were each written "
                  "by the fold that held that patient out")


# ============================================================= F  bootstraps
@check("F", "27 both paired bootstraps resample PATIENTS only, and reproduce")
def f27():
    comp = CFG["comparison"]
    n_patch = N_INSTANCES
    done = []
    for key, spec in (("post_hoc", comp["post_hoc_phase6b_pooling"]),
                      ("primary", comp["primary"])):
        rec = json.load(open(os.path.join(ROOT, *str(spec["output"]).split("/")),
                             "r", encoding="utf-8"))
        bs = rec["bootstrap"]
        if bs["resampling_unit"] != "patient" or not bs["paired"]:
            return False, "%s: not a paired patient-level resample" % key
        if int(bs["n_patients"]) != N_PATIENTS or int(bs["n_patients"]) == n_patch:
            return False, "%s: the unit looks like patches, not patients" % key
        if int(bs["n_resamples_requested"]) != int(comp["bootstrap"]["n_resamples"]):
            return False, "%s: %d resamples" % (key, bs["n_resamples_requested"])
        if key == "post_hoc":
            new = preds(str(spec["new_predictions"])).set_index("PatientID").sort_index()
            if not (rec.get("post_hoc") and rec.get("analysis") == "POST-HOC"
                    and rec.get("pre_specified") is False):
                return False, "the pooling comparison is not labelled POST-HOC"
        else:
            new = preds6c().set_index("PatientID").sort_index()
        ref = preds(str(spec["reference_predictions"])).set_index("PatientID").sort_index()
        pids = sorted(set(new.index) & set(ref.index))
        if len(pids) != N_PATIENTS:
            return False, "%s: %d paired patients" % (key, len(pids))
        y = new.loc[pids, "true_label"].values.astype(int)
        if not np.array_equal(y, ref.loc[pids, "true_label"].values.astype(int)):
            return False, "%s: labels disagree across the pair" % key
        redo = paired_bootstrap(
            y, new.loc[pids, "predicted_probability_ADC"].values.astype(float),
            ref.loc[pids, "predicted_probability_ADC"].values.astype(float),
            pids, int(comp["bootstrap"]["n_resamples"]), int(spec["seed"]),
            float(comp["bootstrap"]["ci"]))
        for k in ("roc_auc", "pr_auc"):
            for fld in ("difference_mean", "ci_low", "ci_high"):
                if abs(redo["metrics"][k][fld] - float(bs["metrics"][k][fld])) > 1e-12:
                    return False, "%s: %s %s not reproducible from seed %d" \
                                  % (key, k, fld, spec["seed"])
            if bool(redo["metrics"][k]["ci_includes_zero"]) != \
                    bool(bs["metrics"][k]["ci_includes_zero"]):
                return False, "%s: %s CI verdict differs" % (key, k)
        done.append(key)
    return True, ("both paired bootstraps (%s) reproduce every CI bound to 1e-12 from "
                  "their recorded seeds; the unit is the patient (%d), never the patch "
                  "(%d); the pooling comparison is explicitly labelled POST-HOC"
                  % ("/".join(done), N_PATIENTS, n_patch))


@check("F", "31 the small bags are reported separately and never excluded")
def f31():
    limit = int(CFG["evaluation"]["small_bag_threshold"])
    small = sorted(p for p in COHORT.patient_ids if COHORT.bags[p].n_patches < limit)
    df = pd.read_csv(ppath(CFG, os.path.join(CFG["outputs"]["results_dir"],
                                             "small_bag_predictions.csv")))
    if sorted(df["PatientID"]) != small:
        return False, "the small-bag table lists %d of %d bags" % (len(df), len(small))
    if len(small) != 11:
        return False, "%d bags below %d instances, expected 11" % (len(small), limit)
    # they must still be present in training and in the OOF predictions
    p = set(preds6c()["PatientID"])
    if set(small) - p:
        return False, "small bags missing from the OOF predictions"
    for f in FOLDS:
        tr, _ = outer_sets(f)
        pb = set(per_bag_df(f)["PatientID"])
        for s in small:
            if s in tr and s not in pb:
                return False, "fold %d: small bag %s was skipped during training" % (f, s)
    if not bool(df["two_k_le_n"].all()) or not bool(df["top_bottom_disjoint"].all()):
        return False, "a small bag violates 2*k <= N or has overlapping sets"
    return True, ("all 11 bags with fewer than %d instances are reported with their "
                  "histology, N, adaptive k, Phase-6B and Phase-6C probabilities and "
                  "correctness; none was excluded from training or from the 202 OOF "
                  "predictions; 2*k <= N and disjointness hold for every one" % limit)


# ========================== G  frozen artefacts and read-only external trees
@check("G", "28 every Phase-1..6B artefact is unchanged")
def g28():
    snap = snapshot()
    changed, missing = [], []
    for rel in sorted(snap):
        if rel in AUDIT_OUTPUT_EXCEPTIONS:
            continue
        p = os.path.join(ROOT, *rel.split("/"))
        if not os.path.isfile(p):
            missing.append(rel)
        elif sha256(p) != snap[rel]:
            changed.append(rel)
    sem = []
    for rel, n in (("reports/phase6a_tests.json", 27), ("reports/phase6b_tests.json", 32)):
        ok, msg = earlier_audit_ok(rel, n)
        if not ok:
            return False, "the %s audit no longer passes: %s" % (rel, msg)
        sem.append(msg)
    if changed or missing:
        return False, "%d changed, %d missing (e.g. %s)" % (len(changed), len(missing),
                                                            (changed + missing)[:3])
    return True, ("%d of %d pre-Phase-6C files byte-identical (0 changed, 0 missing); the "
                  "%d audit OUTPUTS excluded from byte comparison are verified "
                  "semantically instead: %s"
                  % (len(snap) - len(AUDIT_OUTPUT_EXCEPTIONS), len(snap),
                     len(AUDIT_OUTPUT_EXCEPTIONS), "; ".join(sem)))


@check("G", "29 Dataset/ and Vuong5ROI/ remain read-only, and nothing else moved")
def g29():
    from config_io import DEFAULT_CONFIG, load_config as lc
    from readonly_guard import verify
    ok, info = verify(lc(DEFAULT_CONFIG))
    snap = snapshot()
    protected = ("patch_features/", "patient_summaries/", "feature_selection/",
                 "image_patches/", "cnn_embeddings/", "attention/gtv_rim/",
                 "attention/gradient", "attention/gtv_rim_resnet18_phase6b",
                 "models/gtv_rim_", "models/phase5c_", "models/phase6b_cnn_",
                 "results/phase5", "results/phase6b",
                 "predictions/gtv_rim_", "predictions/phase5c_", "predictions/phase6b_",
                 "preprocessing/gtv_rim", "preprocessing/phase5c", "preprocessing/phase6b",
                 "splits/gtv_rim_inner_cv", "splits/phase5c_inner_cv",
                 "splits/phase6b_inner_cv")
    intruders = []
    for pre in protected:
        d = os.path.join(ROOT, *pre.rstrip("/").split("/"))
        base = d if os.path.isdir(d) else os.path.dirname(d)
        if not os.path.isdir(base):
            continue
        for root_, _, fs in os.walk(base):
            for f in fs:
                rel = os.path.relpath(os.path.join(root_, f), ROOT).replace(os.sep, "/")
                if rel.startswith(pre) and rel not in snap:
                    intruders.append(rel)
    ok = ok and not intruders
    return ok, ("Dataset/ + Vuong5ROI/: %d baseline files, %d now, +%d -%d ~%d - guarantee "
                "%s; %d new files inside a protected Phase-1..6B namespace"
                % (info["n_baseline"], info["n_now"], info["n_added"], info["n_removed"],
                   info["n_changed"], "INTACT" if ok else "BROKEN", len(set(intruders))))


# ================================== H  deliverables and reported preprocessing
@check("H", "32 the near-zero-variance screen retained 512/512 in all 20 scalers")
def h32():
    kept, dropped = set(), set()
    rows = []
    for f in FOLDS:
        for j in [None] + list(range(1, N_INNER + 1)):
            z = scaler_npz(f, j)
            k = int(len(z["kept_features"]))
            d = int(len(z["dropped_features"]))
            kept.add(k)
            dropped.add(d)
            rows.append({"fold": f, "scaler": "final" if j is None else "inner_%d" % j,
                         "dimensions_retained": k, "dimensions_dropped": d})
    rec = pd.read_csv(ppath(CFG, os.path.join(CFG["outputs"]["results_dir"],
                                              "dimension_retention.csv")))
    if len(rec) != len(rows):
        return False, "dimension_retention.csv has %d rows, expected %d" % (len(rec),
                                                                            len(rows))
    got = rec.sort_values(["fold", "scaler"])[["dimensions_retained",
                                               "dimensions_dropped"]].values
    want = pd.DataFrame(rows).sort_values(["fold", "scaler"])[
        ["dimensions_retained", "dimensions_dropped"]].values
    if not np.array_equal(got, want):
        return False, "the recorded retention table disagrees with the scalers on disk"
    v = CFG["preprocessing"]["variance"]
    if (float(v["absolute_std_threshold"]), float(v["relative_std_threshold"])) != \
            (1e-8, 1e-8):
        return False, "the variance thresholds were changed: %s" % v
    if kept != {N_DIMS} or dropped != {0}:
        return False, "retained %s, dropped %s" % (sorted(kept), sorted(dropped))
    return True, ("all 5 final and all 15 inner scalers retain 512 of 512 dimensions and "
                  "drop 0, under the unchanged Phase-5/6B thresholds (1e-8 / 1e-8); "
                  "results/phase6c/dimension_retention.csv agrees row for row")


@check("H", "34 every declared Phase-6C deliverable exists")
def h34():
    res = CFG["outputs"]["results_dir"]
    want = [
        "config/phase6c.yaml",
        "predictions/%s_oof.csv" % model_tag(CFG, MODEL),
        "%s/fold_metrics.csv" % res, "%s/oof_metrics.json" % res,
        "%s/phase6b_pooling_bootstrap.json" % res,
        "%s/phase6c_vs_phase6b_bootstrap.json" % res,
        "%s/instance_diagnostics.csv" % res, "%s/attention_concentration.csv" % res,
        "%s/small_bag_predictions.csv" % res, "%s/adaptive_k.csv" % res,
        "%s/dimension_retention.csv" % res,
        "reports/phase6c_run.json", "reports/phase6c_pre_snapshot.json",
        "src/clam_mil.py", "src/train_mil_clam.py", "src/evaluate_phase6c.py",
        "src/phase6c_pooling_bootstrap.py", "tools/phase6c_baseline.py",
    ]
    dirs = ["%s/lambda_epoch_selection" % res, "%s/roc_curves" % res,
            "%s/pr_curves" % res, "%s/confusion_matrices" % res,
            CFG["outputs"]["attention_dir"], CFG["outputs"]["inner_splits_dir"],
            CFG["outputs"]["preprocessing_dir"],
            "models/%s" % model_tag(CFG, MODEL)]
    missing = [w for w in want if not os.path.isfile(os.path.join(ROOT, *w.split("/")))]
    missing += [d for d in dirs if not os.path.isdir(os.path.join(ROOT, *d.split("/")))]
    empty = [d for d in dirs if os.path.isdir(os.path.join(ROOT, *d.split("/")))
             and not os.listdir(os.path.join(ROOT, *d.split("/")))]
    if missing or empty:
        return False, "missing %s; empty %s" % (missing, empty)
    for f in FOLDS:
        for stem in ("fold_%d_surface.csv", "fold_%d_refit_epochs.csv",
                     "fold_%d_refit_per_bag.csv"):
            if not os.path.isfile(os.path.join(sel_dir(), stem % f)):
                return False, "missing %s" % (stem % f)
        for lm in GRID:
            t = ("%.4f" % lm).rstrip("0").rstrip(".").replace(".", "p")
            if not os.path.isfile(os.path.join(sel_dir(), "fold_%d_lambda_%s.csv" % (f, t))):
                return False, "missing fold_%d_lambda_%s.csv" % (f, t)
    return True, ("%d declared files and %d directories exist and are non-empty, "
                  "including all 5 selection surfaces, 10 per-lambda inner curves, 5 "
                  "refit loss curves and 5 per-bag instance records"
                  % (len(want), len(dirs)))


def main() -> int:
    results, n_fail = [], 0
    width = max(len(n) for _, n, _ in CHECKS)
    last = None
    for i, (group, name, fn) in enumerate(CHECKS, 1):
        if group != last:
            print("\n--- group %s ---" % group)
            last = group
        try:
            ok, msg = fn()
        except Exception as exc:                        # noqa: BLE001
            ok, msg = False, "EXCEPTION: %s" % str(exc)[:400]
        n_fail += 0 if ok else 1
        print("%-4s %s  %-*s  %s" % ("PASS" if ok else "FAIL", group, width, name, msg))
        results.append({"n": i, "group": group, "name": name, "pass": bool(ok),
                        "message": msg})

    out = {"n_checks": len(CHECKS), "n_passed": len(CHECKS) - n_fail, "n_failed": n_fail,
           "phase": "6c", "roi": "gtv_rim", "representation": "resnet18_imagenet_512d",
           "model": MODEL, "folds": list(FOLDS), "n_inner_splits": N_INNER,
           "lambda_grid": GRID, "k_rule": CFG["instance_supervision"]["k_rule"],
           "threshold": THR, "n_patients": N_PATIENTS, "n_instances": N_INSTANCES,
           "n_embedding_dimensions": N_DIMS, "checks": results}
    path = ppath(CFG, "reports/phase6c_tests.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\n%d checks, %d passed, %d failed  ->  %s"
          % (len(CHECKS), len(CHECKS) - n_fail, n_fail, path))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
