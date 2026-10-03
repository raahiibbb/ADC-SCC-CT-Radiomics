"""Phase-5C leakage and sanity checks: feature-selection rescue test for GTV+Rim MIL.

Run with:
    ./.venv/Scripts/python.exe tests/test_phase5c.py

Every check re-derives its answer from files on disk.  Nothing is trusted from
the training run: selectors are refitted from the frozen bags on exactly the
recorded patient lists, scalers are refitted, predictions are reproduced from
the saved checkpoints, and the bootstrap is replayed from its recorded seed.

Groups
  A  patient-level summaries - the MIL rule            (required checks 1-5)
  B  selector leakage inside the nested CV             (required checks 6-9, 11, 12)
  C  preprocessing, class weighting, index mapping     (required checks 10, 13-15)
  D  predictions and numerical sanity                  (required checks 16, 17)
  E  frozen inputs and the untouched Phase-5 baseline  (required checks 18-20)
  F  the paired patient bootstrap                      (required checks 21, 22)
  G  scope: no radiomics, no Phase 6                   (required checks 23, 24)
"""
from __future__ import annotations

import ast
import json
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from feature_selection import (absolute_pearson_matrix, anova_f_statistic,  # noqa: E402
                               build_patient_summaries, fit_selected_patch_scaler,
                               jaccard, lasso_select, load_patient_summaries,
                               mrmr_rank, patient_summary)
from mil_data import (load_cohort, load_outer_split, load_phase3_config,  # noqa: E402
                      outer_fold_patients, pos_weight_from_patients, ppath,
                      sha256_file)
from mil_metrics import patient_metrics  # noqa: E402
from mil_models import build_model  # noqa: E402
from roi_paired_compare import bootstrap_differences  # noqa: E402
from train_mil_fs import (choose_setting_and_epoch, fit_selector_inner,  # noqa: E402
                          inner_cv_assignment, unit_path)

CFG = load_phase3_config(os.path.join(ROOT, "config", "phase5c.yaml"))
FOLDS = (1, 2, 3, 4, 5)
TAGS = [p["tag"] for p in CFG["pipelines"]]
BASE_OF = {p["tag"]: p["baseline"] for p in CFG["pipelines"]}
SELECTOR_OF = {p["tag"]: p["selector"] for p in CFG["pipelines"]}
MODEL_OF = {p["tag"]: p["model"] for p in CFG["pipelines"]}
N_PATIENTS = int(CFG["phase3"]["expected_patients"])
N_FEATURES = int(CFG["phase3"]["expected_features"])
N_INNER = int(CFG["inner_cv"]["n_splits"])
THR = float(CFG["evaluation"]["threshold"])
CHECKS = []

COHORT = load_cohort(CFG)
SPLIT = load_outer_split(CFG)
SUMMARIES = load_patient_summaries(CFG, ppath)
NAMES = list(COHORT.feature_names)

PHASE5C_SOURCES = ["src/feature_selection.py", "src/train_mil_fs.py",
                   "src/evaluate_mil_fs.py", "config/phase5c.yaml",
                   "tools/phase5c_baseline.py", "tests/test_phase5c.py"]
# The forbidden-token scans below assert what the Phase-5C PIPELINE does.  This
# audit file is not part of the pipeline - it trains nothing and writes no
# artefact - and it necessarily contains every forbidden token as a literal in
# its own search lists, so scanning it would report guaranteed self-hits.
PHASE5C_PY = [p for p in PHASE5C_SOURCES
              if p.endswith(".py") and not p.startswith("tests/")]


def check(group, name):
    def deco(fn):
        CHECKS.append((group, name, fn))
        return fn
    return deco


def unit(tag, fold):
    with open(unit_path(CFG, tag, fold), "r", encoding="utf-8") as fh:
        return json.load(fh)


def fs_dir(selector):
    return os.path.join(ppath(CFG, CFG["outputs"]["feature_selection_dir"]), selector)


def inner_record(selector, fold):
    with open(os.path.join(fs_dir(selector), "fold_%d_inner.json" % fold),
              "r", encoding="utf-8") as fh:
        return json.load(fh)


def final_record(selector, tag, fold):
    with open(os.path.join(fs_dir(selector), "fold_%d_%s_final.json" % (fold, tag)),
              "r", encoding="utf-8") as fh:
        return json.load(fh)


def preds(tag):
    return pd.read_csv(os.path.join(ppath(CFG, CFG["outputs"]["predictions_dir"]),
                                    "%s%s_oof.csv" % (CFG["outputs"]["model_prefix"], tag)))


def baseline_preds(which):
    return pd.read_csv(ppath(CFG, CFG["baseline"]["model_files"][which]))


def scaler_npz(tag, fold):
    return np.load(os.path.join(ppath(CFG, CFG["outputs"]["preprocessing_dir"]),
                                "%s_fold_%d_final_scaler.npz" % (tag, fold)),
                   allow_pickle=True)


def scaler_meta(tag, fold):
    with open(os.path.join(ppath(CFG, CFG["outputs"]["preprocessing_dir"]),
                           "%s_fold_%d_scaler.json" % (tag, fold)),
              "r", encoding="utf-8") as fh:
        return json.load(fh)


def checkpoint(tag, fold):
    return torch.load(os.path.join(ppath(CFG, CFG["outputs"]["models_dir"]),
                                   "%s%s" % (CFG["outputs"]["model_prefix"], tag),
                                   "fold_%d.pt" % fold),
                      map_location="cpu", weights_only=False)


def outer_sets(fold):
    tr, te = outer_fold_patients(SPLIT, fold)
    return set(tr), set(te)


def pre_snapshot():
    with open(ppath(CFG, "reports/phase5c_pre_snapshot.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)["files"]


# ============================================ A  patient summaries - the MIL rule
@check("A", "1 exactly 202 patient summaries, one row per patient")
def a1():
    n = len(SUMMARIES.patient_ids)
    if n != N_PATIENTS:
        return False, "%d summary rows, expected %d" % (n, N_PATIENTS)
    if len(set(SUMMARIES.patient_ids)) != n:
        return False, "duplicate PatientIDs among the summaries"
    if sorted(SUMMARIES.patient_ids) != sorted(COHORT.patient_ids):
        return False, "the summary patient set differs from the frozen bags"
    if SUMMARIES.matrix.shape[0] != n:
        return False, "matrix has %d rows for %d patients" % (SUMMARIES.matrix.shape[0], n)
    csv = pd.read_csv(ppath(CFG, CFG["patient_summary"]["csv"]))
    if len(csv) != n or csv["PatientID"].duplicated().any():
        return False, "summaries.csv has %d rows / duplicates" % len(csv)
    n_adc = int((SUMMARIES.labels == 1).sum())
    return True, ("%d rows, %d unique patients, %d ADC / %d SCC; summaries.csv agrees"
                  % (n, n, n_adc, n - n_adc))


@check("A", "2 bag size never changes a patient's contribution (900 vs 8 patches)")
def a2():
    npatch = SUMMARIES.n_patches
    big = SUMMARIES.patient_ids[int(np.argmax(npatch))]
    small = SUMMARIES.patient_ids[int(np.argmin(npatch))]
    for pid in (big, small):
        if SUMMARIES.patient_ids.count(pid) != 1:
            return False, "%s appears more than once" % pid
    x, y = SUMMARIES.subset([big, small])
    if x.shape != (2, N_FEATURES) or y.shape != (2,):
        return False, "a 2-patient selector subset is %r, not (2, 74)" % (x.shape,)
    # every fitting set recorded on disk holds each patient exactly once
    for sel in ("mrmr", "lasso"):
        for f in FOLDS:
            rec = inner_record(sel, f)
            for j in map(str, range(1, N_INNER + 1)):
                fp = rec["inner"][j]["fit_patients"]
                if len(fp) != len(set(fp)):
                    return False, "%s fold %d inner %s repeats a patient" % (sel, f, j)
                if rec["inner"][j]["n_fit_patients"] != len(fp):
                    return False, "%s fold %d inner %s: n_fit_patients != len(fit_patients)"
                patches = sum(COHORT.bags[p].n_patches for p in fp)
                if rec["inner"][j]["n_fit_patients"] == patches:
                    return False, "%s fold %d inner %s fitted on patch rows" % (sel, f, j)
    return True, ("%s (%d patches) and %s (%d patches) each contribute exactly ONE row; "
                  "every recorded fitting set counts patients, never patches"
                  % (big, int(npatch.max()), small, int(npatch.min())))


@check("A", "3 the summary dimension is 74 before any selection")
def a3():
    if SUMMARIES.matrix.shape[1] != N_FEATURES:
        return False, "summary dimension %d" % SUMMARIES.matrix.shape[1]
    if SUMMARIES.feature_names != NAMES:
        return False, "summary feature names differ from config/feature_names.txt"
    for sel in ("mrmr", "lasso"):
        rec = inner_record(sel, 1)["inner"]["1"]
        if sel == "mrmr" and len(rec["relevance_f"]) != N_FEATURES:
            return False, "mRMR scored %d features" % len(rec["relevance_f"])
        if sel == "lasso":
            k = list(rec["candidates"])[0]
            if len(rec["candidates"][k]["summary_scaler_mean"]) != N_FEATURES:
                return False, "the LASSO summary scaler covers %d features" % len(
                    rec["candidates"][k]["summary_scaler_mean"])
    return True, ("74-D summaries, names identical to config/feature_names.txt; both "
                  "selectors are offered all 74 original dimensions")


@check("A", "4 patient summary == median of that patient's frozen patch features")
def a4():
    worst, worst_pid = 0.0, None
    for i, pid in enumerate(SUMMARIES.patient_ids):
        want = np.median(COHORT.bags[pid].features.astype(np.float64), axis=0)
        d = float(np.max(np.abs(want - SUMMARIES.matrix[i])))
        if d > worst:
            worst, worst_pid = d, pid
    if worst > 0.0:
        return False, "%s differs by %.3g from the median of its bag" % (worst_pid, worst)
    rebuilt = build_patient_summaries(COHORT, "median")
    if rebuilt.fingerprint() != SUMMARIES.fingerprint():
        return False, "rebuilding the summaries from the frozen bags gives a different matrix"
    meta = json.load(open(ppath(CFG, CFG["patient_summary"]["json"]),
                          "r", encoding="utf-8"))
    if meta["uses_labels"] or meta["uses_other_patients"] or meta["scaled"]:
        return False, "the provenance record claims the summary uses labels/other patients"
    return True, ("all %d summaries reproduce EXACTLY (max |diff| 0) from the frozen bags; "
                  "fingerprint %s" % (N_PATIENTS, SUMMARIES.fingerprint()))


@check("A", "5 no patch-level ADC/SCC selector fitting anywhere")
def a5():
    # structural: the supervised selectors only ever see a [n_patients, 74] matrix
    tr, _ = outer_fold_patients(SPLIT, 1)
    x, y = SUMMARIES.subset(sorted(tr))
    if x.shape[0] != len(tr) or y.shape[0] != len(tr):
        return False, "a selector subset returned %d rows for %d patients" % (x.shape[0],
                                                                              len(tr))
    patch_rows = sum(COHORT.bags[p].n_patches for p in tr)
    if x.shape[0] == patch_rows:
        return False, "the selector matrix has one row per patch"
    # source: no Phase-5C module ever repeats a label across a bag
    forbidden = ("np.repeat", "numpy.repeat", "np.tile", "numpy.tile")
    hits = []
    for rel in PHASE5C_PY:
        src = open(os.path.join(ROOT, rel), "r", encoding="utf-8").read()
        hits += ["%s:%s" % (rel, f) for f in forbidden if f in src]
    if hits:
        return False, "label-replication primitive found: %s" % hits
    # every stored selector record: rows == patients, and the class counts are patients
    for sel in ("mrmr", "lasso"):
        for f in FOLDS:
            rec = inner_record(sel, f)
            for j in map(str, range(1, N_INNER + 1)):
                r = rec["inner"][j]
                if r["n_fit_adc"] + r["n_fit_scc"] != r["n_fit_patients"]:
                    return False, "%s fold %d inner %s class counts are not patients" % (
                        sel, f, j)
                want_adc = int(sum(COHORT.bags[p].label == 1 for p in r["fit_patients"]))
                if r["n_fit_adc"] != want_adc:
                    return False, "%s fold %d inner %s ADC count %d != %d patients" % (
                        sel, f, j, r["n_fit_adc"], want_adc)
    return True, ("selector input is [n_patients, 74] (%d rows, not %d patch rows); no "
                  "label-replication primitive in any Phase-5C module; every recorded "
                  "class count is a PATIENT count" % (len(tr), patch_rows))


# ================================== B  selector leakage inside the nested CV
@check("B", "6 no outer-test patient participates in any selection step")
def b6():
    for f in FOLDS:
        train, test = outer_sets(f)
        for sel in ("mrmr", "lasso"):
            rec = inner_record(sel, f)
            if set(rec["outer_test_patients"]) != test:
                return False, "%s fold %d records the wrong outer-test set" % (sel, f)
            for j in map(str, range(1, N_INNER + 1)):
                fp = set(rec["inner"][j]["fit_patients"])
                iv = set(rec["inner"][j]["inner_val_patients"])
                if fp & test or iv & test:
                    return False, "%s fold %d inner %s touches outer-test patients" % (
                        sel, f, j)
                if not (fp | iv) <= train:
                    return False, "%s fold %d inner %s leaves the outer-train set" % (
                        sel, f, j)
                if fp & iv:
                    return False, "%s fold %d inner %s overlaps its own validation set" % (
                        sel, f, j)
        for t in TAGS:
            fr = final_record(SELECTOR_OF[t], t, f)
            if set(fr["fit_patients"]) & test:
                return False, "%s fold %d final selector saw an outer-test patient" % (t, f)
            u = unit(t, f)
            if set(u["outer_test_patients"]) != test:
                return False, "%s fold %d evaluated the wrong outer-test set" % (t, f)
            sm = scaler_meta(t, f)
            for j in map(str, range(1, N_INNER + 1)):
                if set(sm["inner"][j]["inner_train_patients"]) & test:
                    return False, "%s fold %d inner scaler record touches outer test" % (t, f)
    return True, ("across 5 folds x 2 selectors x 3 inner splits x 4 pipelines, no "
                  "outer-test patient appears in any fitting set, inner split, final "
                  "selector record or scaler record")


@check("B", "7 the inner selector sees inner-training patients only (refit exactly)")
def b7():
    for f in FOLDS:
        train, test = outer_sets(f)
        assign = inner_cv_assignment(CFG, COHORT, f, sorted(train))
        for sel in ("mrmr", "lasso"):
            rec = inner_record(sel, f)
            for j in range(1, N_INNER + 1):
                want_tr = sorted(p for p in sorted(train) if assign[p] != j)
                want_va = sorted(p for p in sorted(train) if assign[p] == j)
                got = rec["inner"][str(j)]
                if list(got["fit_patients"]) != want_tr:
                    return False, "%s fold %d inner %d fitting set differs from the " \
                                  "re-derived inner-training split" % (sel, f, j)
                if list(got["inner_val_patients"]) != want_va:
                    return False, "%s fold %d inner %d validation set differs" % (sel, f, j)
                redo = fit_selector_inner(CFG, SUMMARIES, sel, want_tr)
                for k in got["candidates"]:
                    if (redo["candidates"][k]["selected_indices"]
                            != got["candidates"][k]["selected_indices"]):
                        return False, "%s fold %d inner %d candidate %s does not refit" % (
                            sel, f, j, k)
                    if bool(redo["candidates"][k]["valid"]) != bool(
                            got["candidates"][k]["valid"]):
                        return False, "%s fold %d inner %d candidate %s validity differs" % (
                            sel, f, j, k)
    return True, ("all 30 inner selector records (5 folds x 2 selectors x 3 splits) refit "
                  "EXACTLY from the frozen bags using only their inner-training patients")


@check("B", "8 the LASSO summary scaler sees inner-training patients only")
def b8():
    for f in FOLDS:
        train, test = outer_sets(f)
        assign = inner_cv_assignment(CFG, COHORT, f, sorted(train))
        rec = inner_record("lasso", f)
        for j in range(1, N_INNER + 1):
            itr = sorted(p for p in sorted(train) if assign[p] != j)
            x, _ = SUMMARIES.subset(itr)
            want_mean, want_std = x.mean(axis=0), x.std(axis=0, ddof=0)
            for k, c in rec["inner"][str(j)]["candidates"].items():
                if c["summary_scaler_source"] != "fitting_patient_summaries_only":
                    return False, "fold %d inner %d C=%s scaler source is %r" % (
                        f, j, k, c["summary_scaler_source"])
                if (np.max(np.abs(np.asarray(c["summary_scaler_mean"]) - want_mean)) > 1e-9
                        or np.max(np.abs(np.asarray(c["summary_scaler_std"])
                                         - want_std)) > 1e-9):
                    return False, "fold %d inner %d C=%s scaler != inner-train statistics" % (
                        f, j, k)
            # a scaler that had seen the inner-validation or outer-test patients differs
            for extra, label in ((sorted(p for p in sorted(train) if assign[p] == j),
                                  "inner-val"), (sorted(test), "outer-test")):
                xc, _ = SUMMARIES.subset(sorted(set(itr) | set(extra)))
                if np.max(np.abs(xc.mean(axis=0) - want_mean)) <= 1e-9:
                    return False, "fold %d inner %d: a %s-contaminated scaler would be " \
                                  "indistinguishable" % (f, j, label)
    return True, ("all 15 LASSO summary scalers equal their inner-training statistics to "
                  "<1e-9 and are provably distinct from inner-val- or test-contaminated ones")


@check("B", "9 mRMR relevance and redundancy see inner-training patients only")
def b9():
    for f in FOLDS:
        train, test = outer_sets(f)
        assign = inner_cv_assignment(CFG, COHORT, f, sorted(train))
        rec = inner_record("mrmr", f)
        scfg = CFG["selectors"]["mrmr"]
        for j in range(1, N_INNER + 1):
            itr = sorted(p for p in sorted(train) if assign[p] != j)
            x, y = SUMMARIES.subset(itr)
            fstat = anova_f_statistic(x, y, float(scfg["zero_variance_relevance"]))
            stored = rec["inner"][str(j)]["relevance_f"]
            for i, nm in enumerate(NAMES):
                if abs(float(stored[nm]) - float(fstat[i])) > 1e-9:
                    return False, "fold %d inner %d: F(%s) differs" % (f, j, nm)
            rank = mrmr_rank(x, y, int(max(scfg["candidates"])),
                             float(scfg["zero_variance_relevance"]),
                             float(scfg["zero_variance_correlation"]))
            if list(rank.order) != list(rec["inner"][str(j)]["ranking"]):
                return False, "fold %d inner %d: the mRMR order does not re-derive" % (f, j)
            corr = absolute_pearson_matrix(x, float(scfg["zero_variance_correlation"]))
            for s in rec["inner"][str(j)]["steps"][1:]:
                sel_so_far = [st["selected_index"] for st in
                              rec["inner"][str(j)]["steps"][:s["step"] - 1]]
                red = float(corr[s["selected_index"], sel_so_far].mean())
                if abs(red - float(s["mean_redundancy"])) > 1e-9:
                    return False, "fold %d inner %d step %d redundancy differs" % (
                        f, j, s["step"])
            # contaminated statistics differ, so the stored ones cannot have used them
            xc, yc = SUMMARIES.subset(sorted(set(itr) | set(test)))
            if np.max(np.abs(anova_f_statistic(xc, yc) - fstat)) <= 1e-9:
                return False, "fold %d inner %d: a test-contaminated F would be identical" % (
                    f, j)
    return True, ("all 15 mRMR records reproduce EXACTLY (F statistics, |Pearson| "
                  "redundancy at every greedy step, and the full ranking) from the "
                  "inner-training patients alone")


@check("B", "11 the final selector is refitted on ALL outer-training patients")
def b11():
    for f in FOLDS:
        train, _ = outer_sets(f)
        for t in TAGS:
            sel = SELECTOR_OF[t]
            fr = final_record(sel, t, f)
            if sorted(fr["fit_patients"]) != sorted(train):
                return False, "%s fold %d final selector fitted on %d patients, not the " \
                              "%d outer-training patients" % (t, f, len(fr["fit_patients"]),
                                                              len(train))
            if fr["n_fit_patients"] != len(train):
                return False, "%s fold %d n_fit_patients=%d" % (t, f, fr["n_fit_patients"])
            redo = fit_selector_inner(CFG, SUMMARIES, sel, sorted(train))
            cand = redo["candidates"][fr["chosen_candidate"]]
            if sorted(cand["selected_indices"]) != sorted(fr["selected_indices"]):
                return False, "%s fold %d final selection does not refit" % (t, f)
            u = unit(t, f)
            if sorted(u["selected_indices"]) != sorted(fr["selected_indices"]):
                return False, "%s fold %d unit record disagrees with the selector record" % (
                    t, f)
    return True, ("all 20 final selections (4 pipelines x 5 folds) refit EXACTLY from the "
                  "frozen bags using all outer-training patients and nothing else")


@check("B", "12 the final selector sees no outer-test patient (and would differ if it did)")
def b12():
    for f in FOLDS:
        train, test = outer_sets(f)
        for t in TAGS:
            sel = SELECTOR_OF[t]
            fr = final_record(sel, t, f)
            if set(fr["fit_patients"]) & test:
                return False, "%s fold %d final fitting set intersects the outer test" % (t, f)
            if set(fr["outer_test_patients"]) != test:
                return False, "%s fold %d records the wrong outer test" % (t, f)
        # a selector that had also seen the outer-test patients would rank differently
        x, y = SUMMARIES.subset(sorted(train))
        xc, yc = SUMMARIES.subset(sorted(set(train) | set(test)))
        if np.max(np.abs(anova_f_statistic(xc, yc) - anova_f_statistic(x, y))) <= 1e-9:
            return False, "fold %d: a test-contaminated relevance would be identical" % f
    return True, ("no final selector fitting set intersects its outer-test fold, and a "
                  "contaminated relevance vector is provably different in all 5 folds")


@check("B", "-- the joint (setting, epoch) choice re-derives from the stored surface")
def b_sel():
    for f in FOLDS:
        for t in TAGS:
            u = unit(t, f)
            sdir = ppath(CFG, CFG["outputs"]["surfaces_dir"])
            df = pd.read_csv(os.path.join(sdir, "%s_fold_%d_surface.csv" % (t, f)))
            meta = json.load(open(os.path.join(
                sdir, "%s_fold_%d_selection.json" % (t, f)), "r", encoding="utf-8"))
            order = list(meta["candidates_evaluated"])
            surface = {k: df["%s_mean_val_roc_auc" % k].values for k in order}
            counts = {k: float(meta["per_candidate"][k]["mean_n_selected"]) for k in order}
            key, epoch, auc, _ = choose_setting_and_epoch(surface, counts, order)
            if key != u["selected_candidate"] or epoch != u["selected_epoch"]:
                return False, "%s fold %d: re-derived (%s, %d) != stored (%s, %d)" % (
                    t, f, key, epoch, u["selected_candidate"], u["selected_epoch"])
            if abs(auc - float(u["selection_mean_inner_cv_roc_auc"])) > 1e-12:
                return False, "%s fold %d: selection AUC differs" % (t, f)
            if len(df) != int(CFG["training"]["max_epochs"]):
                return False, "%s fold %d surface covers %d epochs" % (t, f, len(df))
            # the choice never equals the outer-test AUC it must not have seen
            if abs(auc - float(u["fold_metrics"]["roc_auc"])) < 1e-12:
                return False, "%s fold %d: the selection score equals the outer-test AUC" % (
                    t, f)
            for j in range(1, N_INNER + 1):
                if "%s_inner_%d_val_roc_auc" % (u["selected_candidate"], j) not in df.columns:
                    return False, "%s fold %d: inner curve %d missing" % (t, f, j)
    return True, ("all 20 (setting, epoch) choices re-derive from the stored surfaces over "
                  "the full 100-epoch budget; none equals its outer-test ROC-AUC")


# ============================ C  preprocessing, class weighting, index mapping
@check("C", "10 the patch scaler uses selected dimensions of training patients only")
def c10():
    for f in FOLDS:
        train, test = outer_sets(f)
        for t in TAGS:
            z = scaler_npz(t, f)
            sel = [int(i) for i in z["selected_indices"]]
            redo = fit_selected_patch_scaler(COHORT, sorted(train), sel,
                                             float(CFG["preprocessing"]["min_std"]))
            if (np.max(np.abs(redo.mean - z["mean"])) > 1e-9
                    or np.max(np.abs(redo.std - z["std"])) > 1e-9):
                return False, "%s fold %d patch scaler does not refit from outer-train " \
                              "patches" % (t, f)
            contaminated = fit_selected_patch_scaler(
                COHORT, sorted(set(train) | set(test)), sel,
                float(CFG["preprocessing"]["min_std"]))
            if np.max(np.abs(contaminated.mean - redo.mean)) <= 1e-9:
                return False, "%s fold %d: a test-contaminated patch scaler would be " \
                              "identical" % (t, f)
            sm = scaler_meta(t, f)
            if sm["final_scaler_source"] != \
                    "all_outer_train_patches_selected_dimensions_only":
                return False, "%s fold %d scaler source is %r" % (t, f,
                                                                  sm["final_scaler_source"])
            want_patches = sum(COHORT.bags[p].n_patches for p in train)
            if sm["n_train_patches"] != want_patches or sm["n_train_patients"] != len(train):
                return False, "%s fold %d scaler counts %s/%s" % (
                    t, f, sm["n_train_patients"], sm["n_train_patches"])
    return True, ("all 20 patch scalers refit exactly from the outer-training patches over "
                  "the selected dimensions, and differ from a test-contaminated scaler")


@check("C", "13 pos_weight comes from PATIENT counts, never patch counts")
def c13():
    for f in FOLDS:
        train, _ = outer_sets(f)
        y = COHORT.labels(sorted(train))
        want = pos_weight_from_patients(y)
        patch_ratio = (sum(COHORT.bags[p].n_patches for p in train if COHORT.bags[p].label == 0)
                       / sum(COHORT.bags[p].n_patches for p in train
                             if COHORT.bags[p].label == 1))
        for t in TAGS:
            got = float(unit(t, f)["final_pos_weight"])
            if abs(got - want) > 1e-12:
                return False, "%s fold %d pos_weight %.6f != patient ratio %.6f" % (
                    t, f, got, want)
            if abs(got - patch_ratio) < 1e-9:
                return False, "%s fold %d pos_weight equals the PATCH ratio" % (t, f)
            if abs(float(scaler_npz(t, f)["pos_weight"]) - want) > 1e-12:
                return False, "%s fold %d scaler npz pos_weight differs" % (t, f)
            if abs(float(checkpoint(t, f)["pos_weight"]) - want) > 1e-12:
                return False, "%s fold %d checkpoint pos_weight differs" % (t, f)
    return True, ("all 20 pos_weights equal N_SCC/N_ADC over the outer-training PATIENTS "
                  "(2.93-3.03) and none equals the corresponding patch ratio")


@check("C", "14 selected indices map back correctly onto the 74 patch dimensions")
def c14():
    for f in FOLDS:
        train, _ = outer_sets(f)
        for t in TAGS:
            z = scaler_npz(t, f)
            sel = [int(i) for i in z["selected_indices"]]
            if sel != sorted(sel) or len(set(sel)) != len(sel):
                return False, "%s fold %d indices are not a sorted unique list" % (t, f)
            if not all(0 <= i < N_FEATURES for i in sel):
                return False, "%s fold %d index outside 0..73" % (t, f)
            # the stored mean of selected column c IS the mean of ORIGINAL column sel[c]
            x = np.concatenate([COHORT.bags[p].features for p in sorted(train)],
                               axis=0).astype(np.float64)
            if np.max(np.abs(x[:, sel].mean(axis=0) - z["mean"])) > 1e-9:
                return False, "%s fold %d: the scaler mean does not match the ORIGINAL " \
                              "columns it claims to index" % (t, f)
            ck = checkpoint(t, f)
            if list(ck["selected_indices"]) != sel:
                return False, "%s fold %d checkpoint indices differ from the scaler" % (t, f)
            if int(ck["input_dim"]) != len(sel):
                return False, "%s fold %d input_dim %d != %d selected" % (
                    t, f, ck["input_dim"], len(sel))
            enc = ck["model_state_dict"]["encoder.net.0.weight"]
            if int(enc.shape[1]) != len(sel):
                return False, "%s fold %d encoder expects %d inputs, %d selected" % (
                    t, f, enc.shape[1], len(sel))
    return True, ("in all 20 refits the selected indices are sorted, unique, inside 0..73, "
                  "reproduce the ORIGINAL column statistics, and set the encoder's input width")


@check("C", "15 selected feature names match their indices everywhere")
def c15():
    for f in FOLDS:
        for t in TAGS:
            sel = [int(i) for i in scaler_npz(t, f)["selected_indices"]]
            want = [NAMES[i] for i in sel]
            for label, got in (
                    ("scaler npz", [str(s) for s in scaler_npz(t, f)["selected_features"]]),
                    ("scaler json", scaler_meta(t, f)["selected_features"]),
                    ("checkpoint", list(checkpoint(t, f)["selected_features"])),
                    ("unit record", unit(t, f)["selected_features"]),
                    ("selector record",
                     final_record(SELECTOR_OF[t], t, f)["selected_features"])):
                if list(got) != want:
                    return False, "%s fold %d: %s names disagree with the indices" % (
                        t, f, label)
            fam = unit(t, f)["selected_families"]
            if set(fam) != set(want) or any(v not in ("firstorder", "glcm", "glrlm", "glszm")
                                            for v in fam.values()):
                return False, "%s fold %d: family map is wrong" % (t, f)
    return True, ("names, indices and families agree across the scaler npz, scaler json, "
                  "checkpoint, unit record and selector record in all 20 refits")


@check("C", "-- predictions reproduce from the saved checkpoint + scaler")
def c_repro():
    worst = 0.0
    n = 0
    for f in FOLDS:
        _, test = outer_sets(f)
        for t in TAGS:
            ck = checkpoint(t, f)
            sel = [int(i) for i in ck["selected_indices"]]
            mean = np.asarray(ck["scaler_mean"], dtype=np.float64)
            std = np.asarray(ck["scaler_std"], dtype=np.float64)
            model = build_model(MODEL_OF[t], CFG, input_dim=len(sel))
            model.load_state_dict(ck["model_state_dict"])
            model.eval()
            stored = preds(t).set_index("PatientID")
            with torch.no_grad():
                for pid in sorted(test):
                    x = (COHORT.bags[pid].features[:, sel].astype(np.float64) - mean) / std
                    logit, _ = model(torch.from_numpy(x.astype(np.float32)))
                    p = float(torch.sigmoid(logit).item())
                    worst = max(worst, abs(p - float(stored.loc[
                        pid, "predicted_probability_ADC"])))
                    n += 1
    if worst > 1e-9:
        return False, "worst reproduction error %.3g over %d predictions" % (worst, n)
    return True, "all %d out-of-fold predictions reproduce to < 1e-9" % n


# ============================================= D  predictions, numerical sanity
@check("D", "16 exactly one OOF prediction per patient per pipeline")
def d16():
    ref = None
    for t in TAGS:
        df = preds(t)
        if len(df) != N_PATIENTS or df["PatientID"].duplicated().any():
            return False, "%s: %d rows / duplicates" % (t, len(df))
        if sorted(df["PatientID"]) != sorted(COHORT.patient_ids):
            return False, "%s covers a different patient set" % t
        for _, r in df.iterrows():
            _, test = outer_sets(int(r["outer_fold"]))
            if r["PatientID"] not in test:
                return False, "%s: %s scored outside its held-out fold" % (t, r["PatientID"])
            if int(r["true_label"]) != int(COHORT.bags[r["PatientID"]].label):
                return False, "%s: label disagreement on %s" % (t, r["PatientID"])
        cur = df.sort_values("PatientID")[["PatientID", "outer_fold", "true_label"]]
        if ref is None:
            ref = cur
        elif not cur.reset_index(drop=True).equals(ref.reset_index(drop=True)):
            return False, "%s uses different folds or labels from the other pipelines" % t
    return True, ("4 pipelines x %d unique patients, each scored exactly once by the model "
                  "of its own held-out fold; folds and labels identical across pipelines"
                  % N_PATIENTS)


@check("D", "17 no NaN or Inf anywhere in the Phase-5C outputs")
def d17():
    for t in TAGS:
        p = preds(t)["predicted_probability_ADC"].values.astype(float)
        if not np.all(np.isfinite(p)) or p.min() < 0.0 or p.max() > 1.0:
            return False, "%s has non-finite or out-of-range probabilities" % t
    res = ppath(CFG, CFG["outputs"]["results_dir"])
    for f in FOLDS:
        for t in TAGS:
            df = pd.read_csv(os.path.join(res, "selector_surfaces",
                                          "%s_fold_%d_surface.csv" % (t, f)))
            if not np.all(np.isfinite(df.values.astype(float))):
                return False, "%s fold %d surface has non-finite entries" % (t, f)
            z = scaler_npz(t, f)
            if not (np.all(np.isfinite(z["mean"])) and np.all(np.isfinite(z["std"]))
                    and np.all(z["std"] > 0)):
                return False, "%s fold %d scaler has non-finite or zero std" % (t, f)
    m = json.load(open(os.path.join(res, "oof_metrics.json"), "r", encoding="utf-8"))
    for t, rec in m["models"].items():
        for k, v in rec["pooled_oof"].items():
            if isinstance(v, float) and not np.isfinite(v):
                return False, "pooled %s of %s is %r" % (k, t, v)
    # and the pooled metrics recompute from the prediction files
    for t in TAGS:
        df = preds(t)
        want = patient_metrics(df["true_label"].values,
                               df["predicted_probability_ADC"].values, THR)
        for k in ("roc_auc", "pr_auc", "balanced_accuracy", "mcc", "accuracy",
                  "sensitivity_adc", "specificity", "precision", "f1"):
            if abs(float(m["models"][t]["pooled_oof"][k]) - want[k]) > 1e-12:
                return False, "%s pooled %s does not recompute" % (t, k)
    return True, ("%d probabilities, 20 surfaces, 20 scalers and every pooled metric are "
                  "finite; all 9 pooled metrics recompute from the prediction files to 1e-12"
                  % (4 * N_PATIENTS))


# ================================ E  frozen inputs and the Phase-5 baseline
@check("E", "18 the frozen GTV+Rim bags are unchanged")
def e18():
    snap = pre_snapshot()
    changed = []
    for rel, want in snap.items():
        if rel.startswith("patch_features/gtv_rim/"):
            got = sha256_file(os.path.join(ROOT, *rel.split("/")))
            if got != want:
                changed.append(rel)
    if changed:
        return False, "%d GTV+Rim bag file(s) changed: %s" % (len(changed), changed[:5])
    n = len([r for r in snap if r.startswith("patch_features/gtv_rim/")])
    sigs = {b.config_signature for b in COHORT.bags.values()}
    if sigs != {str(CFG["phase3"]["expected_extraction_signature"])}:
        return False, "extraction signatures %s" % sigs
    total = sum(b.n_patches for b in COHORT.bags.values())
    if total != int(CFG["phase3"]["expected_patches"]):
        return False, "%d patches, expected %d" % (total, CFG["phase3"]["expected_patches"])
    run = json.load(open(ppath(CFG, CFG["outputs"]["run_json"]), "r", encoding="utf-8"))
    for fn, h in run["input_manifest"]["bag_sha256"].items():
        if sha256_file(os.path.join(ROOT, "patch_features", "gtv_rim", fn)) != h:
            return False, "%s changed since the Phase-5C run" % fn
    return True, ("%d bag + sidecar files byte-identical to the pre-Phase-5C snapshot; "
                  "signature %s; %d patches" % (n, sigs.pop(), total))


@check("E", "19 the frozen outer split is unchanged")
def e19():
    meta = json.load(open(ppath(CFG, CFG["phase3"]["splits_meta_file"]),
                          "r", encoding="utf-8"))
    now = sha256_file(ppath(CFG, CFG["phase3"]["splits_file"]))
    if now != meta["splits_file_sha256"]:
        return False, "split sha256 %s != recorded %s" % (now[:16],
                                                          meta["splits_file_sha256"][:16])
    run = json.load(open(ppath(CFG, CFG["outputs"]["run_json"]), "r", encoding="utf-8"))
    if now != run["input_manifest"]["splits_file_sha256"]:
        return False, "the split changed during the Phase-5C run"
    seen = set()
    for f in FOLDS:
        _, test = outer_sets(f)
        if seen & test:
            return False, "fold %d test set overlaps an earlier fold" % f
        seen |= test
    if seen != set(COHORT.patient_ids):
        return False, "the 5 test folds do not partition the 202 patients"
    p5c = pd.read_csv(os.path.join(ppath(CFG, CFG["outputs"]["inner_splits_dir"]),
                                   "fold_1.csv"))
    if set(p5c.loc[p5c["assignment"] == "outer_test", "PatientID"]) != outer_sets(1)[1]:
        return False, "the Phase-5C inner-split table disagrees with the frozen split"
    return True, ("sha256 %s matches its metadata and the run manifest; the 5 test folds "
                  "partition all %d patients exactly once" % (now[:16], N_PATIENTS))


@check("E", "20 the Phase-5 baseline predictions and artefacts are unchanged")
def e20():
    snap = pre_snapshot()
    changed = []
    for rel, want in snap.items():
        path = os.path.join(ROOT, *rel.split("/"))
        if not os.path.exists(path):
            changed.append("MISSING " + rel)
        elif sha256_file(path) != want:
            changed.append(rel)
    if changed:
        return False, "%d snapshot file(s) changed: %s" % (len(changed), changed[:6])
    for which in ("mean", "attention"):
        df = baseline_preds(which)
        m = patient_metrics(df["true_label"].values,
                            df["predicted_probability_ADC"].values, THR)
        want = float(CFG["baseline"]["expected_roc_auc"][which])
        if abs(m["roc_auc"] - want) > 5e-5:
            return False, "baseline %s pooled ROC-AUC %.4f != %.4f" % (which,
                                                                       m["roc_auc"], want)
    # Phase 5C wrote only into its own namespaces
    own = [CFG["outputs"]["results_dir"], CFG["outputs"]["feature_selection_dir"],
           CFG["outputs"]["preprocessing_dir"], CFG["outputs"]["inner_splits_dir"],
           CFG["patient_summary"]["dir"]]
    for rel in snap:
        for o in own:
            if rel.startswith(o + "/"):
                return False, "%s lies inside the Phase-5C namespace %s" % (rel, o)
    return True, ("all %d snapshot files byte-identical; the Phase-5 74-feature baseline "
                  "still scores 0.5702 / 0.5767 pooled ROC-AUC; every Phase-5C output path "
                  "is disjoint from the snapshot" % len(snap))


# ================================================ F  the paired patient bootstrap
@check("F", "21 the paired bootstrap resamples PATIENTS, never patches")
def f21():
    res = ppath(CFG, CFG["outputs"]["results_dir"])
    boot = json.load(open(os.path.join(res, "baseline_paired_bootstrap.json"),
                          "r", encoding="utf-8"))
    bcfg = CFG["baseline"]["bootstrap"]
    if boot["resampling_unit"] != "patient":
        return False, "resampling unit is %r" % boot["resampling_unit"]
    total_patches = sum(b.n_patches for b in COHORT.bags.values())
    if boot["n_paired_patients"] == total_patches:
        return False, "the bootstrap unit looks like patches"
    for i, t in enumerate(TAGS):
        c = boot["comparisons"][t]
        if not c["paired"] or c["resampling_unit"] != "patient":
            return False, "%s: not a paired patient-level resample" % t
        if c["n_resamples_requested"] != int(bcfg["n_resamples"]):
            return False, "%s: %d resamples" % (t, c["n_resamples_requested"])
        n = preds(t).sort_values("PatientID")
        b = baseline_preds(BASE_OF[t]).sort_values("PatientID")
        if n["PatientID"].tolist() != b["PatientID"].tolist():
            return False, "%s: the paired vectors are not aligned by PatientID" % t
        redo = bootstrap_differences(
            n["true_label"].values.astype(int),
            n["predicted_probability_ADC"].values.astype(float),
            b["predicted_probability_ADC"].values.astype(float),
            int(bcfg["n_resamples"]), int(bcfg["seed"]) + i, float(bcfg["ci"]))
        for k in ("roc_auc", "pr_auc"):
            for fkey in ("difference_mean", "ci_low", "ci_high"):
                if abs(redo["metrics"][k][fkey]
                       - float(c["metrics"][k][fkey])) > 1e-12:
                    return False, "%s: %s %s not reproducible from seed %d" % (
                        t, k, fkey, int(bcfg["seed"]) + i)
            if bool(redo["metrics"][k]["ci_includes_zero"]) != bool(
                    c["metrics"][k]["ci_includes_zero"]):
                return False, "%s: %s CI verdict differs" % (t, k)
    return True, ("all 4 comparisons replay EXACTLY from their recorded seeds; the unit is "
                  "the patient (%d), never the patch (%d)" % (N_PATIENTS, total_patches))


@check("F", "22 each bootstrap comparison holds exactly 202 paired patients")
def f22():
    res = ppath(CFG, CFG["outputs"]["results_dir"])
    boot = json.load(open(os.path.join(res, "baseline_paired_bootstrap.json"),
                          "r", encoding="utf-8"))
    if boot["n_paired_patients"] != N_PATIENTS:
        return False, "%d paired patients" % boot["n_paired_patients"]
    if int(CFG["baseline"]["bootstrap"]["expected_paired_patients"]) != N_PATIENTS:
        return False, "the config expects a different paired-patient count"
    for t in TAGS:
        n = preds(t).sort_values("PatientID")
        b = baseline_preds(BASE_OF[t]).sort_values("PatientID")
        merged = n.merge(b, on="PatientID", suffixes=("_new", "_base"))
        if len(merged) != N_PATIENTS:
            return False, "%s merges to %d patients" % (t, len(merged))
        if not (merged["true_label_new"] == merged["true_label_base"]).all():
            return False, "%s: label disagreement between the paired vectors" % t
        if merged["PatientID"].duplicated().any():
            return False, "%s: duplicate patients after the merge" % t
    return True, ("all 4 comparisons merge on PatientID to exactly %d paired patients "
                  "(%d ADC / %d SCC) with identical labels on both sides"
                  % (N_PATIENTS, boot["n_adc"], boot["n_scc"]))


# ======================================== G  scope: no radiomics, no Phase 6
@check("G", "23 no radiomics extraction occurs anywhere in Phase 5C")
def g23():
    banned = {"radiomics", "SimpleITK", "sitk", "nibabel", "pydicom"}
    for rel in PHASE5C_PY:
        tree = ast.parse(open(os.path.join(ROOT, rel), "r", encoding="utf-8").read())
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                if m.split(".")[0] in banned:
                    return False, "%s imports %s" % (rel, m)
    if "radiomics" in sys.modules or "SimpleITK" in sys.modules:
        return False, "a radiomics module is loaded in this interpreter"
    run = json.load(open(ppath(CFG, CFG["outputs"]["run_json"]), "r", encoding="utf-8"))
    if bool(run.get("radiomics_imported_during_training", True)):
        return False, "the run record says radiomics was imported during training"
    return True, ("no Phase-5C module imports radiomics/SimpleITK/nibabel/pydicom "
                  "statically or at runtime; the frozen bags are the only feature source")


@check("G", "24 no Phase-6 code, CNN, Transformer or CLAM training occurs")
def g24():
    banned = ("Conv1d", "Conv2d", "Conv3d", "ConvTranspose", "nn.Transformer",
              "MultiheadAttention", "TransformerEncoder", "resnet", "ResNet",
              "clam", "CLAM", "SMOTE", "smote")
    for rel in PHASE5C_PY:
        src = open(os.path.join(ROOT, rel), "r", encoding="utf-8").read()
        hit = [b for b in banned if b in src]
        if hit:
            return False, "%s contains %s" % (rel, hit)
    for f in FOLDS:
        for t in TAGS:
            ck = checkpoint(t, f)
            if ck["model_name"] not in ("mean_mil", "attention_mil"):
                return False, "%s fold %d is a %r model" % (t, f, ck["model_name"])
            for k in ck["model_state_dict"]:
                if "conv" in k.lower():
                    return False, "%s fold %d has a convolutional parameter %s" % (t, f, k)
            if list(ck["config"]["model"]["encoder_hidden"]) != [64, 32]:
                return False, "%s fold %d encoder is not 64 -> 32" % (t, f)
            if abs(float(ck["config"]["model"]["dropout"]) - 0.2) > 1e-12:
                return False, "%s fold %d dropout is not 0.2" % (t, f)
    for stray in ("src/train_cnn.py", "src/cnn_models.py", "src/transformer_mil.py",
                  "config/phase6.yaml", "reports/PHASE6_REPORT.md"):
        if os.path.exists(os.path.join(ROOT, *stray.split("/"))):
            return False, "Phase-6 artefact exists: %s" % stray
    return True, ("no convolutional, Transformer, CLAM or SMOTE construct in any Phase-5C "
                  "module; all 20 checkpoints are MeanMIL/GatedAttentionMIL with the "
                  "unchanged 64 -> 32 encoder and dropout 0.2; no Phase-6 file exists")


@check("G", "-- feature selection is the ONLY changed variable vs Phase 5")
def g_only():
    p5 = load_phase3_config(os.path.join(ROOT, "config", "phase5.yaml"))
    for key in ("learning_rate", "weight_decay", "max_epochs", "bags_per_batch",
                "loss", "optimizer", "device", "pos_weight_source"):
        if str(CFG["training"][key]) != str(p5["training"][key]):
            return False, "training.%s differs from Phase 5" % key
    for key in ("encoder_hidden", "activation", "dropout", "attention_hidden", "pooling"):
        if str(CFG["model"][key]) != str(p5["model"][key]):
            return False, "model.%s differs from Phase 5" % key
    for key in ("n_splits", "shuffle", "base_seed"):
        if str(CFG["inner_cv"][key]) != str(p5["inner_cv"][key]):
            return False, "inner_cv.%s differs from Phase 5" % key
    if str(CFG["evaluation"]["threshold"]) != str(p5["evaluation"]["threshold"]):
        return False, "the threshold differs from Phase 5"
    for key in ("features_dir", "splits_file", "feature_names_file", "expected_patients",
                "expected_adc", "expected_scc", "expected_features",
                "expected_extraction_signature"):
        if str(CFG["phase3"][key]) != str(p5["phase3"][key]):
            return False, "phase3.%s differs from Phase 5" % key
    for f in FOLDS:
        train, _ = outer_sets(f)
        a5 = inner_cv_assignment(CFG, COHORT, f, sorted(train))
        a05 = inner_cv_assignment(p5, COHORT, f, sorted(train))
        if a5 != a05:
            return False, "the Phase-5C inner CV of fold %d differs from Phase 5" % f
    return True, ("bags, split, inner-CV assignment, encoder, dropout, optimiser, loss, "
                  "class-weighting rule, epoch budget and threshold are identical to "
                  "Phase 5; only the input feature set changes")


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
        except Exception as exc:                                  # noqa: BLE001
            ok, msg = False, "EXCEPTION: %s" % str(exc)[:300]
        n_fail += 0 if ok else 1
        print("%-4s %s  %-*s  %s" % ("PASS" if ok else "FAIL", group, width, name, msg))
        results.append({"n": i, "group": group, "name": name, "pass": bool(ok),
                        "message": msg})

    out = {"n_checks": len(CHECKS), "n_passed": len(CHECKS) - n_fail, "n_failed": n_fail,
           "phase": "5c", "roi": "gtv_rim", "pipelines": TAGS, "folds": list(FOLDS),
           "n_inner_splits": N_INNER, "threshold": THR, "n_patients": N_PATIENTS,
           "n_features_before_selection": N_FEATURES, "checks": results}
    path = ppath(CFG, "reports/phase5c_tests.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\n%d checks, %d passed, %d failed  ->  %s"
          % (len(CHECKS), len(CHECKS) - n_fail, n_fail, path))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
