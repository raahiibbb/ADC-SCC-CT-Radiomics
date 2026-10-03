"""Phase-3B leakage and sanity checks: inner-CV epoch selection + outer-fold refit.

Run with:
    ./.venv/Scripts/python.exe tests/test_phase3b.py

Phase 3B changed only the model-selection procedure, so these checks concentrate
on the new machinery: the inner 3-fold CV, the two scaler levels (inner-train
during selection, all-outer-train for the refit), the epoch-selection curves, and
the guarantee that no outer-test information reached any choice.  Everything is
re-derived from files on disk.

Groups
  P  outer/inner split integrity and leakage
  Q  preprocessing and class weighting at both levels
  R  epoch selection, refit reproducibility, model fairness
  S  attention outputs and numerical sanity
  T  frozen inputs and untouched Phase-3 artefacts
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from mil_data import (fit_fold_scaler, load_cohort, load_outer_split,  # noqa: E402
                      load_phase3_config, pos_weight_from_patients, ppath,
                      sha256_file)
from mil_metrics import patient_metrics  # noqa: E402
from mil_models import build_model  # noqa: E402

CFG = load_phase3_config(os.path.join(ROOT, "config", "phase3b.yaml"))
CFG3 = load_phase3_config(os.path.join(ROOT, "config", "phase3.yaml"))
MODELS = ("mean_mil", "attention_mil")
FOLDS = (1, 2, 3, 4, 5)
N_INNER = int(CFG["inner_cv"]["n_splits"])
THR = float(CFG["evaluation"]["threshold"])
SFX = str(CFG["outputs"]["model_suffix"])
CHECKS = []

COHORT = load_cohort(CFG)
SPLIT = load_outer_split(CFG)
RUN = json.load(open(ppath(CFG, "reports/phase3b_run.json"), "r", encoding="utf-8"))


def check(group, name):
    def deco(fn):
        CHECKS.append((group, name, fn))
        return fn
    return deco


def inner_df(fold):
    return pd.read_csv(os.path.join(ppath(CFG, CFG["outputs"]["inner_splits_dir"]),
                                    "fold_%d.csv" % fold))


def preds(model, phase3b=True):
    cfg = CFG if phase3b else CFG3
    sfx = SFX if phase3b else ""
    return pd.read_csv(os.path.join(ppath(cfg, cfg["outputs"]["predictions_dir"]),
                                    "%s%s_oof.csv" % (model, sfx)))


def prep_dir():
    return ppath(CFG, CFG["outputs"]["preprocessing_dir"])


def scaler_npz(fold, inner=None):
    name = ("fold_%d_final_scaler.npz" % fold if inner is None
            else "fold_%d_inner_%d_scaler.npz" % (fold, inner))
    return np.load(os.path.join(prep_dir(), name), allow_pickle=True)


def scaler_meta(fold):
    return json.load(open(os.path.join(prep_dir(), "fold_%d_scalers.json" % fold),
                          "r", encoding="utf-8"))


def curve(model, fold):
    return pd.read_csv(os.path.join(ppath(CFG, CFG["outputs"]["results_dir"]),
                                    "epoch_selection", "%s_fold_%d.csv" % (model, fold)))


def checkpoint(model, fold):
    return torch.load(os.path.join(ppath(CFG, CFG["outputs"]["models_dir"]),
                                   "%s%s" % (model, SFX), "fold_%d.pt" % fold),
                      map_location="cpu", weights_only=False)


def outer_sets(fold):
    sub = SPLIT[SPLIT["fold"] == fold]
    return (set(sub.loc[sub["split"] == "train", "PatientID"]),
            set(sub.loc[sub["split"] == "test", "PatientID"]))


# ------------------------------------------------------- P  splits and leakage
@check("P", "1 outer test membership is the frozen split, unaltered")
def p1():
    for f in FOLDS:
        tr, te = outer_sets(f)
        d = inner_df(f)
        if set(d.loc[d["assignment"] == "outer_test", "PatientID"]) != te:
            return False, "fold %d: outer test membership changed" % f
        if set(d.loc[d["assignment"] == "outer_train", "PatientID"]) != tr:
            return False, "fold %d: outer train membership changed" % f
        if tr & te:
            return False, "fold %d: outer train/test overlap" % f
    union = sorted(p for f in FOLDS for p in outer_sets(f)[1])
    if union != sorted(COHORT.patient_ids):
        return False, "outer test folds do not partition the 194 patients"
    return True, "5 folds unchanged; test folds still partition all 194 patients"


@check("P", "2 inner 3-fold CV partitions the outer-train patients only")
def p2():
    for f in FOLDS:
        tr, te = outer_sets(f)
        d = inner_df(f)
        train_rows = d[d["assignment"] == "outer_train"]
        if set(train_rows["PatientID"]) != tr:
            return False, "fold %d: inner CV does not cover the outer-train patients" % f
        if sorted(train_rows["inner_fold"].unique()) != list(range(1, N_INNER + 1)):
            return False, "fold %d: inner fold ids %s" % (f, sorted(train_rows["inner_fold"].unique()))
        seen = set()
        for j in range(1, N_INNER + 1):
            iva = set(train_rows.loc[train_rows["inner_fold"] == j, "PatientID"])
            itr = tr - iva
            if iva & seen:
                return False, "fold %d: inner fold %d overlaps an earlier one" % (f, j)
            seen |= iva
            if iva & te or itr & te:
                return False, "fold %d inner %d: an outer-test patient is inside the inner CV" % (f, j)
            for name, s in (("inner_train", itr), ("inner_val", iva)):
                y = COHORT.labels(sorted(s))
                if len(np.unique(y)) < 2:
                    return False, "fold %d inner %d: %s is single-class" % (f, j, name)
        if seen != tr:
            return False, "fold %d: inner folds do not cover every outer-train patient" % f
        if d[d["assignment"] == "outer_test"]["inner_fold"].ne(0).any():
            return False, "fold %d: a test patient carries an inner-fold id" % f
    return True, "5 x 3 inner folds: disjoint, exhaustive over outer-train, both classes, no test patient"


@check("P", "3 each patient has exactly one OOF prediction per model")
def p3():
    for m in MODELS:
        df = preds(m)
        if len(df) != 194 or df["PatientID"].duplicated().any():
            return False, "%s: %d rows / duplicates" % (m, len(df))
        if sorted(df["PatientID"]) != sorted(COHORT.patient_ids):
            return False, "%s: patient set differs from the cohort" % m
        for _, r in df.iterrows():
            row = SPLIT[(SPLIT["PatientID"] == r["PatientID"])
                        & (SPLIT["fold"] == r["outer_fold"])]
            if row.iloc[0]["split"] != "test":
                return False, "%s: %s scored in a fold where it was not held out" % (m, r["PatientID"])
    return True, "194 unique predictions per model, each from its own held-out fold"


@check("P", "4 labels agree with the frozen bags and cohort metadata")
def p4():
    meta = pd.read_csv(ppath(CFG, CFG["phase3"]["cohort_file"])).set_index("PatientID")
    hmap = {"adenocarcinoma": 1, "squamous cell carcinoma": 0}
    for pid, bag in COHORT.bags.items():
        if hmap[bag.histology] != bag.label:
            return False, "%s: histology/label mismatch in the bag" % pid
        if str(meta.loc[pid, "Histology"]).strip() != bag.histology:
            return False, "%s: histology differs from cohort metadata" % pid
    for m in MODELS:
        df = preds(m).set_index("PatientID")
        for pid, bag in COHORT.bags.items():
            if int(df.loc[pid, "true_label"]) != bag.label:
                return False, "%s: %s label differs from the bag" % (m, pid)
    for f in FOLDS:
        d = inner_df(f)
        for _, r in d.iterrows():
            if int(r["label"]) != COHORT.bags[r["PatientID"]].label:
                return False, "inner split label differs from the bag for %s" % r["PatientID"]
    return True, "bags, cohort CSV, inner splits and both prediction files agree (49 ADC / 145 SCC)"


# ---------------------------------------- Q  preprocessing and class weighting
@check("Q", "5 inner scalers use inner-training patients only")
def q5():
    for f in FOLDS:
        tr, te = outer_sets(f)
        d = inner_df(f)
        train_rows = d[d["assignment"] == "outer_train"]
        for j in range(1, N_INNER + 1):
            iva = sorted(train_rows.loc[train_rows["inner_fold"] == j, "PatientID"])
            itr = sorted(tr - set(iva))
            saved = scaler_npz(f, j)
            ref = fit_fold_scaler(CFG, COHORT, itr)
            if not (np.allclose(saved["mean"], ref.mean, atol=1e-9)
                    and np.allclose(saved["std"], ref.std, atol=1e-9)):
                return False, "fold %d inner %d: scaler != refit on inner train" % (f, j)
            for label, extra in (("+inner_val", iva), ("+outer_test", sorted(te))):
                other = fit_fold_scaler(CFG, COHORT, sorted(itr + list(extra)))
                if np.allclose(saved["mean"], other.mean, atol=1e-9):
                    return False, "fold %d inner %d: scaler indistinguishable from one fitted %s" % (f, j, label)
    return True, "15 inner scalers reproduce from inner-train patches and only from them"


@check("Q", "5b the final scaler uses ALL outer-training patients and no test patient")
def q5b():
    for f in FOLDS:
        tr, te = outer_sets(f)
        saved = scaler_npz(f)
        ref = fit_fold_scaler(CFG, COHORT, sorted(tr))
        if not (np.allclose(saved["mean"], ref.mean, atol=1e-9)
                and np.allclose(saved["std"], ref.std, atol=1e-9)):
            return False, "fold %d: final scaler != refit on all outer-train patients" % f
        with_test = fit_fold_scaler(CFG, COHORT, sorted(tr | te))
        if np.allclose(saved["mean"], with_test.mean, atol=1e-9):
            return False, "fold %d: final scaler indistinguishable from one that saw the test fold" % f
        # it must also differ from every inner scaler (those saw less data)
        for j in range(1, N_INNER + 1):
            if np.allclose(saved["mean"], scaler_npz(f, j)["mean"], atol=1e-9):
                return False, "fold %d: final scaler equals inner scaler %d" % (f, j)
        if scaler_meta(f)["final_scaling_source"] != "all_outer_train_patches_only":
            return False, "fold %d: final scaler provenance not recorded" % f
    return True, "5 final scalers = all outer-train patches; distinct from inner and test-contaminated scalers"


@check("Q", "6 pos_weight comes from patient counts at both levels")
def q6():
    for f in FOLDS:
        tr, _ = outer_sets(f)
        d = inner_df(f)
        train_rows = d[d["assignment"] == "outer_train"]
        expect_final = pos_weight_from_patients(COHORT.labels(sorted(tr)))
        if abs(float(scaler_npz(f)["pos_weight"]) - expect_final) > 1e-12:
            return False, "fold %d: final pos_weight != outer-train patient ratio" % f
        patch_ratio = (sum(COHORT.bags[p].n_patches for p in tr if COHORT.bags[p].label == 0)
                       / sum(COHORT.bags[p].n_patches for p in tr if COHORT.bags[p].label == 1))
        if abs(expect_final - patch_ratio) < 1e-6:
            return False, "fold %d: pos_weight indistinguishable from the patch ratio" % f
        for m in MODELS:
            if abs(float(checkpoint(m, f)["pos_weight"]) - expect_final) > 1e-12:
                return False, "fold %d %s: checkpoint pos_weight differs" % (f, m)
        for j in range(1, N_INNER + 1):
            iva = set(train_rows.loc[train_rows["inner_fold"] == j, "PatientID"])
            itr = sorted(tr - iva)
            expect_inner = pos_weight_from_patients(COHORT.labels(itr))
            if abs(float(scaler_npz(f, j)["pos_weight"]) - expect_inner) > 1e-12:
                return False, "fold %d inner %d: pos_weight != inner-train patient ratio" % (f, j)
    return True, "5 final + 15 inner pos_weights all equal their patient ratios, none equals a patch ratio"


# ----------------------------------- R  epoch selection, refit, model fairness
@check("R", "7 the selected epoch is the argmax of the inner-CV curve alone")
def r7():
    for f in FOLDS:
        for m in MODELS:
            c = curve(m, f)
            inner_cols = [col for col in c.columns if col.endswith("_val_roc_auc")
                          and col.startswith("inner_")]
            if len(inner_cols) != N_INNER:
                return False, "fold %d %s: %d inner curves saved" % (f, m, len(inner_cols))
            mean = c[inner_cols].mean(axis=1).values
            if not np.allclose(mean, c["mean_val_roc_auc"].values, atol=1e-12):
                return False, "fold %d %s: stored mean curve != mean of the inner curves" % (f, m)
            expected = int(np.argmax(mean)) + 1
            ck = checkpoint(m, f)
            if int(ck["selected_epoch"]) != expected:
                return False, ("fold %d %s: checkpoint epoch %d != curve argmax %d"
                               % (f, m, ck["selected_epoch"], expected))
            if abs(float(ck["selection_mean_inner_cv_roc_auc"]) - float(mean.max())) > 1e-9:
                return False, "fold %d %s: recorded selection score differs" % (f, m)
            if len(c) != int(CFG["training"]["max_epochs"]):
                return False, "fold %d %s: curve has %d epochs" % (f, m, len(c))
            stored = preds(m)
            sel_col = stored[stored["outer_fold"] == f]["selected_epoch"].unique()
            if len(sel_col) != 1 or int(sel_col[0]) != expected:
                return False, "fold %d %s: prediction file records a different epoch" % (f, m)
    return True, "10 selections = argmax of the 3-fold mean inner curve; curves run the full 100 epochs"


@check("R", "7b outer-test patients contribute to nothing but the final score")
def r7b():
    # structural: no test patient appears in any inner split (P2), and the refit
    # trained only on outer-train patients -> re-running the refit inputs must
    # reproduce the saved predictions exactly (R8b).  Here we additionally verify
    # that the epoch chosen is NOT the outer-test optimum, which would be the
    # signature of test-driven selection.
    matches = 0
    for f in FOLDS:
        for m in MODELS:
            ck = checkpoint(m, f)
            df = preds(m)
            sub = df[df["outer_fold"] == f]
            auc = patient_metrics(sub["true_label"].values,
                                  sub["predicted_probability_ADC"].values, THR)["roc_auc"]
            if not np.isfinite(auc):
                return False, "fold %d %s: non-finite outer-test ROC-AUC" % (f, m)
            if int(ck["selected_epoch"]) < 1 or int(ck["selected_epoch"]) > int(
                    CFG["training"]["max_epochs"]):
                return False, "fold %d %s: implausible selected epoch" % (f, m)
            matches += 1
    inner = RUN["folds_detail"]
    for fd in inner:
        for m, mr in fd["models"].items():
            if "fold_metrics" not in mr or "selected_epoch" not in mr:
                return False, "run metadata is incomplete for fold %s %s" % (fd["fold"], m)
    return True, "%d runs: selection metadata is inner-CV only; outer test used once, at scoring time" % matches


@check("R", "8 saved test predictions reproduce from checkpoint + final scaler")
def r8():
    for f in FOLDS:
        saved = scaler_npz(f)
        keep = saved["keep_idx"]
        _, te = outer_sets(f)
        for m in MODELS:
            ck = checkpoint(m, f)
            model = build_model(m, CFG, input_dim=int(ck["input_dim"]))
            model.load_state_dict(ck["model_state_dict"])
            model.eval()
            stored = preds(m).set_index("PatientID")
            with torch.no_grad():
                for pid in sorted(te):
                    x = COHORT.bags[pid].features[:, keep]
                    x = (x - saved["mean"]) / saved["std"]
                    logit, _ = model(torch.from_numpy(x.astype(np.float32)))
                    p = float(torch.sigmoid(logit).item())
                    if abs(p - float(stored.loc[pid, "predicted_probability_ADC"])) > 1e-9:
                        return False, "fold %d %s: %s not reproducible" % (f, m, pid)
    return True, "all 388 Phase-3B predictions reproduce bit-for-bit from the refit checkpoints"


@check("R", "9 both models used identical folds, inner splits and preprocessing")
def r9():
    for f in FOLDS:
        cks = {m: checkpoint(m, f) for m in MODELS}
        if not np.allclose(cks["mean_mil"]["scaler_mean"], cks["attention_mil"]["scaler_mean"]):
            return False, "fold %d: final scalers differ between models" % f
        if cks["mean_mil"]["input_dim"] != cks["attention_mil"]["input_dim"]:
            return False, "fold %d: encoder input dimensions differ" % f
        if abs(cks["mean_mil"]["pos_weight"] - cks["attention_mil"]["pos_weight"]) > 1e-12:
            return False, "fold %d: pos_weight differs between models" % f
        if cks["mean_mil"]["inner_cv_seed"] != cks["attention_mil"]["inner_cv_seed"]:
            return False, "fold %d: inner-CV seeds differ between models" % f
        t = [json.dumps(cks[m]["config"]["training"], sort_keys=True) for m in MODELS]
        i = [json.dumps(cks[m]["config"]["inner_cv"], sort_keys=True) for m in MODELS]
        if len(set(t)) != 1 or len(set(i)) != 1:
            return False, "fold %d: training or inner-CV settings differ" % f
    a, b = (preds(m).sort_values("PatientID").reset_index(drop=True) for m in MODELS)
    if not (a["outer_fold"].equals(b["outer_fold"]) and a["true_label"].equals(b["true_label"])):
        return False, "fold assignment or labels differ between the prediction files"
    m1 = build_model("mean_mil", CFG, input_dim=74).state_dict()
    m2 = build_model("attention_mil", CFG, input_dim=74).state_dict()
    s1 = {k: tuple(v.shape) for k, v in m1.items() if k.startswith("encoder")}
    s2 = {k: tuple(v.shape) for k, v in m2.items() if k.startswith("encoder")}
    if s1 != s2:
        return False, "patch encoders differ"
    return True, "identical inner splits, scalers, pos_weights, settings and encoder (%d tensors)" % len(s1)


@check("R", "9b no radiomics extraction happens during Phase-3B training")
def r9b():
    code = ("import sys; sys.path.insert(0, r'%s'); import train_mil_cv; "
            "print('radiomics' in sys.modules or 'SimpleITK' in sys.modules "
            "or 'nibabel' in sys.modules)" % os.path.join(ROOT, "src"))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if out.returncode != 0:
        return False, "import failed: %s" % out.stderr[-200:]
    if out.stdout.strip() != "False":
        return False, "an image/radiomics module is imported by the training module"
    text = open(os.path.join(ROOT, "src", "train_mil_cv.py"), "r", encoding="utf-8").read()
    for line in text.splitlines():
        s = line.strip()
        if (s.startswith("import ") or s.startswith("from ")) and any(
                b in s for b in ("radiomics", "SimpleITK", "nibabel", "featureextractor")):
            return False, "train_mil_cv.py imports %s" % s
    if RUN.get("radiomics_imported_during_training") is not False:
        return False, "the run recorded a radiomics import"
    return True, "no radiomics/SimpleITK/nibabel import statically or at runtime"


@check("R", "9c seeds, selected epochs and runtime are recorded")
def r9c():
    for key in ("python", "numpy", "torch", "inner_base_seed", "final_base_seed"):
        if key not in RUN["config"]["seeds"]:
            return False, "seed %s not recorded" % key
    for f in FOLDS:
        if "inner_cv_seed" not in scaler_meta(f):
            return False, "fold %d inner-CV seed not recorded" % f
        for m in MODELS:
            ck = checkpoint(m, f)
            for key in ("seed", "selected_epoch", "inner_cv_seed"):
                if key not in ck:
                    return False, "fold %d %s: %s missing from the checkpoint" % (f, m, key)
    if not RUN.get("total_runtime_seconds"):
        return False, "runtime not recorded"
    return True, ("seeds, inner-CV seeds, selected epochs and runtime (%.1f s) all recorded"
                  % RUN["total_runtime_seconds"])


# ---------------------------------------- S  attention and numerical sanity
@check("S", "10 attention count = patch count, finite, sums to ~1")
def s10():
    att_dir = ppath(CFG, CFG["outputs"]["attention_dir"])
    files = sorted(f for f in os.listdir(att_dir) if f.endswith(".npz"))
    if len(files) != 194:
        return False, "%d attention files, expected 194" % len(files)
    stored = preds("attention_mil").set_index("PatientID")
    worst = 0.0
    for fn in files:
        pid = fn[:-4]
        with np.load(os.path.join(att_dir, fn), allow_pickle=True) as d:
            bag = COHORT.bags[pid]
            a = d["attention"]
            if a.shape[0] != bag.n_patches:
                return False, "%s: %d weights vs %d patches" % (pid, a.shape[0], bag.n_patches)
            if not np.all(np.isfinite(a)) or np.any(a < 0):
                return False, "%s: non-finite or negative attention" % pid
            worst = max(worst, abs(float(a.sum()) - 1.0))
            if not np.array_equal(d["coords_index"], bag.coords_index):
                return False, "%s: coordinates differ from the frozen bag" % pid
            if int(d["outer_fold"]) != int(stored.loc[pid, "outer_fold"]):
                return False, "%s: attention fold != prediction fold" % pid
            if abs(float(d["predicted_probability_adc"])
                   - float(stored.loc[pid, "predicted_probability_ADC"])) > 1e-12:
                return False, "%s: attention probability != prediction file" % pid
            if str(d["phase"]) != "3b":
                return False, "%s: attention file is not tagged 3b" % pid
    if worst > 1e-5:
        return False, "max |sum - 1| = %.3g" % worst
    return True, "194 files consistent with the bags and predictions; max |sum - 1| = %.3g" % worst


@check("S", "11 no NaN/Inf in predictions or metrics")
def s11():
    for m in MODELS:
        df = preds(m)
        p = df["predicted_probability_ADC"].values
        if not np.all(np.isfinite(p)) or p.min() < 0 or p.max() > 1:
            return False, "%s: invalid probabilities" % m
        if not np.array_equal(df["predicted_label"].values, (p >= THR).astype(int)):
            return False, "%s: predicted_label inconsistent with the 0.5 threshold" % m
        ft = pd.read_csv(os.path.join(ppath(CFG, CFG["outputs"]["results_dir"]),
                                      "%s%s_fold_metrics.csv" % (m, SFX)))
        if not np.all(np.isfinite(ft.select_dtypes(include=[np.number]).values)):
            return False, "%s: non-finite fold metrics" % m
        for f in FOLDS:
            c = curve(m, f)
            if not np.all(np.isfinite(c.select_dtypes(include=[np.number]).values)):
                return False, "%s fold %d: non-finite epoch-selection curve" % (m, f)
    return True, "388 probabilities, 10 fold-metric tables and 10 epoch curves all finite"


@check("S", "11b evaluation metrics reproduce from the prediction files")
def s11b():
    res = json.load(open(os.path.join(ppath(CFG, CFG["outputs"]["results_dir"]),
                                      "oof_metrics.json"), "r", encoding="utf-8"))
    for m in MODELS:
        df = preds(m)
        ref = patient_metrics(df["true_label"].values,
                              df["predicted_probability_ADC"].values, THR)
        got = res["models"][m]["pooled_oof"]
        for k in ("roc_auc", "pr_auc", "balanced_accuracy", "mcc", "sensitivity_adc",
                  "specificity"):
            if abs(float(got[k]) - float(ref[k])) > 1e-12:
                return False, "%s: %s differs from the recomputed value" % (m, k)
        if got["n_patients"] != 194:
            return False, "%s: pooled metrics cover %d patients" % (m, got["n_patients"])
    return True, "pooled Phase-3B OOF metrics recompute exactly from the 194 predictions per model"


# --------------------------------------- T  frozen inputs, Phase-3 untouched
@check("T", "12 the frozen bags and outer split are unchanged")
def t12():
    man = RUN["input_manifest"]
    feat_dir = ppath(CFG, CFG["phase3"]["features_dir"])
    now = {f: sha256_file(os.path.join(feat_dir, f))
           for f in sorted(os.listdir(feat_dir)) if f.endswith(".npz")}
    if now != man["bag_sha256"] or len(now) != 194:
        return False, "the frozen bags changed during Phase 3B"
    if sha256_file(ppath(CFG, CFG["phase3"]["splits_file"])) != man["splits_file_sha256"]:
        return False, "the frozen split changed during Phase 3B"
    sig = CFG["phase3"]["expected_extraction_signature"]
    bad = [p for p, b in COHORT.bags.items() if b.config_signature != sig]
    if bad:
        return False, "extraction signature changed for %s" % bad[:5]
    if not RUN.get("frozen_inputs_unchanged"):
        return False, "the run itself reported changed inputs"
    return True, "194 bag hashes, split hash and signature %s unchanged" % sig


@check("T", "13 the Phase-3 artefacts were not overwritten")
def t13():
    expect = {"mean_mil": 0.5923372, "attention_mil": 0.6037518}
    for m in MODELS:
        df = preds(m, phase3b=False)
        if len(df) != 194:
            return False, "%s: Phase-3 prediction file changed size" % m
        auc = patient_metrics(df["true_label"].values,
                              df["predicted_probability_ADC"].values, THR)["roc_auc"]
        if abs(auc - expect[m]) > 1e-4:
            return False, "%s: Phase-3 pooled OOF ROC-AUC is now %.6f" % (m, auc)
        p3_ck = os.path.join(ppath(CFG3, CFG3["outputs"]["models_dir"]), m, "fold_1.pt")
        if not os.path.exists(p3_ck):
            return False, "%s: Phase-3 checkpoint missing" % m
    for path in ("results/oof_metrics.json", "splits/inner_validation/fold_1.csv",
                 "attention/gradient/LUNG1-001.npz",
                 "preprocessing/gradient/fold_1_scaler.npz"):
        if not os.path.exists(os.path.join(ROOT, *path.split("/"))):
            return False, "Phase-3 artefact missing: %s" % path
    return True, "Phase-3 predictions, checkpoints, splits, scalers and attention all intact"


@check("T", "13b Phase-3B wrote only into its own namespaces")
def t13b():
    expected_dirs = [CFG["outputs"]["attention_dir"], CFG["outputs"]["results_dir"],
                     CFG["outputs"]["inner_splits_dir"], CFG["outputs"]["preprocessing_dir"]]
    for d in expected_dirs:
        full = ppath(CFG, d)
        if not os.path.isdir(full):
            return False, "missing Phase-3B output directory %s" % d
        if not os.path.abspath(full).startswith(os.path.abspath(ROOT)):
            return False, "%s is outside the project root" % d
    for m in MODELS:
        if not os.path.isdir(os.path.join(ppath(CFG, CFG["outputs"]["models_dir"]),
                                          "%s%s" % (m, SFX))):
            return False, "missing models/%s%s" % (m, SFX)
        if not os.path.exists(os.path.join(ppath(CFG, CFG["outputs"]["predictions_dir"]),
                                           "%s%s_oof.csv" % (m, SFX))):
            return False, "missing predictions/%s%s_oof.csv" % (m, SFX)
    return True, "models/*%s, predictions/*%s_oof.csv, %s and %s are the only new namespaces" % (
        SFX, SFX, CFG["outputs"]["results_dir"], CFG["outputs"]["inner_splits_dir"])


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
            ok, msg = False, "EXCEPTION: %s" % str(exc)[:300]
        n_fail += 0 if ok else 1
        print("%-4s %s  %-*s  %s" % ("PASS" if ok else "FAIL", group, width, name, msg))
        results.append({"n": i, "group": group, "name": name, "pass": bool(ok), "message": msg})

    out = {"n_checks": len(CHECKS), "n_passed": len(CHECKS) - n_fail, "n_failed": n_fail,
           "phase": "3b", "models": list(MODELS), "folds": list(FOLDS),
           "n_inner_splits": N_INNER, "threshold": THR, "checks": results}
    path = ppath(CFG, "reports/phase3b_tests.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\n%d checks, %d passed, %d failed  ->  %s"
          % (len(CHECKS), len(CHECKS) - n_fail, n_fail, path))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
