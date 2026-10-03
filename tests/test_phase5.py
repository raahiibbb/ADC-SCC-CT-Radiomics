"""Phase-5 leakage and sanity checks: GTV+Rim robust MIL evaluation.

Run with:
    ./.venv/Scripts/python.exe tests/test_phase5.py

Phase 5 re-uses the Phase-3B procedure on the frozen 202 GTV+Rim bags, so these
checks re-derive the whole chain from files on disk: the frozen outer split, the
inner 3-fold CV, both scaler levels, the epoch-selection curves, the refit
predictions, the attention outputs, and the paired 194-patient ROI comparison
against the untouched Gradient Phase-3B predictions.

Groups
  A  outer/inner split integrity and leakage        (required checks 1-3, 8, 9)
  B  preprocessing and class weighting              (required checks 4-6)
  C  epoch selection, refit reproducibility, fairness (required checks 7, 10, 11, 15)
  D  attention outputs and numerical sanity         (required checks 12-14)
  E  frozen inputs and untouched earlier phases     (required checks 16, 17)
  F  the paired ROI comparison                      (required checks 18-20)
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
                      load_phase3_config, model_tag, pos_weight_from_patients,
                      ppath, sha256_file)
from mil_metrics import patient_metrics  # noqa: E402
from mil_models import build_model  # noqa: E402
from roi_paired_compare import bootstrap_differences  # noqa: E402

CFG = load_phase3_config(os.path.join(ROOT, "config", "phase5.yaml"))
CFG3B = load_phase3_config(os.path.join(ROOT, "config", "phase3b.yaml"))
MODELS = ("mean_mil", "attention_mil")
FOLDS = (1, 2, 3, 4, 5)
N_INNER = int(CFG["inner_cv"]["n_splits"])
THR = float(CFG["evaluation"]["threshold"])
N_PATIENTS = int(CFG["phase3"]["expected_patients"])
N_SHARED = int(CFG["roi_comparison"]["expected_shared_patients"])
CHECKS = []

COHORT = load_cohort(CFG)
SPLIT = load_outer_split(CFG)
RUN = json.load(open(ppath(CFG, CFG["outputs"]["run_json"]), "r", encoding="utf-8"))


def check(group, name):
    def deco(fn):
        CHECKS.append((group, name, fn))
        return fn
    return deco


def inner_df(fold):
    return pd.read_csv(os.path.join(ppath(CFG, CFG["outputs"]["inner_splits_dir"]),
                                    "fold_%d.csv" % fold))


def preds(model, cfg=CFG):
    return pd.read_csv(os.path.join(ppath(cfg, cfg["outputs"]["predictions_dir"]),
                                    "%s_oof.csv" % model_tag(cfg, model)))


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
                                   model_tag(CFG, model), "fold_%d.pt" % fold),
                      map_location="cpu", weights_only=False)


def outer_sets(fold):
    sub = SPLIT[SPLIT["fold"] == fold]
    return (set(sub.loc[sub["split"] == "train", "PatientID"]),
            set(sub.loc[sub["split"] == "test", "PatientID"]))


def pre_snapshot():
    return json.load(open(ppath(CFG, "reports/phase5_pre_snapshot.json"),
                          "r", encoding="utf-8"))["files"]


# ------------------------------------------- A  splits, leakage, OOF integrity
@check("A", "1 the frozen 202-patient outer split is unchanged")
def a1():
    meta = json.load(open(ppath(CFG, CFG["phase3"]["splits_meta_file"]),
                          "r", encoding="utf-8"))
    now = sha256_file(ppath(CFG, CFG["phase3"]["splits_file"]))
    if now != meta["splits_file_sha256"]:
        return False, "split sha256 %s != recorded %s" % (now[:16], meta["splits_file_sha256"][:16])
    if now != RUN["input_manifest"]["splits_file_sha256"]:
        return False, "the split changed during the Phase-5 run"
    if int(meta["n_patients"]) != N_PATIENTS or int(meta["n_adc"]) != 51:
        return False, "split meta describes a different cohort"
    for f in FOLDS:
        tr, te = outer_sets(f)
        if tr & te:
            return False, "fold %d: outer train/test overlap" % f
        d = inner_df(f)
        if set(d.loc[d["assignment"] == "outer_test", "PatientID"]) != te:
            return False, "fold %d: outer test membership changed" % f
        if set(d.loc[d["assignment"] == "outer_train", "PatientID"]) != tr:
            return False, "fold %d: outer train membership changed" % f
    union = sorted(p for f in FOLDS for p in outer_sets(f)[1])
    if union != sorted(COHORT.patient_ids):
        return False, "outer test folds do not partition the %d patients" % N_PATIENTS
    return True, ("sha256 %s… verified against its meta; 5 test folds partition all %d patients"
                  % (now[:16], N_PATIENTS))


@check("A", "2/3 inner CV is patient-level and holds outer-training patients only")
def a2():
    for f in FOLDS:
        tr, te = outer_sets(f)
        d = inner_df(f)
        train_rows = d[d["assignment"] == "outer_train"]
        if set(train_rows["PatientID"]) != tr:
            return False, "fold %d: inner CV does not cover the outer-train patients" % f
        if sorted(train_rows["inner_fold"].unique()) != list(range(1, N_INNER + 1)):
            return False, "fold %d: inner fold ids %s" % (
                f, sorted(train_rows["inner_fold"].unique()))
        if train_rows["PatientID"].duplicated().any():
            return False, "fold %d: a patient appears twice in the inner CV" % f
        seen = set()
        for j in range(1, N_INNER + 1):
            iva = set(train_rows.loc[train_rows["inner_fold"] == j, "PatientID"])
            itr = tr - iva
            if iva & seen:
                return False, "fold %d: inner fold %d overlaps an earlier one" % (f, j)
            seen |= iva
            if iva & te or itr & te:
                return False, ("fold %d inner %d: an outer-test patient is inside the "
                               "inner CV" % (f, j))
            for name, s in (("inner_train", itr), ("inner_val", iva)):
                if len(np.unique(COHORT.labels(sorted(s)))) < 2:
                    return False, "fold %d inner %d: %s is single-class" % (f, j, name)
        if seen != tr:
            return False, "fold %d: inner folds do not cover every outer-train patient" % f
        if d[d["assignment"] == "outer_test"]["inner_fold"].ne(0).any():
            return False, "fold %d: a test patient carries an inner-fold id" % f
    return True, ("5 x %d inner folds: patient-level, disjoint, exhaustive over outer-train, "
                  "both classes, no outer-test patient anywhere" % N_INNER)


@check("A", "8 each of the 202 patients has exactly one OOF prediction per model")
def a8():
    for m in MODELS:
        df = preds(m)
        if len(df) != N_PATIENTS or df["PatientID"].duplicated().any():
            return False, "%s: %d rows / duplicates" % (m, len(df))
        if sorted(df["PatientID"]) != sorted(COHORT.patient_ids):
            return False, "%s: patient set differs from the cohort" % m
        for _, r in df.iterrows():
            row = SPLIT[(SPLIT["PatientID"] == r["PatientID"])
                        & (SPLIT["fold"] == r["outer_fold"])]
            if row.iloc[0]["split"] != "test":
                return False, "%s: %s scored in a fold where it was not held out" % (
                    m, r["PatientID"])
    return True, "%d unique predictions per model, each from its own held-out fold" % N_PATIENTS


@check("A", "9 labels agree with the frozen bags and the cohort metadata")
def a9():
    meta = pd.read_csv(ppath(CFG, CFG["phase3"]["cohort_file"])).set_index("PatientID")
    hmap = {"adenocarcinoma": 1, "squamous cell carcinoma": 0}
    for pid, bag in COHORT.bags.items():
        if hmap[bag.histology] != bag.label:
            return False, "%s: histology/label mismatch inside the bag" % pid
        if str(meta.loc[pid, "Histology"]).strip() != bag.histology:
            return False, "%s: histology differs from cohort metadata" % pid
    for m in MODELS:
        df = preds(m).set_index("PatientID")
        for pid, bag in COHORT.bags.items():
            if int(df.loc[pid, "true_label"]) != bag.label:
                return False, "%s: %s label differs from the bag" % (m, pid)
    for f in FOLDS:
        for _, r in inner_df(f).iterrows():
            if int(r["label"]) != COHORT.bags[r["PatientID"]].label:
                return False, "inner split label differs from the bag for %s" % r["PatientID"]
    y = COHORT.labels(COHORT.patient_ids)
    if int((y == 1).sum()) != 51 or int((y == 0).sum()) != 151:
        return False, "cohort is %d ADC / %d SCC" % ((y == 1).sum(), (y == 0).sum())
    return True, "bags, cohort CSV, split file, inner splits and both prediction files agree (51 ADC / 151 SCC)"


# ------------------------------------ B  preprocessing and class weighting
@check("B", "4 inner scalers use inner-training patients only")
def b4():
    for f in FOLDS:
        tr, te = outer_sets(f)
        train_rows = inner_df(f)
        train_rows = train_rows[train_rows["assignment"] == "outer_train"]
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
                    return False, ("fold %d inner %d: scaler indistinguishable from one "
                                   "fitted %s" % (f, j, label))
    return True, "%d inner scalers reproduce from inner-train patches and only from them" % (
        5 * N_INNER)


@check("B", "6 the final scaler uses ALL outer-training patients and no test patient")
def b6():
    for f in FOLDS:
        tr, te = outer_sets(f)
        saved = scaler_npz(f)
        ref = fit_fold_scaler(CFG, COHORT, sorted(tr))
        if not (np.allclose(saved["mean"], ref.mean, atol=1e-9)
                and np.allclose(saved["std"], ref.std, atol=1e-9)):
            return False, "fold %d: final scaler != refit on all outer-train patients" % f
        with_test = fit_fold_scaler(CFG, COHORT, sorted(tr | te))
        if np.allclose(saved["mean"], with_test.mean, atol=1e-9):
            return False, ("fold %d: final scaler indistinguishable from one that saw the "
                           "test fold" % f)
        for j in range(1, N_INNER + 1):
            if np.allclose(saved["mean"], scaler_npz(f, j)["mean"], atol=1e-9):
                return False, "fold %d: final scaler equals inner scaler %d" % (f, j)
        if scaler_meta(f)["final_scaling_source"] != "all_outer_train_patches_only":
            return False, "fold %d: final scaler provenance not recorded" % f
        if scaler_meta(f)["n_outer_train_patients"] != len(tr):
            return False, "fold %d: recorded outer-train size differs" % f
    return True, "5 final scalers = all outer-train patches; distinct from inner and from test-contaminated scalers"


@check("B", "5 pos_weight comes from PATIENT counts, never patch counts")
def b5():
    for f in FOLDS:
        tr, _ = outer_sets(f)
        train_rows = inner_df(f)
        train_rows = train_rows[train_rows["assignment"] == "outer_train"]
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
            expect_inner = pos_weight_from_patients(COHORT.labels(sorted(tr - iva)))
            if abs(float(scaler_npz(f, j)["pos_weight"]) - expect_inner) > 1e-12:
                return False, "fold %d inner %d: pos_weight != inner-train patient ratio" % (f, j)
    return True, ("5 final + %d inner pos_weights equal their patient ratios; none equals a "
                  "patch ratio" % (5 * N_INNER))


# --------------------------- C  epoch selection, refit, fairness, no radiomics
@check("C", "7 the selected epoch is the argmax of the inner-CV curve alone")
def c7():
    for f in FOLDS:
        for m in MODELS:
            c = curve(m, f)
            inner_cols = [col for col in c.columns
                          if col.startswith("inner_") and col.endswith("_val_roc_auc")]
            if len(inner_cols) != N_INNER:
                return False, "fold %d %s: %d inner curves saved" % (f, m, len(inner_cols))
            mean = c[inner_cols].mean(axis=1).values
            if not np.allclose(mean, c["mean_val_roc_auc"].values, atol=1e-12):
                return False, "fold %d %s: stored mean curve != mean of the inner curves" % (f, m)
            expected = int(np.argmax(mean)) + 1       # argmax -> lowest epoch on ties
            ck = checkpoint(m, f)
            if int(ck["selected_epoch"]) != expected:
                return False, ("fold %d %s: checkpoint epoch %d != curve argmax %d"
                               % (f, m, ck["selected_epoch"], expected))
            if abs(float(ck["selection_mean_inner_cv_roc_auc"]) - float(mean.max())) > 1e-9:
                return False, "fold %d %s: recorded selection score differs" % (f, m)
            if len(c) != int(CFG["training"]["max_epochs"]):
                return False, "fold %d %s: curve has %d epochs" % (f, m, len(c))
            sel_col = preds(m).query("outer_fold == @f")["selected_epoch"].unique()
            if len(sel_col) != 1 or int(sel_col[0]) != expected:
                return False, "fold %d %s: prediction file records a different epoch" % (f, m)
    return True, ("10 selections = argmax of the %d-fold mean inner curve; every curve runs "
                  "the full %d epochs" % (N_INNER, int(CFG["training"]["max_epochs"])))


@check("C", "7b outer-test data reached nothing but the final score")
def c7b():
    for f in FOLDS:
        for m in MODELS:
            ck = checkpoint(m, f)
            if not 1 <= int(ck["selected_epoch"]) <= int(CFG["training"]["max_epochs"]):
                return False, "fold %d %s: implausible selected epoch" % (f, m)
    for fd in RUN["folds_detail"]:
        for m, mr in fd["models"].items():
            for key in ("selected_epoch", "selection_mean_inner_cv_roc_auc",
                        "inner_best_epoch_per_split", "fold_metrics"):
                if key not in mr:
                    return False, "run metadata incomplete for fold %s %s" % (fd["fold"], m)
            # the selection score must come from the inner CV, not the outer test
            if abs(float(mr["selection_mean_inner_cv_roc_auc"])
                   - float(mr["fold_metrics"]["roc_auc"])) < 1e-12:
                return False, ("fold %s %s: selection score equals the outer-test ROC-AUC"
                               % (fd["fold"], m))
    return True, "10 runs: selection metadata is inner-CV only; the outer test is used once, at scoring time"


@check("C", "10 predictions reproduce from the refit checkpoint + final scaler")
def c10():
    n = 0
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
                    n += 1
    return True, "all %d Phase-5 predictions reproduce bit-for-bit from the refit checkpoints" % n


@check("C", "11 both models used identical splits, preprocessing and settings")
def c11():
    for f in FOLDS:
        cks = {m: checkpoint(m, f) for m in MODELS}
        if not np.allclose(cks["mean_mil"]["scaler_mean"], cks["attention_mil"]["scaler_mean"]):
            return False, "fold %d: final scalers differ between models" % f
        if not np.allclose(cks["mean_mil"]["scaler_std"], cks["attention_mil"]["scaler_std"]):
            return False, "fold %d: final scaler std differs between models" % f
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
    s1 = {k: tuple(v.shape) for k, v in
          build_model("mean_mil", CFG, input_dim=74).state_dict().items()
          if k.startswith("encoder")}
    s2 = {k: tuple(v.shape) for k, v in
          build_model("attention_mil", CFG, input_dim=74).state_dict().items()
          if k.startswith("encoder")}
    if s1 != s2:
        return False, "the two patch encoders differ"
    return True, ("identical inner splits, scalers, pos_weights, seeds, settings and a "
                  "%d-tensor shared encoder; pooling is the only difference" % len(s1))


@check("C", "15 no radiomics extraction happens during Phase-5 training")
def c15():
    code = ("import sys; sys.path.insert(0, r'%s'); import train_mil_cv, roi_paired_compare; "
            "print('radiomics' in sys.modules or 'SimpleITK' in sys.modules "
            "or 'nibabel' in sys.modules)" % os.path.join(ROOT, "src"))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if out.returncode != 0:
        return False, "import failed: %s" % out.stderr[-200:]
    if out.stdout.strip() != "False":
        return False, "an image/radiomics module is imported by the training modules"
    for fname in ("train_mil_cv.py", "roi_paired_compare.py", "mil_data.py"):
        text = open(os.path.join(ROOT, "src", fname), "r", encoding="utf-8").read()
        for line in text.splitlines():
            s = line.strip()
            if (s.startswith("import ") or s.startswith("from ")) and any(
                    b in s for b in ("radiomics", "SimpleITK", "nibabel", "featureextractor")):
                return False, "%s imports %s" % (fname, s)
    if RUN.get("radiomics_imported_during_training") is not False:
        return False, "the run recorded a radiomics import"
    return True, "no radiomics/SimpleITK/nibabel import statically or at runtime in any Phase-5 module"


# ------------------------------------------ D  attention and numerical sanity
@check("D", "12/13 attention count = patch count, finite, sums to ~1")
def d12():
    att_dir = ppath(CFG, CFG["outputs"]["attention_dir"])
    files = sorted(f for f in os.listdir(att_dir) if f.endswith(".npz"))
    if len(files) != N_PATIENTS:
        return False, "%d attention files, expected %d" % (len(files), N_PATIENTS)
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
            if not np.array_equal(d["coords_world"], bag.coords_world):
                return False, "%s: world coordinates differ from the frozen bag" % pid
            if int(d["outer_fold"]) != int(stored.loc[pid, "outer_fold"]):
                return False, "%s: attention fold != prediction fold" % pid
            if abs(float(d["predicted_probability_adc"])
                   - float(stored.loc[pid, "predicted_probability_ADC"])) > 1e-12:
                return False, "%s: attention probability != prediction file" % pid
            if str(d["phase"]) != "5" or str(d["roi_name"]) != "gtv_rim":
                return False, "%s: attention file is not tagged phase 5 / gtv_rim" % pid
    if worst > 1e-5:
        return False, "max |sum - 1| = %.3g" % worst
    return True, ("%d files carry one weight per patch with the frozen coordinates; "
                  "max |sum - 1| = %.3g" % (len(files), worst))


@check("D", "14 no NaN/Inf anywhere in predictions, curves or metrics")
def d14():
    for m in MODELS:
        df = preds(m)
        p = df["predicted_probability_ADC"].values
        if not np.all(np.isfinite(p)) or p.min() < 0 or p.max() > 1:
            return False, "%s: invalid probabilities" % m
        if not np.array_equal(df["predicted_label"].values, (p >= THR).astype(int)):
            return False, "%s: predicted_label inconsistent with the 0.5 threshold" % m
        ft = pd.read_csv(os.path.join(ppath(CFG, CFG["outputs"]["results_dir"]),
                                      "%s_fold_metrics.csv" % m))
        if not np.all(np.isfinite(ft.select_dtypes(include=[np.number]).values)):
            return False, "%s: non-finite fold metrics" % m
        for f in FOLDS:
            c = curve(m, f)
            if not np.all(np.isfinite(c.select_dtypes(include=[np.number]).values)):
                return False, "%s fold %d: non-finite epoch-selection curve" % (m, f)
    return True, "%d probabilities, 10 fold-metric tables and 10 epoch curves are all finite" % (
        2 * N_PATIENTS)


@check("D", "14b pooled metrics recompute from the prediction files")
def d14b():
    res = json.load(open(os.path.join(ppath(CFG, CFG["outputs"]["results_dir"]),
                                      "oof_metrics.json"), "r", encoding="utf-8"))
    for m in MODELS:
        df = preds(m)
        ref = patient_metrics(df["true_label"].values,
                              df["predicted_probability_ADC"].values, THR)
        got = res["models"][m]["pooled_oof"]
        for k in ("roc_auc", "pr_auc", "balanced_accuracy", "mcc", "sensitivity_adc",
                  "specificity", "precision", "f1", "accuracy"):
            if abs(float(got[k]) - float(ref[k])) > 1e-12:
                return False, "%s: %s differs from the recomputed value" % (m, k)
        if got["n_patients"] != N_PATIENTS:
            return False, "%s: pooled metrics cover %d patients" % (m, got["n_patients"])
    return True, "pooled OOF metrics recompute exactly from the %d predictions per model" % N_PATIENTS


# ------------------------------- E  frozen inputs, earlier phases untouched
@check("E", "16 the frozen GTV+Rim bags are unchanged")
def e16():
    man = RUN["input_manifest"]
    feat_dir = ppath(CFG, CFG["phase3"]["features_dir"])
    now = {f: sha256_file(os.path.join(feat_dir, f))
           for f in sorted(os.listdir(feat_dir)) if f.endswith(".npz")}
    if len(now) != N_PATIENTS or now != man["bag_sha256"]:
        return False, "the frozen GTV+Rim bags changed during Phase 5"
    snap = pre_snapshot()
    changed = [p for p, h in snap.items()
               if p.startswith("patch_features/gtv_rim/")
               and (not os.path.exists(os.path.join(ROOT, *p.split("/")))
                    or sha256_file(os.path.join(ROOT, *p.split("/"))) != h)]
    if changed:
        return False, "%d GTV+Rim files changed: %s" % (len(changed), changed[:3])
    sig = CFG["phase3"]["expected_extraction_signature"]
    bad = [p for p, b in COHORT.bags.items() if b.config_signature != sig]
    if bad:
        return False, "extraction signature changed for %s" % bad[:5]
    if not RUN.get("frozen_inputs_unchanged"):
        return False, "the run itself reported changed inputs"
    return True, ("%d bags + %d sidecars byte-identical; signature %s unchanged"
                  % (N_PATIENTS, N_PATIENTS, sig))


@check("E", "17 the Gradient bags and every Phase-3/3B artefact are unchanged")
def e17():
    snap = pre_snapshot()
    watched = {p: h for p, h in snap.items()
               if not p.startswith("patch_features/gtv_rim/")}
    changed, missing = [], []
    for p, h in watched.items():
        full = os.path.join(ROOT, *p.split("/"))
        if not os.path.isfile(full):
            missing.append(p)
        elif sha256_file(full) != h:
            changed.append(p)
    if changed or missing:
        return False, "changed=%s missing=%s" % (changed[:5], missing[:5])
    # and the headline Phase-3B numbers still reproduce
    expect = {"mean_mil": 0.5894, "attention_mil": 0.5679}
    for m in MODELS:
        df = preds(m, CFG3B)
        if len(df) != 194:
            return False, "%s: Phase-3B prediction file changed size" % m
        auc = patient_metrics(df["true_label"].values,
                              df["predicted_probability_ADC"].values, THR)["roc_auc"]
        if abs(auc - expect[m]) > 1e-3:
            return False, "%s: Phase-3B pooled OOF ROC-AUC is now %.6f" % (m, auc)
    return True, ("%d Gradient / Phase-1-4 files byte-identical; Phase-3B pooled OOF ROC-AUC "
                  "still 0.5894 / 0.5679" % len(watched))


@check("E", "17b Phase 5 wrote only into its own namespaces")
def e17b():
    for d in (CFG["outputs"]["attention_dir"], CFG["outputs"]["results_dir"],
              CFG["outputs"]["inner_splits_dir"], CFG["outputs"]["preprocessing_dir"]):
        full = ppath(CFG, d)
        if not os.path.isdir(full):
            return False, "missing Phase-5 output directory %s" % d
        if not os.path.abspath(full).startswith(os.path.abspath(ROOT)):
            return False, "%s is outside the project root" % d
        for other in (CFG3B["outputs"]["attention_dir"], CFG3B["outputs"]["results_dir"],
                      CFG3B["outputs"]["inner_splits_dir"],
                      CFG3B["outputs"]["preprocessing_dir"]):
            if os.path.abspath(full) == os.path.abspath(ppath(CFG, other)):
                return False, "Phase 5 shares the directory %s with Phase 3B" % d
    for m in MODELS:
        tag = model_tag(CFG, m)
        if not tag.startswith("gtv_rim_"):
            return False, "%s is not in the GTV+Rim namespace" % tag
        if not os.path.isdir(os.path.join(ppath(CFG, CFG["outputs"]["models_dir"]), tag)):
            return False, "missing models/%s" % tag
        if not os.path.exists(os.path.join(ppath(CFG, CFG["outputs"]["predictions_dir"]),
                                           "%s_oof.csv" % tag)):
            return False, "missing predictions/%s_oof.csv" % tag
    return True, ("models/gtv_rim_*, predictions/gtv_rim_*_oof.csv, %s, %s, %s and %s are "
                  "disjoint from every Phase-3B path"
                  % (CFG["outputs"]["results_dir"], CFG["outputs"]["attention_dir"],
                     CFG["outputs"]["inner_splits_dir"], CFG["outputs"]["preprocessing_dir"]))


# --------------------------------------------------- F  paired ROI comparison
@check("F", "18/19 the paired comparison uses exactly the 194 shared patients, labels agree")
def f18():
    ov = pd.read_csv(ppath(CFG, CFG["roi_comparison"]["overlap_file"]))
    declared = sorted(ov.loc[ov["membership"] == "shared", "PatientID"].tolist())
    if len(declared) != N_SHARED:
        return False, "overlap report declares %d shared patients" % len(declared)
    new_ids = set(preds("mean_mil")["PatientID"])
    ref_ids = set(preds("mean_mil", CFG3B)["PatientID"])
    if sorted(new_ids & ref_ids) != declared:
        return False, "the intersection of the two prediction files is not the declared shared set"
    if not (ref_ids <= new_ids):
        return False, "the Gradient cohort is not a subset of the GTV+Rim cohort"
    new_lab = preds("mean_mil").set_index("PatientID").loc[declared, "true_label"]
    ref_lab = preds("mean_mil", CFG3B).set_index("PatientID").loc[declared, "true_label"]
    if not np.array_equal(new_lab.values, ref_lab.values):
        return False, "label disagreement between the two ROI experiments"
    bag_lab = np.array([COHORT.bags[p].label for p in declared])
    if not np.array_equal(bag_lab, new_lab.values):
        return False, "shared-set labels differ from the frozen bags"
    paired = pd.read_csv(ppath(CFG, CFG["roi_comparison"]["outputs"]["paired_csv"]))
    if set(paired["n_patients"].unique()) != {N_SHARED}:
        return False, "the paired CSV reports %s patients" % paired["n_patients"].unique()
    return True, ("%d shared patients (%d ADC / %d SCC); labels identical in bags, cohort and "
                  "both ROI prediction files" % (N_SHARED, int(bag_lab.sum()),
                                                 int((bag_lab == 0).sum())))


@check("F", "18b paired metrics recompute, and fold indices are never aligned")
def f18b():
    boot = json.load(open(ppath(CFG, CFG["roi_comparison"]["outputs"]["bootstrap_json"]),
                          "r", encoding="utf-8"))
    ov = pd.read_csv(ppath(CFG, CFG["roi_comparison"]["overlap_file"]))
    shared = sorted(ov.loc[ov["membership"] == "shared", "PatientID"].tolist())
    for m in MODELS:
        n = preds(m).set_index("PatientID").loc[shared]
        r = preds(m, CFG3B).set_index("PatientID").loc[shared]
        y = n["true_label"].values.astype(int)
        mn = patient_metrics(y, n["predicted_probability_ADC"].values, THR)
        mr = patient_metrics(y, r["predicted_probability_ADC"].values, THR)
        for k in ("roc_auc", "pr_auc", "balanced_accuracy", "mcc", "sensitivity_adc",
                  "specificity", "f1"):
            if abs(float(boot["models"][m]["gtv_rim_phase5"][k]) - mn[k]) > 1e-12:
                return False, "%s: stored GTV+Rim %s differs" % (m, k)
            if abs(float(boot["models"][m]["gradient_phase3b"][k]) - mr[k]) > 1e-12:
                return False, "%s: stored Gradient %s differs" % (m, k)
    # the reported fold-index agreement must NOT have been used: verify the two
    # splits genuinely disagree, so any fold-aligned comparison would be wrong.
    same_fold = int((ov.loc[ov["membership"] == "shared", "reference_fold"]
                     == ov.loc[ov["membership"] == "shared", "new_fold"]).sum())
    if same_fold == N_SHARED:
        return False, "the two outer splits are identical - check the split provenance"
    if "fold" in [c.lower() for c in pd.read_csv(
            ppath(CFG, CFG["roi_comparison"]["outputs"]["paired_csv"])).columns]:
        return False, "the paired CSV carries a fold column - fold indices must not be compared"
    return True, ("paired metrics recompute exactly; the two splits share a fold index for only "
                  "%d/%d patients and no fold-wise quantity is reported" % (same_fold, N_SHARED))


@check("F", "20 the bootstrap resamples PATIENTS only, and is reproducible")
def f20():
    boot = json.load(open(ppath(CFG, CFG["roi_comparison"]["outputs"]["bootstrap_json"]),
                          "r", encoding="utf-8"))
    rc = CFG["roi_comparison"]["bootstrap"]
    ov = pd.read_csv(ppath(CFG, CFG["roi_comparison"]["overlap_file"]))
    shared = sorted(ov.loc[ov["membership"] == "shared", "PatientID"].tolist())
    for m in MODELS:
        bs = boot["bootstrap"][m]
        if bs["resampling_unit"] != "patient" or not bs["paired"]:
            return False, "%s: bootstrap is not a paired patient-level resample" % m
        if bs["n_resamples_requested"] != int(rc["n_resamples"]):
            return False, "%s: %d resamples" % (m, bs["n_resamples_requested"])
        n = preds(m).set_index("PatientID").loc[shared]
        r = preds(m, CFG3B).set_index("PatientID").loc[shared]
        redo = bootstrap_differences(
            n["true_label"].values.astype(int),
            n["predicted_probability_ADC"].values.astype(float),
            r["predicted_probability_ADC"].values.astype(float),
            int(rc["n_resamples"]),
            int(rc["seed"]) + (0 if m == "mean_mil" else 1),
            float(rc["ci"]))
        for k in ("roc_auc", "pr_auc"):
            for f in ("difference_mean", "ci_low", "ci_high"):
                if abs(redo["metrics"][k][f] - float(bs["metrics"][k][f])) > 1e-12:
                    return False, "%s: %s %s not reproducible from the recorded seed" % (m, k, f)
            if bool(redo["metrics"][k]["ci_includes_zero"]) != bool(
                    bs["metrics"][k]["ci_includes_zero"]):
                return False, "%s: %s CI-includes-zero verdict differs" % (m, k)
    total_patches = sum(COHORT.bags[p].n_patches for p in shared)
    if boot["shared_patients"] != N_SHARED or boot["shared_patients"] == total_patches:
        return False, "the bootstrap unit looks like patches, not patients"
    return True, ("both bootstraps reproduce exactly from their recorded seeds; unit is the "
                  "patient (%d), never the patch (%d)" % (N_SHARED, total_patches))


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
           "phase": "5", "roi": "gtv_rim", "models": list(MODELS), "folds": list(FOLDS),
           "n_inner_splits": N_INNER, "threshold": THR, "n_patients": N_PATIENTS,
           "n_shared_patients": N_SHARED, "checks": results}
    path = ppath(CFG, "reports/phase5_tests.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\n%d checks, %d passed, %d failed  ->  %s"
          % (len(CHECKS), len(CHECKS) - n_fail, n_fail, path))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
