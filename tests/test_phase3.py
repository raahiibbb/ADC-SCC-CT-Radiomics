"""Phase-3 leakage and sanity checks: mean MIL vs gated attention MIL.

Run with:
    ./.venv/Scripts/python.exe tests/test_phase3.py

Every check re-derives its evidence from the files on disk (frozen bags, frozen
split, saved inner splits, saved scalers, saved checkpoints, saved predictions,
saved attention).  Nothing is taken on trust from the training log.

Groups
  K  split integrity and leakage (requirements 1, 2, 3, 4)
  L  preprocessing and class weighting provenance (5, 6)
  M  early stopping, model fairness, reproducibility (7, 8, 9)
  N  attention outputs (10, 11, 12)
  O  numerical sanity and frozen-input integrity (13, 14)
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

CFG = load_phase3_config()
MODELS = ("mean_mil", "attention_mil")
FOLDS = (1, 2, 3, 4, 5)
THR = float(CFG["evaluation"]["threshold"])
CHECKS = []

COHORT = load_cohort(CFG)
SPLIT = load_outer_split(CFG)
RUN = json.load(open(ppath(CFG, "reports/phase3_training_run.json"), "r", encoding="utf-8"))


def check(group, name):
    def deco(fn):
        CHECKS.append((group, name, fn))
        return fn
    return deco


def inner_df(fold: int) -> pd.DataFrame:
    return pd.read_csv(os.path.join(ppath(CFG, CFG["outputs"]["inner_splits_dir"]),
                                    "fold_%d.csv" % fold))


def preds(model: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(ppath(CFG, CFG["outputs"]["predictions_dir"]),
                                    "%s_oof.csv" % model))


def scaler_meta(fold: int) -> dict:
    return json.load(open(os.path.join(ppath(CFG, CFG["outputs"]["preprocessing_dir"]),
                                       "fold_%d_scaler.json" % fold), "r", encoding="utf-8"))


def scaler_npz(fold: int):
    return np.load(os.path.join(ppath(CFG, CFG["outputs"]["preprocessing_dir"]),
                                "fold_%d_scaler.npz" % fold), allow_pickle=True)


def checkpoint(model: str, fold: int) -> dict:
    return torch.load(os.path.join(ppath(CFG, CFG["outputs"]["models_dir"]), model,
                                   "fold_%d.pt" % fold), map_location="cpu",
                      weights_only=False)


# ------------------------------------------------------- K  splits and leakage
@check("K", "1 outer train/test never overlap")
def k1():
    for f in FOLDS:
        sub = SPLIT[SPLIT["fold"] == f]
        tr = set(sub.loc[sub["split"] == "train", "PatientID"])
        te = set(sub.loc[sub["split"] == "test", "PatientID"])
        if tr & te:
            return False, "fold %d overlap: %s" % (f, sorted(tr & te))
        if len(tr) + len(te) != 194:
            return False, "fold %d covers %d patients" % (f, len(tr) + len(te))
    test_union = [p for f in FOLDS for p in
                  SPLIT[(SPLIT["fold"] == f) & (SPLIT["split"] == "test")]["PatientID"]]
    if sorted(test_union) != sorted(COHORT.patient_ids):
        return False, "test folds do not partition the cohort"
    return True, "5 folds, 0 overlaps, test folds partition all 194 patients"


@check("K", "2 inner train/val and outer test never overlap")
def k2():
    for f in FOLDS:
        d = inner_df(f)
        sets = {a: set(d.loc[d["assignment"] == a, "PatientID"]) for a in
                ("inner_train", "inner_val", "outer_test")}
        for a, b in (("inner_train", "inner_val"), ("inner_train", "outer_test"),
                     ("inner_val", "outer_test")):
            if sets[a] & sets[b]:
                return False, "fold %d: %s & %s overlap" % (f, a, b)
        if len(d) != 194 or d["PatientID"].duplicated().any():
            return False, "fold %d: %d rows / duplicates" % (f, len(d))
        # inner train + inner val must reconstitute the frozen outer train set
        sub = SPLIT[(SPLIT["fold"] == f) & (SPLIT["split"] == "train")]
        if sets["inner_train"] | sets["inner_val"] != set(sub["PatientID"]):
            return False, "fold %d: inner split does not reconstitute outer train" % f
        if sets["outer_test"] != set(
                SPLIT[(SPLIT["fold"] == f) & (SPLIT["split"] == "test")]["PatientID"]):
            return False, "fold %d: outer test membership was altered" % f
        for a in ("inner_train", "inner_val"):
            y = COHORT.labels(sorted(sets[a]))
            if len(np.unique(y)) < 2:
                return False, "fold %d: %s is single-class" % (f, a)
    return True, "5 folds x 3 partitions disjoint; both classes present; outer test unaltered"


@check("K", "3 each patient has exactly one OOF prediction per model")
def k3():
    for m in MODELS:
        df = preds(m)
        if len(df) != 194 or df["PatientID"].duplicated().any():
            return False, "%s: %d rows, duplicates=%s" % (m, len(df),
                                                          df["PatientID"].duplicated().any())
        if sorted(df["PatientID"]) != sorted(COHORT.patient_ids):
            return False, "%s: patient set differs from the cohort" % m
        for _, r in df.iterrows():
            fold_rows = SPLIT[(SPLIT["PatientID"] == r["PatientID"])
                              & (SPLIT["fold"] == r["outer_fold"])]
            if fold_rows.iloc[0]["split"] != "test":
                return False, "%s: %s predicted in a fold where it was not held out" % (
                    m, r["PatientID"])
    return True, "194 unique predictions per model, each from its own held-out fold"


@check("K", "4 labels agree with the frozen bags and cohort metadata")
def k4():
    meta = pd.read_csv(ppath(CFG, CFG["phase3"]["cohort_file"]))
    meta = meta.set_index("PatientID")
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
            if str(df.loc[pid, "histology"]) != bag.histology:
                return False, "%s: %s histology differs from the bag" % (m, pid)
    for f in FOLDS:
        sub = SPLIT[SPLIT["fold"] == f]
        for _, r in sub.iterrows():
            if int(r["label"]) != COHORT.bags[r["PatientID"]].label:
                return False, "split file label differs from the bag for %s" % r["PatientID"]
    n_adc = sum(1 for b in COHORT.bags.values() if b.label == 1)
    return True, "194 bags, split file, cohort CSV and both prediction files agree (%d ADC)" % n_adc


# ------------------------------------------- L  preprocessing and class weights
@check("L", "5 scaler statistics come from inner-training patients only")
def l5():
    for f in FOLDS:
        d = inner_df(f)
        itr = sorted(d.loc[d["assignment"] == "inner_train", "PatientID"])
        iva = sorted(d.loc[d["assignment"] == "inner_val", "PatientID"])
        ote = sorted(d.loc[d["assignment"] == "outer_test", "PatientID"])
        saved = scaler_npz(f)
        ref = fit_fold_scaler(CFG, COHORT, itr)
        if not (np.allclose(saved["mean"], ref.mean, rtol=0, atol=1e-9)
                and np.allclose(saved["std"], ref.std, rtol=0, atol=1e-9)):
            return False, "fold %d: saved scaler != scaler refit on inner train" % f
        # and it must NOT match a scaler that saw validation or test patients
        for name, extra in (("+inner_val", iva), ("+outer_test", ote)):
            other = fit_fold_scaler(CFG, COHORT, sorted(itr + extra))
            if np.allclose(saved["mean"], other.mean, rtol=0, atol=1e-9):
                return False, "fold %d: scaler is indistinguishable from one fitted %s" % (f, name)
        meta = scaler_meta(f)
        if meta["scaling_source"] != "inner_train_patches_only":
            return False, "fold %d: scaler provenance not recorded" % f
        n_patches = sum(COHORT.bags[p].n_patches for p in itr)
        if meta["n_inner_train_patches"] != n_patches:
            return False, "fold %d: recorded inner-train patch count is wrong" % f
    return True, "5 folds: scalers reproduce exactly from inner-train patches and only from them"


@check("L", "5b every fold's scaler is applied to val and test unchanged")
def l5b():
    for f in FOLDS:
        saved = scaler_npz(f)
        for m in MODELS:
            ck = checkpoint(m, f)
            if not (np.allclose(ck["scaler_mean"], saved["mean"])
                    and np.allclose(ck["scaler_std"], saved["std"])):
                return False, "fold %d %s: checkpoint scaler differs from the saved scaler" % (f, m)
            if list(ck["kept_features"]) != [str(x) for x in saved["kept_features"]]:
                return False, "fold %d %s: feature selection differs" % (f, m)
    return True, "both models carry the identical fold scaler and feature list"


@check("L", "6 pos_weight uses patient counts, not patch counts")
def l6():
    for f in FOLDS:
        d = inner_df(f)
        itr = sorted(d.loc[d["assignment"] == "inner_train", "PatientID"])
        y = COHORT.labels(itr)
        expect = pos_weight_from_patients(y)
        patch_based = (sum(COHORT.bags[p].n_patches for p in itr if COHORT.bags[p].label == 0)
                       / sum(COHORT.bags[p].n_patches for p in itr if COHORT.bags[p].label == 1))
        saved = float(scaler_npz(f)["pos_weight"])
        if abs(saved - expect) > 1e-12:
            return False, "fold %d: pos_weight %.6f != patient ratio %.6f" % (f, saved, expect)
        if abs(saved - patch_based) < 1e-6:
            return False, "fold %d: pos_weight is indistinguishable from the patch ratio" % f
        for m in MODELS:
            if abs(float(checkpoint(m, f)["pos_weight"]) - expect) > 1e-12:
                return False, "fold %d %s: checkpoint pos_weight differs" % (f, m)
    return True, "5 folds: pos_weight = N_SCC_patients / N_ADC_patients, distinct from the patch ratio"


# ------------------------------ M  early stopping, fairness, no radiomics in training
@check("M", "7 outer-test bags never participate in early stopping")
def m7():
    for f in FOLDS:
        d = inner_df(f)
        iva = sorted(d.loc[d["assignment"] == "inner_val", "PatientID"])
        ote = set(d.loc[d["assignment"] == "outer_test", "PatientID"])
        if set(iva) & ote:
            return False, "fold %d: test patients are inside the validation set" % f
        saved = scaler_npz(f)
        keep = saved["keep_idx"]
        for m in MODELS:
            ck = checkpoint(m, f)
            hist = json.load(open(os.path.join(ppath(CFG, CFG["outputs"]["models_dir"]), m,
                                               "fold_%d_history.json" % f), "r",
                                  encoding="utf-8"))
            best = max(e["val_roc_auc"] for e in hist["history"])
            if abs(best - ck["best_val_roc_auc"]) > 1e-12:
                return False, "fold %d %s: checkpoint is not the best-validation epoch" % (f, m)
            # the selected checkpoint must reproduce exactly that validation ROC-AUC
            model = build_model(m, CFG, input_dim=int(ck["input_dim"]))
            model.load_state_dict(ck["model_state_dict"])
            model.eval()
            probs = []
            with torch.no_grad():
                for pid in iva:
                    x = COHORT.bags[pid].features[:, keep]
                    x = (x - saved["mean"]) / saved["std"]
                    logit, _ = model(torch.from_numpy(x.astype(np.float32)))
                    probs.append(float(torch.sigmoid(logit).item()))
            auc = patient_metrics(COHORT.labels(iva), probs, THR)["roc_auc"]
            if abs(auc - ck["best_val_roc_auc"]) > 1e-9:
                return False, ("fold %d %s: recomputed val ROC-AUC %.6f != stored %.6f"
                               % (f, m, auc, ck["best_val_roc_auc"]))
    return True, "checkpoint = best-validation epoch, reproduced from validation patients alone"


@check("M", "7b saved test predictions reproduce from checkpoint + fold scaler")
def m7b():
    for f in FOLDS:
        saved = scaler_npz(f)
        keep = saved["keep_idx"]
        d = inner_df(f)
        ote = sorted(d.loc[d["assignment"] == "outer_test", "PatientID"])
        for m in MODELS:
            ck = checkpoint(m, f)
            model = build_model(m, CFG, input_dim=int(ck["input_dim"]))
            model.load_state_dict(ck["model_state_dict"])
            model.eval()
            stored = preds(m).set_index("PatientID")
            with torch.no_grad():
                for pid in ote:
                    x = COHORT.bags[pid].features[:, keep]
                    x = (x - saved["mean"]) / saved["std"]
                    logit, _ = model(torch.from_numpy(x.astype(np.float32)))
                    p = float(torch.sigmoid(logit).item())
                    if abs(p - float(stored.loc[pid, "predicted_probability_ADC"])) > 1e-9:
                        return False, "fold %d %s: %s prediction not reproducible" % (f, m, pid)
    return True, "all 388 out-of-fold predictions reproduce bit-for-bit from the checkpoints"


@check("M", "8 both models use identical folds, splits and preprocessing")
def m8():
    for f in FOLDS:
        fps = {m: checkpoint(m, f)["fold_fingerprint"] for m in MODELS}
        if len(set(fps.values())) != 1:
            return False, "fold %d: fold fingerprints differ %s" % (f, fps)
        meta_fp = scaler_meta(f)["fold_fingerprint"]
        if meta_fp != fps["mean_mil"]:
            return False, "fold %d: saved preprocessing fingerprint differs" % f
        cks = {m: checkpoint(m, f) for m in MODELS}
        if cks["mean_mil"]["input_dim"] != cks["attention_mil"]["input_dim"]:
            return False, "fold %d: encoder input dimensions differ" % f
        cfgs = [json.dumps(cks[m]["config"]["training"], sort_keys=True) for m in MODELS]
        if len(set(cfgs)) != 1:
            return False, "fold %d: training settings differ between models" % f
    a, b = (preds(m).sort_values("PatientID").reset_index(drop=True) for m in MODELS)
    if not (a["outer_fold"].equals(b["outer_fold"]) and a["true_label"].equals(b["true_label"])):
        return False, "fold assignment or labels differ between the two prediction files"
    # the encoders must be structurally identical (same shapes for shared parameters)
    m1 = build_model("mean_mil", CFG, input_dim=74)
    m2 = build_model("attention_mil", CFG, input_dim=74)
    s1 = {k: tuple(v.shape) for k, v in m1.state_dict().items() if k.startswith("encoder")}
    s2 = {k: tuple(v.shape) for k, v in m2.state_dict().items() if k.startswith("encoder")}
    if s1 != s2:
        return False, "patch encoders differ: %s vs %s" % (s1, s2)
    return True, "identical fingerprints, identical encoder (%d shared tensors), identical settings" % len(s1)


@check("M", "9 no radiomics extraction happens during model training")
def m9():
    code = ("import sys; sys.path.insert(0, r'%s'); import train_mil; "
            "print('radiomics' in sys.modules or 'SimpleITK' in sys.modules "
            "or 'nibabel' in sys.modules)" % os.path.join(ROOT, "src"))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if out.returncode != 0:
        return False, "import failed: %s" % out.stderr[-200:]
    if out.stdout.strip() != "False":
        return False, "an image/radiomics module is imported by the training module"
    banned = ("radiomics", "SimpleITK", "nibabel", "featureextractor")
    for src in ("train_mil.py", "mil_data.py", "mil_models.py", "mil_metrics.py",
                "evaluate_mil.py"):
        text = open(os.path.join(ROOT, "src", src), "r", encoding="utf-8").read()
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("import ") or s.startswith("from "):
                if any(b in s for b in banned):
                    return False, "%s imports %s" % (src, s)
    if RUN.get("radiomics_imported_during_training") is not False:
        return False, "the training run recorded a radiomics import"
    return True, "no radiomics/SimpleITK/nibabel import in 5 Phase-3 modules or at runtime"


@check("M", "9b seeds and runtime are recorded")
def m9b():
    for key in ("python", "numpy", "torch", "train_base_seed"):
        if key not in RUN["config"]["seeds"]:
            return False, "seed %s not recorded" % key
    for f in FOLDS:
        if "inner_seed" not in scaler_meta(f):
            return False, "fold %d inner seed not recorded" % f
        for m in MODELS:
            if "seed" not in checkpoint(m, f):
                return False, "fold %d %s seed not recorded" % (f, m)
    if not RUN.get("total_runtime_seconds"):
        return False, "runtime not recorded"
    return True, ("python/numpy/torch/train_base/inner seeds recorded; runtime %.1f s"
                  % RUN["total_runtime_seconds"])


# ------------------------------------------------------------- N  attention output
@check("N", "10 attention length equals the patch count for every test patient")
def n10():
    att_dir = ppath(CFG, CFG["outputs"]["attention_dir"])
    files = sorted(f for f in os.listdir(att_dir) if f.endswith(".npz"))
    if len(files) != 194:
        return False, "%d attention files, expected 194" % len(files)
    for fn in files:
        pid = fn[:-4]
        with np.load(os.path.join(att_dir, fn), allow_pickle=True) as d:
            bag = COHORT.bags[pid]
            if d["attention"].shape[0] != bag.n_patches:
                return False, "%s: %d weights vs %d patches" % (pid, d["attention"].shape[0],
                                                                bag.n_patches)
            for key, ref in (("coords_index", bag.coords_index),
                             ("coords_world", bag.coords_world)):
                if d[key].shape != ref.shape or not np.array_equal(d[key], ref):
                    return False, "%s: %s differs from the frozen bag" % (pid, key)
            if int(d["outer_fold"]) != int(
                    preds("attention_mil").set_index("PatientID").loc[pid, "outer_fold"]):
                return False, "%s: attention fold != prediction fold" % pid
            if int(d["true_label"]) != bag.label:
                return False, "%s: attention label differs from the bag" % pid
            stored_p = float(preds("attention_mil").set_index("PatientID").loc[
                pid, "predicted_probability_ADC"])
            if abs(float(d["predicted_probability_adc"]) - stored_p) > 1e-12:
                return False, "%s: attention probability differs from the prediction file" % pid
            if int(d["patch_index"].shape[0]) != bag.n_patches:
                return False, "%s: patch identifiers missing" % pid
    return True, "194 files: weights, coords, fold, label and probability all consistent"


@check("N", "11 attention weights are finite")
def n11():
    att_dir = ppath(CFG, CFG["outputs"]["attention_dir"])
    lo, hi = 1.0, 0.0
    for fn in sorted(os.listdir(att_dir)):
        if not fn.endswith(".npz"):
            continue
        with np.load(os.path.join(att_dir, fn), allow_pickle=True) as d:
            a = d["attention"]
            if not np.all(np.isfinite(a)):
                return False, "%s: non-finite attention" % fn
            if np.any(a < 0):
                return False, "%s: negative attention" % fn
            lo, hi = min(lo, float(a.min())), max(hi, float(a.max()))
    return True, "all weights finite and non-negative (min %.3g, max %.3g)" % (lo, hi)


@check("N", "12 attention weights sum to ~1 within each bag")
def n12():
    att_dir = ppath(CFG, CFG["outputs"]["attention_dir"])
    worst, worst_pid = 0.0, ""
    for fn in sorted(os.listdir(att_dir)):
        if not fn.endswith(".npz"):
            continue
        with np.load(os.path.join(att_dir, fn), allow_pickle=True) as d:
            s = float(d["attention"].sum())
            if abs(s - 1.0) > worst:
                worst, worst_pid = abs(s - 1.0), fn[:-4]
    if worst > 1e-5:
        return False, "%s deviates by %.3g from 1" % (worst_pid, worst)
    return True, "194 bags, max |sum - 1| = %.3g (%s)" % (worst, worst_pid)


# --------------------------------------- O  numerical sanity and frozen integrity
@check("O", "13 no NaN/Inf in predictions or metrics")
def o13():
    for m in MODELS:
        df = preds(m)
        p = df["predicted_probability_ADC"].values
        if not np.all(np.isfinite(p)):
            return False, "%s: non-finite probabilities" % m
        if p.min() < 0 or p.max() > 1:
            return False, "%s: probabilities outside [0,1]" % m
        if not np.array_equal(df["predicted_label"].values, (p >= THR).astype(int)):
            return False, "%s: predicted_label inconsistent with the 0.5 threshold" % m
        ft = pd.read_csv(os.path.join(ppath(CFG, CFG["outputs"]["results_dir"]),
                                      "%s_fold_metrics.csv" % m))
        num = ft.select_dtypes(include=[np.number])
        if not np.all(np.isfinite(num.values)):
            return False, "%s: non-finite fold metrics" % m
    res = json.load(open(os.path.join(ppath(CFG, CFG["outputs"]["results_dir"]),
                                      "oof_metrics.json"), "r", encoding="utf-8"))

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                bad = walk(v, path + "/" + str(k))
                if bad:
                    return bad
        elif isinstance(node, list):
            for i, v in enumerate(node):
                bad = walk(v, path + "[%d]" % i)
                if bad:
                    return bad
        elif isinstance(node, float) and not np.isfinite(node):
            return path
        return None

    bad = walk(res)
    if bad:
        return False, "non-finite metric at %s" % bad
    return True, "388 probabilities, 10 fold-metric tables and oof_metrics.json all finite"


@check("O", "14 the frozen bags and split are unchanged")
def o14():
    man = RUN["input_manifest"]
    feat_dir = ppath(CFG, CFG["phase3"]["features_dir"])
    now = {f: sha256_file(os.path.join(feat_dir, f))
           for f in sorted(os.listdir(feat_dir)) if f.endswith(".npz")}
    if now != man["bag_sha256"]:
        changed = [k for k in set(now) | set(man["bag_sha256"])
                   if now.get(k) != man["bag_sha256"].get(k)]
        return False, "bags changed: %s" % changed[:5]
    if len(now) != 194:
        return False, "%d bags on disk" % len(now)
    split_sha = sha256_file(ppath(CFG, CFG["phase3"]["splits_file"]))
    if split_sha != man["splits_file_sha256"]:
        return False, "split file changed during the run"
    meta = json.load(open(ppath(CFG, CFG["phase3"]["splits_meta_file"]), "r", encoding="utf-8"))
    recorded = meta.get("sha256") or meta.get("split_sha256") or meta.get("file_sha256")
    if recorded and recorded != split_sha:
        return False, "split sha256 differs from splits/stratified_5fold_meta.json"
    sig = CFG["phase3"]["expected_extraction_signature"]
    bad = [p for p, b in COHORT.bags.items() if b.config_signature != sig]
    if bad:
        return False, "extraction signature changed for %s" % bad[:5]
    if not RUN.get("frozen_inputs_unchanged"):
        return False, "the training run itself reported changed inputs"
    return True, "194 bag hashes, split sha256 and signature %s all unchanged" % sig


@check("O", "14b five small bags (<20 patches) are present, not excluded")
def o14b():
    small = sorted(p for p, b in COHORT.bags.items()
                   if b.n_patches < int(CFG["evaluation"]["small_bag_threshold"]))
    if len(small) != 5:
        return False, "%d small bags found: %s" % (len(small), small)
    path = os.path.join(ppath(CFG, CFG["outputs"]["results_dir"]), "small_bag_predictions.csv")
    df = pd.read_csv(path)
    for m in MODELS:
        got = sorted(df[df["model"] == m]["PatientID"].tolist())
        if got != small:
            return False, "%s: small-bag report lists %s" % (m, got)
        if not set(small).issubset(set(preds(m)["PatientID"])):
            return False, "%s: a small bag is missing from the predictions" % m
    return True, "%s all trained on, predicted and reported" % ", ".join(small)


@check("O", "14c evaluation metrics reproduce from the prediction files")
def o14c():
    res = json.load(open(os.path.join(ppath(CFG, CFG["outputs"]["results_dir"]),
                                      "oof_metrics.json"), "r", encoding="utf-8"))
    for m in MODELS:
        df = preds(m)
        ref = patient_metrics(df["true_label"].values,
                              df["predicted_probability_ADC"].values, THR)
        got = res["models"][m]["pooled_oof"]
        for k in ("roc_auc", "pr_auc", "balanced_accuracy", "mcc", "sensitivity_adc",
                  "specificity", "precision", "f1"):
            if abs(float(got[k]) - float(ref[k])) > 1e-12:
                return False, "%s: %s %.6f != recomputed %.6f" % (m, k, got[k], ref[k])
        if got["n_patients"] != 194:
            return False, "%s: pooled metrics cover %d patients" % (m, got["n_patients"])
    return True, "pooled OOF metrics for both models recompute exactly from the 194 predictions"


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
           "phase": 3, "models": list(MODELS), "folds": list(FOLDS), "threshold": THR,
           "extraction_signature": CFG["phase3"]["expected_extraction_signature"],
           "checks": results}
    path = ppath(CFG, "reports/phase3_tests.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\n%d checks, %d passed, %d failed  ->  %s"
          % (len(CHECKS), len(CHECKS) - n_fail, n_fail, path))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
