"""Phase 3B - inner-CV epoch selection, then a from-scratch refit per outer fold.

Only the MODEL-SELECTION procedure differs from Phase 3.  Bags, outer split,
features, both architectures, loss, optimizer, dropout, learning rate, weight
decay, epoch budget, class weighting rule and the 0.5 threshold are unchanged and
are re-used from `src/mil_data.py` / `src/mil_models.py`.

Per outer fold, per model:

  1. outer-test patients are untouched;
  2. stratified 3-fold inner CV over the outer-TRAIN patients (seed 3000+fold);
  3. each inner run trains from scratch for the full 100-epoch budget on its
     inner-training split, scaled by a scaler fitted on that split alone, and
     records patient-level ROC-AUC on its inner-validation split at every epoch;
  4. the per-epoch ROC-AUC is averaged over the 3 inner folds;
  5. the epoch with the best mean inner-CV ROC-AUC is selected (lowest epoch wins ties);
  6. no outer-test information enters any of the above;
  7. the final scaler is fitted on ALL outer-training patches;
  8. pos_weight is recomputed from ALL outer-training patient counts;
  9. the model is retrained from scratch on all outer-training patients for
     exactly the selected number of epochs;
 10. it is evaluated once on the outer-test fold.

Usage:
    python src/train_mil_cv.py                    # both models, folds 1..5
    python src/train_mil_cv.py --folds 1 --models mean_mil
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mil_data import (Cohort, FoldScaler, fit_fold_scaler, load_cohort,  # noqa: E402
                      load_outer_split, load_phase3_config, model_tag,
                      outer_fold_patients, pos_weight_from_patients, ppath,
                      sha256_file)
from mil_metrics import patient_metrics  # noqa: E402
from mil_models import build_model  # noqa: E402
from train_mil import predict, train_one_epoch  # noqa: E402

MODELS = ("mean_mil", "attention_mil")
DEFAULT_CONFIG = "config/phase3b.yaml"


def phase_tag(cfg: dict) -> str:
    """Which phase this run belongs to; defaults to 3b so Phase 3B is unchanged."""
    return str(cfg["outputs"].get("phase_tag", "3b"))


# --------------------------------------------------------------------- helpers
@dataclass
class Split:
    """A trainable train/eval pair sharing one scaler and one pos_weight."""
    train: List[str]
    evaluate: List[str]
    scaler: FoldScaler
    pos_weight: float
    tensors: Dict[str, np.ndarray]


def make_split(cfg: dict, cohort: Cohort, train_pids: Sequence[str],
               eval_pids: Sequence[str]) -> Split:
    """Scaler and pos_weight are derived from `train_pids` ONLY."""
    scaler = fit_fold_scaler(cfg, cohort, train_pids)
    pw = pos_weight_from_patients(cohort.labels(train_pids))
    tensors = {p: scaler.transform(cohort.bags[p].features)
               for p in list(train_pids) + list(eval_pids)}
    return Split(train=list(train_pids), evaluate=list(eval_pids), scaler=scaler,
                 pos_weight=pw, tensors=tensors)


class _BagSource:
    """Adapter so train_one_epoch/predict can consume a Split unchanged."""

    def __init__(self, split: Split):
        self.tensors = split.tensors


def set_seeds(cfg: dict, seed: int) -> int:
    scfg = cfg["seeds"]
    random.seed(int(scfg["python"]) + seed)
    np.random.seed(int(scfg["numpy"]) + seed)
    torch.manual_seed(int(scfg["torch"]) + seed)
    if bool(scfg.get("deterministic_torch", True)):
        torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    return seed


def run_training(cfg: dict, cohort: Cohort, split: Split, model_name: str, seed: int,
                 n_epochs: int, eval_every_epoch: bool):
    """Train from scratch for exactly `n_epochs`; optionally score the eval set each epoch."""
    tcfg = cfg["training"]
    device = torch.device(str(tcfg.get("device", "cpu")))
    set_seeds(cfg, seed)

    model = build_model(model_name, cfg, input_dim=len(split.scaler.kept_features)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tcfg["learning_rate"]),
                                 weight_decay=float(tcfg["weight_decay"]))
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([split.pos_weight], dtype=torch.float32, device=device))
    source = _BagSource(split)
    rng = random.Random(seed)
    y_eval = cohort.labels(split.evaluate) if eval_every_epoch else None
    curve: List[dict] = []

    for epoch in range(1, int(n_epochs) + 1):
        order = list(split.train)
        rng.shuffle(order)
        loss = train_one_epoch(model, source, cohort, order, optimizer, loss_fn,
                               device, int(tcfg["bags_per_batch"]))
        row = {"epoch": epoch, "train_loss": float(loss)}
        if eval_every_epoch:
            prob, _ = predict(model, source, split.evaluate, device)
            if not np.all(np.isfinite(prob)):
                raise RuntimeError("non-finite validation probabilities at epoch %d" % epoch)
            if len(np.unique(y_eval)) < 2:
                raise RuntimeError("split-generation failure: single-class inner validation set")
            m = patient_metrics(y_eval, prob, float(cfg["evaluation"]["threshold"]))
            row["val_roc_auc"] = m["roc_auc"]
            row["val_pr_auc"] = m["pr_auc"]
        curve.append(row)
    return model, curve, source


# ---------------------------------------------------------------- one outer fold
def inner_cv_assignment(cfg: dict, cohort: Cohort, fold: int,
                        train_pids: Sequence[str]) -> Dict[str, int]:
    """Stratified K-fold over the outer-training patients; deterministic seed."""
    icfg = cfg["inner_cv"]
    pids = sorted(train_pids)
    y = cohort.labels(pids)
    skf = StratifiedKFold(n_splits=int(icfg["n_splits"]), shuffle=bool(icfg["shuffle"]),
                          random_state=int(icfg["base_seed"]) + int(fold))
    assign: Dict[str, int] = {}
    for j, (_, val_idx) in enumerate(skf.split(np.zeros(len(pids)), y), start=1):
        for i in val_idx:
            assign[pids[i]] = j
    for j in range(1, int(icfg["n_splits"]) + 1):
        members = [p for p, k in assign.items() if k == j]
        yy = cohort.labels(members)
        if len(np.unique(yy)) < 2:
            raise RuntimeError("split-generation failure: inner fold %d of outer fold %d "
                               "is single-class" % (j, fold))
    return assign


def select_epoch(curves: List[List[dict]], tie_break: str = "lowest_epoch"):
    """Mean inner-CV validation ROC-AUC per epoch -> selected epoch."""
    n_epochs = min(len(c) for c in curves)
    aucs = np.array([[c[e]["val_roc_auc"] for e in range(n_epochs)] for c in curves])
    mean = aucs.mean(axis=0)
    sd = aucs.std(axis=0, ddof=1) if aucs.shape[0] > 1 else np.zeros(n_epochs)
    if not np.all(np.isfinite(mean)):
        raise RuntimeError("non-finite epoch-selection curve")
    if tie_break != "lowest_epoch":
        raise ValueError("unsupported tie_break %r" % tie_break)
    idx = int(np.argmax(mean))          # np.argmax returns the FIRST maximum -> lowest epoch
    return idx + 1, float(mean[idx]), mean, sd, aucs


def process_fold(cfg: dict, cohort: Cohort, split_df: pd.DataFrame, fold: int,
                 models: Sequence[str], log) -> dict:
    train_pids, test_pids = outer_fold_patients(split_df, fold)
    assign = inner_cv_assignment(cfg, cohort, fold, train_pids)
    n_inner = int(cfg["inner_cv"]["n_splits"])
    y_tr = cohort.labels(train_pids)

    log("-" * 78)
    log("fold %d: outer_train %d (%d ADC/%d SCC), outer_test %d | inner CV %d-fold, seed %d"
        % (fold, len(train_pids), int((y_tr == 1).sum()), int((y_tr == 0).sum()),
           len(test_pids), n_inner, int(cfg["inner_cv"]["base_seed"]) + fold))

    # ---- inner splits, shared by both models (built once, identical inputs)
    inner_splits = {}
    for j in range(1, n_inner + 1):
        itr = sorted(p for p in train_pids if assign[p] != j)
        iva = sorted(p for p in train_pids if assign[p] == j)
        inner_splits[j] = make_split(cfg, cohort, itr, iva)
        yv = cohort.labels(iva)
        log("  inner fold %d: train %d, val %d (%d ADC/%d SCC), pos_weight %.4f"
            % (j, len(itr), len(iva), int((yv == 1).sum()), int((yv == 0).sum()),
               inner_splits[j].pos_weight))

    # ---- final split: scaler and pos_weight from ALL outer-training patients
    final_split = make_split(cfg, cohort, sorted(train_pids), sorted(test_pids))
    log("  final refit: scaler + pos_weight from all %d outer-train patients (pw %.4f)"
        % (len(train_pids), final_split.pos_weight))

    save_fold_splits(cfg, cohort, fold, assign, train_pids, test_pids,
                     inner_splits, final_split)

    out = {"fold": fold, "n_outer_train": len(train_pids), "n_outer_test": len(test_pids),
           "inner_cv_seed": int(cfg["inner_cv"]["base_seed"]) + fold,
           "final_pos_weight": final_split.pos_weight,
           "inner_pos_weight": {j: inner_splits[j].pos_weight for j in inner_splits},
           "models": {}}

    for model_name in models:
        offset = 0 if model_name == "mean_mil" else 1
        t0 = time.time()
        curves = []
        for j in range(1, n_inner + 1):
            seed = (int(cfg["seeds"]["inner_base_seed"]) + 100 * fold + 10 * j + offset)
            _, curve, _ = run_training(cfg, cohort, inner_splits[j], model_name, seed,
                                       int(cfg["training"]["max_epochs"]),
                                       eval_every_epoch=True)
            curves.append(curve)
            best_j = max(r["val_roc_auc"] for r in curve)
            log("    %s inner %d: best epoch-wise val ROC-AUC %.4f (seed %d)"
                % (model_name, j, best_j, seed))
        sel_epoch, sel_auc, mean_curve, sd_curve, all_aucs = select_epoch(
            curves, str(cfg["inner_cv"]["tie_break"]))
        selection_seconds = time.time() - t0
        save_epoch_curve(cfg, fold, model_name, mean_curve, sd_curve, all_aucs, curves)
        log("    %s -> selected epoch %d (mean inner-CV ROC-AUC %.4f, sd %.4f)"
            % (model_name, sel_epoch, sel_auc, float(sd_curve[sel_epoch - 1])))

        # ---- final refit on ALL outer-training patients, then one test evaluation
        t1 = time.time()
        final_seed = int(cfg["seeds"]["final_base_seed"]) + 10 * fold + offset
        model, _, source = run_training(cfg, cohort, final_split, model_name, final_seed,
                                        sel_epoch, eval_every_epoch=False)
        want_att = model_name == "attention_mil"
        device = torch.device(str(cfg["training"]["device"]))
        test_prob, test_att = predict(model, source, final_split.evaluate, device,
                                      want_attention=want_att)
        refit_seconds = time.time() - t1
        if not np.all(np.isfinite(test_prob)):
            raise RuntimeError("non-finite test probabilities, fold %d, %s" % (fold, model_name))
        y_test = cohort.labels(final_split.evaluate)
        fm = patient_metrics(y_test, test_prob, float(cfg["evaluation"]["threshold"]))
        log("    %s fold %d: test ROC-AUC %.4f  PR-AUC %.4f  balacc %.4f  MCC %.4f "
            "(select %.1f s, refit %.1f s)"
            % (model_name, fold, fm["roc_auc"], fm["pr_auc"], fm["balanced_accuracy"],
               fm["mcc"], selection_seconds, refit_seconds))

        ck_dir = ppath(cfg, "%s/%s" % (cfg["outputs"]["models_dir"],
                                       model_tag(cfg, model_name)))
        os.makedirs(ck_dir, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_name": model_name, "fold": fold, "phase": phase_tag(cfg),
            "roi": str(cfg["phase3"]["roi"]),
            "input_dim": len(final_split.scaler.kept_features),
            "kept_features": final_split.scaler.kept_features,
            "scaler_mean": final_split.scaler.mean, "scaler_std": final_split.scaler.std,
            "selected_epoch": sel_epoch, "selection_mean_inner_cv_roc_auc": sel_auc,
            "pos_weight": final_split.pos_weight, "seed": final_seed,
            "inner_cv_seed": int(cfg["inner_cv"]["base_seed"]) + fold,
            "config": {"model": cfg["model"], "training": cfg["training"],
                       "inner_cv": cfg["inner_cv"], "seeds": cfg["seeds"]},
        }, os.path.join(ck_dir, "fold_%d.pt" % fold))

        if want_att:
            save_attention(cfg, cohort, fold, final_split.evaluate, test_att, test_prob)

        out["models"][model_name] = {
            "selected_epoch": sel_epoch,
            "selection_mean_inner_cv_roc_auc": sel_auc,
            "selection_sd_at_selected_epoch": float(sd_curve[sel_epoch - 1]),
            "inner_best_epoch_per_split": [int(np.argmax([r["val_roc_auc"] for r in c])) + 1
                                           for c in curves],
            "final_seed": final_seed,
            "selection_seconds": selection_seconds,
            "refit_seconds": refit_seconds,
            "fold_metrics": fm,
            "test_probabilities": {p: float(v)
                                   for p, v in zip(final_split.evaluate, test_prob)},
        }
    return out


# ------------------------------------------------------------------ persistence
def save_fold_splits(cfg: dict, cohort: Cohort, fold: int, assign: Dict[str, int],
                     train_pids: Sequence[str], test_pids: Sequence[str],
                     inner_splits: Dict[int, Split], final_split: Split) -> None:
    inner_dir = ppath(cfg, cfg["outputs"]["inner_splits_dir"])
    prep_dir = ppath(cfg, cfg["outputs"]["preprocessing_dir"])
    os.makedirs(inner_dir, exist_ok=True)
    os.makedirs(prep_dir, exist_ok=True)

    rows = []
    for pid in sorted(train_pids):
        rows.append({"fold": fold, "PatientID": pid,
                     "Histology": cohort.bags[pid].histology,
                     "label": cohort.bags[pid].label,
                     "assignment": "outer_train", "inner_fold": assign[pid]})
    for pid in sorted(test_pids):
        rows.append({"fold": fold, "PatientID": pid,
                     "Histology": cohort.bags[pid].histology,
                     "label": cohort.bags[pid].label,
                     "assignment": "outer_test", "inner_fold": 0})
    pd.DataFrame(rows).to_csv(os.path.join(inner_dir, "fold_%d.csv" % fold), index=False)

    for j, sp in inner_splits.items():
        np.savez_compressed(
            os.path.join(prep_dir, "fold_%d_inner_%d_scaler.npz" % (fold, j)),
            keep_idx=sp.scaler.keep_idx, mean=sp.scaler.mean, std=sp.scaler.std,
            kept_features=np.array(sp.scaler.kept_features, dtype=object),
            dropped_features=np.array(sp.scaler.dropped_features, dtype=object),
            pos_weight=np.float64(sp.pos_weight), fold=np.int32(fold), inner_fold=np.int32(j))
    np.savez_compressed(
        os.path.join(prep_dir, "fold_%d_final_scaler.npz" % fold),
        keep_idx=final_split.scaler.keep_idx, mean=final_split.scaler.mean,
        std=final_split.scaler.std,
        kept_features=np.array(final_split.scaler.kept_features, dtype=object),
        dropped_features=np.array(final_split.scaler.dropped_features, dtype=object),
        pos_weight=np.float64(final_split.pos_weight), fold=np.int32(fold))

    meta = {
        "fold": fold,
        "inner_cv_seed": int(cfg["inner_cv"]["base_seed"]) + fold,
        "n_inner_splits": int(cfg["inner_cv"]["n_splits"]),
        "n_outer_train_patients": len(train_pids),
        "n_outer_test_patients": len(test_pids),
        "final_pos_weight": final_split.pos_weight,
        "final_pos_weight_definition": "n_scc_outer_train_patients / n_adc_outer_train_patients",
        "final_scaling_source": "all_outer_train_patches_only",
        "final_scaler_fingerprint": final_split.scaler.fingerprint(),
        "n_features_kept": len(final_split.scaler.kept_features),
        "dropped_features": final_split.scaler.dropped_features,
        "inner": {str(j): {
            "n_inner_train_patients": len(sp.train),
            "n_inner_val_patients": len(sp.evaluate),
            "n_inner_train_adc": int((cohort.labels(sp.train) == 1).sum()),
            "n_inner_train_scc": int((cohort.labels(sp.train) == 0).sum()),
            "n_inner_val_adc": int((cohort.labels(sp.evaluate) == 1).sum()),
            "n_inner_val_scc": int((cohort.labels(sp.evaluate) == 0).sum()),
            "n_inner_train_patches": int(sum(cohort.bags[p].n_patches for p in sp.train)),
            "n_inner_val_patches": int(sum(cohort.bags[p].n_patches for p in sp.evaluate)),
            "pos_weight": sp.pos_weight,
            "scaler_fingerprint": sp.scaler.fingerprint(),
            "scaling_source": "inner_train_patches_only",
        } for j, sp in inner_splits.items()},
        "variance_rule": cfg["preprocessing"]["variance"],
    }
    with open(os.path.join(prep_dir, "fold_%d_scalers.json" % fold), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def save_epoch_curve(cfg: dict, fold: int, model_name: str, mean_curve: np.ndarray,
                     sd_curve: np.ndarray, all_aucs: np.ndarray,
                     curves: List[List[dict]]) -> None:
    out_dir = os.path.join(ppath(cfg, cfg["outputs"]["results_dir"]), "epoch_selection")
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame({"epoch": np.arange(1, len(mean_curve) + 1)})
    for j in range(all_aucs.shape[0]):
        df["inner_%d_val_roc_auc" % (j + 1)] = all_aucs[j]
        df["inner_%d_train_loss" % (j + 1)] = [r["train_loss"] for r in curves[j]]
    df["mean_val_roc_auc"] = mean_curve
    df["sd_val_roc_auc"] = sd_curve
    df.to_csv(os.path.join(out_dir, "%s_fold_%d.csv" % (model_name, fold)), index=False)


def save_attention(cfg: dict, cohort: Cohort, fold: int, test_pids: Sequence[str],
                   attns: Dict[str, np.ndarray], test_prob: np.ndarray) -> None:
    """Raw attention weights only.  No maps, no spatial interpretation."""
    out_dir = ppath(cfg, cfg["outputs"]["attention_dir"])
    os.makedirs(out_dir, exist_ok=True)
    thr = float(cfg["evaluation"]["threshold"])
    prob_of = {p: float(v) for p, v in zip(test_pids, test_prob)}
    for pid in test_pids:
        alpha = attns[pid]
        bag = cohort.bags[pid]
        if alpha.shape[0] != bag.n_patches:
            raise RuntimeError("attention length %d != %d patches for %s"
                               % (alpha.shape[0], bag.n_patches, pid))
        if not np.all(np.isfinite(alpha)):
            raise RuntimeError("non-finite attention for %s" % pid)
        payload = dict(
            patient_id=np.str_(pid), histology=np.str_(bag.histology),
            true_label=np.int8(bag.label), attention=alpha.astype(np.float64),
            patch_index=np.arange(bag.n_patches, dtype=np.int32),
            coords_index=bag.coords_index, coords_world=bag.coords_world,
            outer_fold=np.int32(fold),
            predicted_probability_adc=np.float64(prob_of[pid]),
            predicted_label=np.int8(1 if prob_of[pid] >= thr else 0),
            model=np.str_("attention_mil"), phase=np.str_(phase_tag(cfg)),
            roi_name=np.str_(cfg["phase3"]["roi"]),
            extraction_signature=np.str_(bag.config_signature),
            attention_sum=np.float64(float(alpha.sum())), npz_version=np.int32(1))
        if bag.roi_voxels is not None:
            payload["roi_voxels"] = bag.roi_voxels
        else:
            # CNN-embedding instances carry no radiomic ROI-voxel count; the
            # required `attention_weights` alias is written for every phase.
            payload["attention_weights"] = alpha.astype(np.float64)
            for k, v in (bag.provenance or {}).items():
                payload[k] = np.str_(v)
        np.savez_compressed(os.path.join(out_dir, "%s.npz" % pid), **payload)


def verify_frozen_inputs(cfg: dict, cohort: Cohort) -> dict:
    """Pre-flight verification of the frozen inputs, BEFORE any training.

    Runs only when `phase3.verify_frozen_inputs` is true, so Phases 3/3B/5/5C
    keep their exact code path.  Every claim is re-derived from files on disk:
    the outer split's SHA-256 against its own Phase-4 metadata, the instance
    count, the feature width, and - for a CNN-embedding cohort - the frozen
    encoder identity, the input transform and the crop signature recorded in
    every embedding file.  A disagreement raises; nothing is silently repaired.
    """
    p3 = cfg["phase3"]
    if not bool(p3.get("verify_frozen_inputs", False)):
        return {}
    out: dict = {}

    split_path = ppath(cfg, p3["splits_file"])
    split_sha = sha256_file(split_path)
    with open(ppath(cfg, p3["splits_meta_file"]), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    declared = str(meta.get("splits_file_sha256", ""))
    if split_sha != declared:
        raise RuntimeError("frozen outer split changed: %s != %s (metadata)"
                           % (split_sha, declared))
    if not bool(meta.get("frozen", False)):
        raise RuntimeError("outer split metadata does not declare the split frozen")
    out["splits_file_sha256"] = split_sha
    out["splits_file_sha256_matches_metadata"] = True
    out["split_n_patients"] = int(meta.get("n_patients", -1))

    n_inst = int(sum(b.n_patches for b in cohort.bags.values()))
    if "expected_instances" in p3 and n_inst != int(p3["expected_instances"]):
        raise RuntimeError("instance count %d != expected %d"
                           % (n_inst, int(p3["expected_instances"])))
    if "expected_features" in p3 and len(cohort.feature_names) != int(p3["expected_features"]):
        raise RuntimeError("feature width %d != expected %d"
                           % (len(cohort.feature_names), int(p3["expected_features"])))
    out["n_instances"] = n_inst
    out["n_features"] = len(cohort.feature_names)

    sig = str(p3.get("expected_extraction_signature", ""))
    if sig:
        bad = sorted(b.patient_id for b in cohort.bags.values()
                     if b.config_signature != sig)
        if bad:
            raise RuntimeError("extraction signature mismatch in %d bags (%s...)"
                               % (len(bad), bad[0]))
        out["extraction_signature"] = sig

    enc = p3.get("expected_encoder") or {}
    if enc:
        want = {"encoder_architecture": str(enc["architecture"]),
                "encoder_weights_enum": str(enc["weights_enum"]),
                "encoder_state_fingerprint": str(enc["state_fingerprint"]),
                "encoder_head": str(enc["head"])}
        for b in cohort.bags.values():
            for k, v in want.items():
                if b.provenance.get(k) != v:
                    raise RuntimeError("%s: %s = %r, expected %r"
                                       % (b.patient_id, k, b.provenance.get(k), v))
        out["encoder"] = want
        out["encoder_frozen_in_every_file"] = True
    for key, cfg_key in (("transform_signature", "expected_transform_signature"),
                         ("image_crop_signature", "expected_image_crop_signature")):
        want_v = str(p3.get(cfg_key, ""))
        if want_v:
            bad = sorted(b.patient_id for b in cohort.bags.values()
                         if b.provenance.get(key) != want_v)
            if bad:
                raise RuntimeError("%s mismatch in %d files (%s...)"
                                   % (key, len(bad), bad[0]))
            out[key] = want_v

    # the outer split must describe exactly the loaded cohort
    split_df = load_outer_split(cfg)
    split_pids = sorted(set(split_df["PatientID"].tolist()))
    if split_pids != sorted(cohort.bags):
        raise RuntimeError("outer split and loaded cohort describe different patients")
    lab = split_df.drop_duplicates("PatientID").set_index("PatientID")["label"]
    bad = sorted(p for p in split_pids if int(lab[p]) != int(cohort.bags[p].label))
    if bad:
        raise RuntimeError("label disagreement between split and bags: %s" % bad[:5])
    out["split_matches_cohort"] = True
    out["labels_match_split"] = True
    return out


def input_manifest(cfg: dict) -> dict:
    feat_dir = ppath(cfg, cfg["phase3"]["features_dir"])
    bags = {f: sha256_file(os.path.join(feat_dir, f))
            for f in sorted(os.listdir(feat_dir)) if f.endswith(".npz")}
    return {"n_bags": len(bags), "bag_sha256": bags,
            "splits_file_sha256": sha256_file(ppath(cfg, cfg["phase3"]["splits_file"])),
            "feature_names_sha256": sha256_file(
                ppath(cfg, cfg["phase3"]["feature_names_file"]))}


# ---------------------------------------------------------------------- driver
def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3B inner-CV epoch selection")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--folds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    ap.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    args = ap.parse_args()

    cfg = load_phase3_config(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), *args.config.split("/")))
    log_path = ppath(cfg, str(cfg["outputs"].get("training_log",
                                                 "logs/phase3b_training.log")))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_fh = open(log_path, "a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    t_start = time.time()
    log("=" * 78)
    log("PHASE %s - inner-CV epoch selection (ROI %s)  %s"
        % (phase_tag(cfg).upper(), cfg["phase3"]["roi"],
           time.strftime("%Y-%m-%d %H:%M:%S")))
    log("models=%s folds=%s config=%s" % (args.models, args.folds, args.config))

    manifest_before = input_manifest(cfg)
    cohort = load_cohort(cfg)
    split_df = load_outer_split(cfg)
    p3 = cfg["phase3"]
    labels = cohort.labels(cohort.patient_ids)
    assert len(cohort.bags) == int(p3["expected_patients"])
    assert int((labels == 1).sum()) == int(p3["expected_adc"])
    assert int((labels == 0).sum()) == int(p3["expected_scc"])
    log("cohort: %d bags, %d ADC / %d SCC, %d features, %d patches"
        % (len(cohort.bags), int((labels == 1).sum()), int((labels == 0).sum()),
           len(cohort.feature_names), sum(b.n_patches for b in cohort.bags.values())))

    preflight = verify_frozen_inputs(cfg, cohort)
    for k in sorted(preflight):
        log("  preflight %s = %s" % (k, preflight[k]))

    fold_results = [process_fold(cfg, cohort, split_df, f, args.models, log)
                    for f in args.folds]

    manifest_after = input_manifest(cfg)
    total_seconds = time.time() - t_start

    rows = []
    for fr in fold_results:
        for model_name, mr in fr["models"].items():
            for pid, prob in mr["test_probabilities"].items():
                bag = cohort.bags[pid]
                rows.append({"PatientID": pid, "histology": bag.histology,
                             "true_label": int(bag.label),
                             "predicted_probability_ADC": float(prob),
                             "predicted_label": int(
                                 prob >= float(cfg["evaluation"]["threshold"])),
                             "outer_fold": int(fr["fold"]), "model": model_name,
                             "selected_epoch": int(mr["selected_epoch"]),
                             "n_patches": bag.n_patches})
    df = pd.DataFrame(rows)
    pred_dir = ppath(cfg, cfg["outputs"]["predictions_dir"])
    os.makedirs(pred_dir, exist_ok=True)
    for model_name in args.models:
        sub = df[df["model"] == model_name].sort_values("PatientID")
        tag = model_tag(cfg, model_name)
        sub.to_csv(os.path.join(pred_dir, "%s_oof.csv" % tag), index=False)
        log("wrote %s/%s_oof.csv  (%d rows)"
            % (cfg["outputs"]["predictions_dir"], tag, len(sub)))

    meta = {
        "phase": phase_tag(cfg),
        "roi": str(cfg["phase3"]["roi"]),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "procedure": "stratified 3-fold inner CV epoch selection + from-scratch refit "
                     "on all outer-training patients",
        "models": list(args.models), "folds": list(args.folds),
        "total_runtime_seconds": total_seconds,
        "selection_seconds_by_model": {
            m: sum(fr["models"][m]["selection_seconds"] for fr in fold_results
                   if m in fr["models"]) for m in args.models},
        "refit_seconds_by_model": {
            m: sum(fr["models"][m]["refit_seconds"] for fr in fold_results
                   if m in fr["models"]) for m in args.models},
        "selected_epochs": {m: {fr["fold"]: fr["models"][m]["selected_epoch"]
                                for fr in fold_results if m in fr["models"]}
                            for m in args.models},
        "frozen_inputs_unchanged": bool(manifest_before == manifest_after),
        "preflight_verification": preflight,
        "input_manifest": manifest_before,
        "folds_detail": [{k: v for k, v in fr.items() if k != "models"} |
                         {"models": {m: {k: v for k, v in mr.items()
                                         if k != "test_probabilities"}
                                     for m, mr in fr["models"].items()}}
                         for fr in fold_results],
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "torch": torch.__version__, "numpy": np.__version__,
                        "device": str(cfg["training"]["device"]),
                        "torch_threads": torch.get_num_threads()},
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "config_path": cfg["_config_path"],
        "radiomics_imported_during_training": ("radiomics" in sys.modules
                                               or "SimpleITK" in sys.modules),
    }
    with open(ppath(cfg, str(cfg["outputs"].get("run_json",
                                                "reports/phase3b_run.json"))),
              "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)

    log("-" * 78)
    log("selected epochs: %s" % json.dumps(meta["selected_epochs"]))
    log("frozen inputs unchanged: %s" % meta["frozen_inputs_unchanged"])
    log("radiomics/SimpleITK imported during training: %s"
        % meta["radiomics_imported_during_training"])
    log("total runtime %.1f s" % total_seconds)
    log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
