"""Phase 6C - CLAM-inspired instance supervision inside the frozen Phase-6B nested CV.

Exactly ONE model is trained: the Phase-6B gated-attention MIL on the frozen
Phase-6A 512-D ResNet18 embeddings, plus an auxiliary instance-level classifier
and instance loss.  The image encoder, the representation, the ROI, the patch
centres, the cohort, the frozen outer split, the bag classifier, the gated
attention module, the optimiser, the bag loss, the preprocessing rule and the
evaluation are all inherited unchanged.  **The instance-level auxiliary loss is
the only changed variable.**

Per outer fold:

  1. outer-test patients are untouched;
  2. stratified 3-fold inner CV over the outer-TRAIN patients (seed 3000 + fold),
     the identical assignment Phase 6B used;
  3. for each candidate lambda_instance in {0.1, 0.3}, each inner run trains from
     scratch for the full 100-epoch budget and records patient-level ROC-AUC on
     its inner-validation split at every epoch;
  4. the per-epoch ROC-AUC is averaged over the 3 inner folds, giving one
     (lambda_instance x epoch) surface;
  5. (lambda_instance, epoch) is chosen JOINTLY as the argmax of that surface -
     ties broken by smaller lambda_instance first, then lower epoch;
  6. no outer-test information enters any of the above;
  7. the final scaler is fitted on ALL outer-training patches;
  8. pos_weight is recomputed from ALL outer-training PATIENT counts;
  9. the model is retrained from scratch on all outer-training patients with the
     selected lambda_instance for exactly the selected number of epochs;
 10. it is evaluated once on the outer-test fold, with NO instance loss.

Usage:
    python src/train_mil_clam.py --config config/phase6c.yaml
    python src/train_mil_clam.py --config config/phase6c.yaml --folds 1
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clam_mil import (MODEL_NAME, bag_k, build_clam_model, k_rule_from_config,  # noqa: E402
                      predict_clam, train_one_epoch_clam)
from mil_data import (Cohort, load_cohort, load_outer_split,  # noqa: E402
                      load_phase3_config, model_tag, outer_fold_patients, ppath,
                      sha256_file)
from mil_metrics import patient_metrics  # noqa: E402
from train_mil_cv import (Split, inner_cv_assignment, input_manifest,  # noqa: E402
                          make_split, save_fold_splits, set_seeds,
                          verify_frozen_inputs)

DEFAULT_CONFIG = "config/phase6c.yaml"


def lambda_grid(cfg: dict) -> List[float]:
    """Ascending, so the flattened argmax already breaks ties toward smaller lambda."""
    grid = sorted(float(v) for v in cfg["instance_supervision"]["lambda_instance_grid"])
    if not grid:
        raise ValueError("empty lambda_instance grid")
    return grid


def lam_tag(lam: float) -> str:
    return ("%.4f" % float(lam)).rstrip("0").rstrip(".").replace(".", "p")


# ------------------------------------------------------------------ one training run
def run_training(cfg: dict, cohort: Cohort, split: Split, seed: int, n_epochs: int,
                 lambda_instance: float, eval_every_epoch: bool,
                 collect_last_epoch: bool = False):
    """Train from scratch for exactly `n_epochs`; optionally score the eval set each epoch."""
    tcfg = cfg["training"]
    device = torch.device(str(tcfg.get("device", "cpu")))
    set_seeds(cfg, seed)

    model = build_clam_model(cfg, input_dim=len(split.scaler.kept_features)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tcfg["learning_rate"]),
                                 weight_decay=float(tcfg["weight_decay"]))
    # BAG loss: weighted BCE, pos_weight from TRAINING PATIENT counts (Phase-6B rule)
    bag_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([split.pos_weight], dtype=torch.float32, device=device))
    # INSTANCE loss: plain BCE.  top-k and bottom-k are equal in size inside every
    # bag, so the pseudo targets are already balanced and the ADC/SCC pos_weight
    # must NOT be reused here.
    inst_loss_fn = nn.BCEWithLogitsLoss()

    labels = {p: int(cohort.bags[p].label) for p in split.train}
    krule = k_rule_from_config(cfg)
    rng = random.Random(seed)
    y_eval = cohort.labels(split.evaluate) if eval_every_epoch else None
    curve: List[dict] = []
    per_bag: List[dict] = []

    for epoch in range(1, int(n_epochs) + 1):
        order = list(split.train)
        rng.shuffle(order)
        collect = per_bag if (collect_last_epoch and epoch == int(n_epochs)) else None
        stats = train_one_epoch_clam(model, split.tensors, labels, order, optimizer,
                                     bag_loss_fn, inst_loss_fn, device,
                                     lambda_instance, krule, collect=collect)
        row = {"epoch": epoch, "lambda_instance": float(lambda_instance)}
        row.update({k: stats[k] for k in ("bag_loss", "instance_loss", "total_loss",
                                          "n_bags_with_instance_loss", "mean_k",
                                          "median_k", "min_k", "max_k",
                                          "mean_instance_pseudo_accuracy",
                                          "mean_attention_separation",
                                          "mean_fraction_participating")})
        # `train_loss` keeps the Phase-6B column name for the BAG objective
        row["train_loss"] = stats["bag_loss"]
        if eval_every_epoch:
            prob, _ = predict_clam(model, split.tensors, split.evaluate, device)
            if not np.all(np.isfinite(prob)):
                raise RuntimeError("non-finite validation probabilities at epoch %d" % epoch)
            if len(np.unique(y_eval)) < 2:
                raise RuntimeError("split-generation failure: single-class inner validation set")
            m = patient_metrics(y_eval, prob, float(cfg["evaluation"]["threshold"]))
            row["val_roc_auc"] = m["roc_auc"]
            row["val_pr_auc"] = m["pr_auc"]
        curve.append(row)
    return model, curve, per_bag


# ------------------------------------------------------------------- joint selection
def select_lambda_and_epoch(surface: np.ndarray, grid: Sequence[float]):
    """Joint argmax of the (lambda x epoch) mean inner-CV ROC-AUC surface.

    `surface` is [n_lambda, n_epochs] with the lambda axis in ASCENDING order, so
    a flattened `np.argmax` - which returns the FIRST maximum in row-major order -
    already implements the required tie-break: smaller lambda first, then lower
    epoch.
    """
    if not np.all(np.isfinite(surface)):
        raise RuntimeError("non-finite (lambda x epoch) selection surface")
    flat = int(np.argmax(surface))
    li, ei = divmod(flat, surface.shape[1])
    return float(grid[li]), int(ei + 1), float(surface[li, ei]), li, ei


# ---------------------------------------------------------------------- one fold
def process_fold(cfg: dict, cohort: Cohort, split_df: pd.DataFrame, fold: int, log) -> dict:
    train_pids, test_pids = outer_fold_patients(split_df, fold)
    assign = inner_cv_assignment(cfg, cohort, fold, train_pids)
    n_inner = int(cfg["inner_cv"]["n_splits"])
    grid = lambda_grid(cfg)
    max_epochs = int(cfg["training"]["max_epochs"])
    y_tr = cohort.labels(train_pids)

    log("-" * 78)
    log("fold %d: outer_train %d (%d ADC/%d SCC), outer_test %d | inner CV %d-fold, seed %d"
        % (fold, len(train_pids), int((y_tr == 1).sum()), int((y_tr == 0).sum()),
           len(test_pids), n_inner, int(cfg["inner_cv"]["base_seed"]) + fold))

    inner_splits: Dict[int, Split] = {}
    for j in range(1, n_inner + 1):
        itr = sorted(p for p in train_pids if assign[p] != j)
        iva = sorted(p for p in train_pids if assign[p] == j)
        inner_splits[j] = make_split(cfg, cohort, itr, iva)
        yv = cohort.labels(iva)
        log("  inner fold %d: train %d, val %d (%d ADC/%d SCC), pos_weight %.4f"
            % (j, len(itr), len(iva), int((yv == 1).sum()), int((yv == 0).sum()),
               inner_splits[j].pos_weight))

    final_split = make_split(cfg, cohort, sorted(train_pids), sorted(test_pids))
    log("  final refit: scaler + pos_weight from all %d outer-train patients (pw %.4f)"
        % (len(train_pids), final_split.pos_weight))
    save_fold_splits(cfg, cohort, fold, assign, train_pids, test_pids,
                     inner_splits, final_split)

    out = {"fold": fold, "n_outer_train": len(train_pids), "n_outer_test": len(test_pids),
           "inner_cv_seed": int(cfg["inner_cv"]["base_seed"]) + fold,
           "final_pos_weight": final_split.pos_weight,
           "inner_pos_weight": {j: inner_splits[j].pos_weight for j in inner_splits},
           "n_features_kept": len(final_split.scaler.kept_features),
           "n_features_dropped": len(final_split.scaler.dropped_features),
           "lambda_grid": grid}

    # ---------------- inner CV: every (lambda, epoch) cell, on inner data only
    t0 = time.time()
    surface = np.full((len(grid), max_epochs), np.nan, dtype=float)
    sd_surface = np.full((len(grid), max_epochs), np.nan, dtype=float)
    per_lambda_curves: Dict[float, List[List[dict]]] = {}
    inner_argmax: Dict[float, List[int]] = {}

    for li, lam in enumerate(grid):
        curves = []
        for j in range(1, n_inner + 1):
            seed = int(cfg["seeds"]["inner_base_seed"]) + 100 * fold + 10 * j
            _, curve, _ = run_training(cfg, cohort, inner_splits[j], seed, max_epochs,
                                       lam, eval_every_epoch=True)
            curves.append(curve)
            log("    lambda %.2f inner %d: best epoch-wise val ROC-AUC %.4f (seed %d)"
                % (lam, j, max(r["val_roc_auc"] for r in curve), seed))
        aucs = np.array([[r["val_roc_auc"] for r in c] for c in curves], dtype=float)
        surface[li] = aucs.mean(axis=0)
        sd_surface[li] = aucs.std(axis=0, ddof=1)
        per_lambda_curves[lam] = curves
        inner_argmax[lam] = [int(np.argmax([r["val_roc_auc"] for r in c])) + 1 for c in curves]

    sel_lambda, sel_epoch, sel_auc, li, ei = select_lambda_and_epoch(surface, grid)
    selection_seconds = time.time() - t0
    save_selection(cfg, fold, grid, surface, sd_surface, per_lambda_curves,
                   sel_lambda, sel_epoch)
    log("    -> selected lambda_instance %.2f, epoch %d (mean inner-CV ROC-AUC %.4f, "
        "sd %.4f) in %.1f s"
        % (sel_lambda, sel_epoch, sel_auc, float(sd_surface[li, ei]), selection_seconds))
    for lj, lam in enumerate(grid):
        b = int(np.argmax(surface[lj]))
        log("       lambda %.2f best %.4f @ epoch %d | inner argmaxes %s"
            % (lam, float(surface[lj, b]), b + 1, inner_argmax[lam]))

    # ---------------- final refit on ALL outer-training patients, one test evaluation
    t1 = time.time()
    final_seed = int(cfg["seeds"]["final_base_seed"]) + 10 * fold
    model, refit_curve, per_bag = run_training(
        cfg, cohort, final_split, final_seed, sel_epoch, sel_lambda,
        eval_every_epoch=False, collect_last_epoch=True)
    device = torch.device(str(cfg["training"]["device"]))
    test_prob, test_att = predict_clam(model, final_split.tensors, final_split.evaluate,
                                       device, want_attention=True)
    refit_seconds = time.time() - t1
    if not np.all(np.isfinite(test_prob)):
        raise RuntimeError("non-finite test probabilities, fold %d" % fold)
    y_test = cohort.labels(final_split.evaluate)
    fm = patient_metrics(y_test, test_prob, float(cfg["evaluation"]["threshold"]))
    log("    fold %d: test ROC-AUC %.4f  PR-AUC %.4f  balacc %.4f  MCC %.4f (refit %.1f s)"
        % (fold, fm["roc_auc"], fm["pr_auc"], fm["balanced_accuracy"], fm["mcc"],
           refit_seconds))

    save_refit_diagnostics(cfg, fold, sel_lambda, sel_epoch, refit_curve, per_bag)

    ck_dir = ppath(cfg, "%s/%s" % (cfg["outputs"]["models_dir"],
                                   model_tag(cfg, MODEL_NAME)))
    os.makedirs(ck_dir, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name": MODEL_NAME, "fold": fold, "phase": "6c",
        "roi": str(cfg["phase3"]["roi"]),
        "input_dim": len(final_split.scaler.kept_features),
        "kept_features": final_split.scaler.kept_features,
        "scaler_mean": final_split.scaler.mean, "scaler_std": final_split.scaler.std,
        "selected_epoch": sel_epoch,
        "selected_lambda_instance": sel_lambda,
        "selection_mean_inner_cv_roc_auc": sel_auc,
        "lambda_grid": grid,
        "pos_weight": final_split.pos_weight, "seed": final_seed,
        "inner_cv_seed": int(cfg["inner_cv"]["base_seed"]) + fold,
        "config": {"model": cfg["model"], "training": cfg["training"],
                   "inner_cv": cfg["inner_cv"], "seeds": cfg["seeds"],
                   "instance_supervision": cfg["instance_supervision"]},
    }, os.path.join(ck_dir, "fold_%d.pt" % fold))

    save_attention(cfg, cohort, fold, final_split.evaluate, test_att, test_prob,
                   sel_lambda, sel_epoch)

    last = refit_curve[-1]
    bag_ks = {p: bag_k(cohort.bags[p].n_patches, k_rule_from_config(cfg))
              for p in final_split.train}
    out.update({
        "selected_lambda_instance": sel_lambda,
        "selected_epoch": sel_epoch,
        "selection_mean_inner_cv_roc_auc": sel_auc,
        "selection_sd_at_selected_cell": float(sd_surface[li, ei]),
        "per_lambda_best": {("%.2f" % lam): {
            "best_mean_inner_cv_roc_auc": float(surface[lj].max()),
            "best_epoch": int(np.argmax(surface[lj])) + 1,
            "inner_argmax_epochs": inner_argmax[lam]} for lj, lam in enumerate(grid)},
        "final_seed": final_seed,
        "selection_seconds": selection_seconds,
        "refit_seconds": refit_seconds,
        "fold_metrics": fm,
        "refit_last_epoch": {k: last[k] for k in
                             ("bag_loss", "instance_loss", "total_loss", "mean_k",
                              "median_k", "min_k", "max_k",
                              "mean_instance_pseudo_accuracy",
                              "mean_attention_separation",
                              "mean_fraction_participating",
                              "n_bags_with_instance_loss")},
        "k_stats_outer_train": {
            "n_bags": len(bag_ks),
            "mean_k": float(np.mean(list(bag_ks.values()))),
            "median_k": float(np.median(list(bag_ks.values()))),
            "min_k": int(min(bag_ks.values())), "max_k": int(max(bag_ks.values())),
            "all_2k_le_n": bool(all(2 * bag_ks[p] <= cohort.bags[p].n_patches
                                    for p in bag_ks))},
        "test_probabilities": {p: float(v)
                               for p, v in zip(final_split.evaluate, test_prob)},
    })
    return out


# ------------------------------------------------------------------- persistence
def save_selection(cfg: dict, fold: int, grid: Sequence[float], surface: np.ndarray,
                   sd_surface: np.ndarray, per_lambda: Dict[float, List[List[dict]]],
                   sel_lambda: float, sel_epoch: int) -> None:
    out_dir = ppath(cfg, cfg["outputs"]["selection_dir"])
    os.makedirs(out_dir, exist_ok=True)
    n_epochs = surface.shape[1]
    epochs = np.arange(1, n_epochs + 1)

    # one file per lambda: every inner curve plus the mean and SD
    for li, lam in enumerate(grid):
        df = pd.DataFrame({"epoch": epochs, "lambda_instance": float(lam)})
        for j, c in enumerate(per_lambda[lam], start=1):
            df["inner_%d_val_roc_auc" % j] = [r["val_roc_auc"] for r in c]
            df["inner_%d_bag_loss" % j] = [r["bag_loss"] for r in c]
            df["inner_%d_instance_loss" % j] = [r["instance_loss"] for r in c]
            df["inner_%d_total_loss" % j] = [r["total_loss"] for r in c]
            df["inner_%d_instance_pseudo_accuracy" % j] = [
                r["mean_instance_pseudo_accuracy"] for r in c]
        df["mean_val_roc_auc"] = surface[li]
        df["sd_val_roc_auc"] = sd_surface[li]
        df.to_csv(os.path.join(out_dir, "fold_%d_lambda_%s.csv" % (fold, lam_tag(lam))),
                  index=False)

    # the joint surface, long form
    rows = []
    for li, lam in enumerate(grid):
        for e in range(n_epochs):
            rows.append({"fold": fold, "lambda_instance": float(lam), "epoch": e + 1,
                         "mean_inner_cv_roc_auc": float(surface[li, e]),
                         "sd_inner_cv_roc_auc": float(sd_surface[li, e]),
                         "selected": bool(abs(lam - sel_lambda) < 1e-12
                                          and (e + 1) == sel_epoch)})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "fold_%d_surface.csv" % fold),
                              index=False)


def save_refit_diagnostics(cfg: dict, fold: int, lam: float, sel_epoch: int,
                           refit_curve: List[dict], per_bag: List[dict]) -> None:
    out_dir = ppath(cfg, cfg["outputs"]["selection_dir"])
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(refit_curve)
    df.insert(0, "fold", fold)
    df.to_csv(os.path.join(out_dir, "fold_%d_refit_epochs.csv" % fold), index=False)
    if per_bag:
        pb = pd.DataFrame(per_bag)
        pb.insert(0, "fold", fold)
        pb.insert(1, "lambda_instance", float(lam))
        pb.insert(2, "epoch", int(sel_epoch))
        cols = ["fold", "lambda_instance", "epoch", "PatientID", "n_instances", "k",
                "n_pseudo_instances", "fraction_participating", "overlap",
                "mean_top_attention", "mean_bottom_attention", "attention_separation",
                "attention_separation_ratio", "instance_pseudo_accuracy",
                "instance_loss", "bag_loss"]
        pb[cols].to_csv(os.path.join(out_dir, "fold_%d_refit_per_bag.csv" % fold),
                        index=False)


def save_attention(cfg: dict, cohort: Cohort, fold: int, test_pids: Sequence[str],
                   attns: Dict[str, np.ndarray], test_prob: np.ndarray,
                   lam: float, sel_epoch: int) -> None:
    """Raw outer-test attention weights only.  No maps, no spatial interpretation."""
    out_dir = ppath(cfg, cfg["outputs"]["attention_dir"])
    os.makedirs(out_dir, exist_ok=True)
    thr = float(cfg["evaluation"]["threshold"])
    krule = k_rule_from_config(cfg)
    prob_of = {p: float(v) for p, v in zip(test_pids, test_prob)}
    for pid in test_pids:
        alpha = attns[pid]
        bag = cohort.bags[pid]
        if alpha.shape[0] != bag.n_patches:
            raise RuntimeError("attention length %d != %d instances for %s"
                               % (alpha.shape[0], bag.n_patches, pid))
        if not np.all(np.isfinite(alpha)):
            raise RuntimeError("non-finite attention for %s" % pid)
        payload = dict(
            patient_id=np.str_(pid), histology=np.str_(bag.histology),
            true_label=np.int8(bag.label), attention=alpha.astype(np.float64),
            attention_weights=alpha.astype(np.float64),
            patch_index=np.arange(bag.n_patches, dtype=np.int32),
            coords_index=bag.coords_index, coords_world=bag.coords_world,
            outer_fold=np.int32(fold),
            predicted_probability_adc=np.float64(prob_of[pid]),
            predicted_label=np.int8(1 if prob_of[pid] >= thr else 0),
            model=np.str_(MODEL_NAME), phase=np.str_("6c"),
            roi_name=np.str_(cfg["phase3"]["roi"]),
            extraction_signature=np.str_(bag.config_signature),
            attention_sum=np.float64(float(alpha.sum())),
            selected_lambda_instance=np.float64(float(lam)),
            selected_epoch=np.int32(int(sel_epoch)),
            adaptive_k_if_it_were_a_training_bag=np.int32(bag_k(bag.n_patches, krule)),
            instance_loss_applied_to_this_bag=np.bool_(False),
            npz_version=np.int32(1))
        for k, v in (bag.provenance or {}).items():
            payload[k] = np.str_(v)
        np.savez_compressed(os.path.join(out_dir, "%s.npz" % pid), **payload)


def write_instance_diagnostics(cfg: dict, cohort: Cohort, fold_results: List[dict]) -> None:
    """Per-fold optimisation diagnostics.  DESCRIPTIVE ONLY.

    `instance_pseudo_accuracy` measures how well the auxiliary head separates the
    model's OWN high-attention group from its OWN low-attention group.  It is an
    optimisation number and carries NO biological meaning whatsoever.
    """
    rows = []
    for fr in fold_results:
        last = fr["refit_last_epoch"]
        ks = fr["k_stats_outer_train"]
        rows.append({
            "fold": fr["fold"],
            "selected_lambda_instance": fr["selected_lambda_instance"],
            "selected_epoch": fr["selected_epoch"],
            "selection_mean_inner_cv_roc_auc": fr["selection_mean_inner_cv_roc_auc"],
            "n_outer_train_bags": ks["n_bags"],
            "n_bags_with_instance_loss": last["n_bags_with_instance_loss"],
            "bag_loss_final_epoch": last["bag_loss"],
            "instance_loss_final_epoch": last["instance_loss"],
            "total_loss_final_epoch": last["total_loss"],
            "mean_k": ks["mean_k"], "median_k": ks["median_k"],
            "min_k": ks["min_k"], "max_k": ks["max_k"],
            "k_range": "%d-%d" % (ks["min_k"], ks["max_k"]),
            "all_2k_le_n": ks["all_2k_le_n"],
            "mean_fraction_instances_in_pseudo_supervision":
                last["mean_fraction_participating"],
            "mean_attention_separation_top_minus_bottom":
                last["mean_attention_separation"],
            "instance_classifier_pseudo_label_accuracy":
                last["mean_instance_pseudo_accuracy"],
            "outer_test_roc_auc": fr["fold_metrics"]["roc_auc"],
        })
    out = ppath(cfg, os.path.join(cfg["outputs"]["results_dir"], "instance_diagnostics.csv"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    # the adaptive-k distribution over every patient, independent of fold role
    krule = k_rule_from_config(cfg)
    krows = []
    for pid in cohort.patient_ids:
        b = cohort.bags[pid]
        k = bag_k(b.n_patches, krule)
        krows.append({"PatientID": pid, "histology": b.histology, "label": int(b.label),
                      "n_instances": b.n_patches, "adaptive_k": k,
                      "n_pseudo_instances": 2 * k,
                      "fraction_participating": (2.0 * k) / float(b.n_patches),
                      "two_k_le_n": bool(2 * k <= b.n_patches)})
    kout = ppath(cfg, os.path.join(cfg["outputs"]["results_dir"], "adaptive_k.csv"))
    pd.DataFrame(krows).to_csv(kout, index=False)


# ------------------------------------------------------------------------ driver
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 6C - CLAM-inspired instance supervision, nested CV")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--folds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_phase3_config(os.path.join(root, *args.config.split("/")))
    log_path = ppath(cfg, str(cfg["outputs"]["training_log"]))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_fh = open(log_path, "a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    t_start = time.time()
    log("=" * 78)
    log("PHASE 6C - CLAM-inspired instance supervision (ROI %s)  %s"
        % (cfg["phase3"]["roi"], time.strftime("%Y-%m-%d %H:%M:%S")))
    log("folds=%s config=%s lambda_grid=%s" % (args.folds, args.config, lambda_grid(cfg)))

    manifest_before = input_manifest(cfg)
    cohort = load_cohort(cfg)
    split_df = load_outer_split(cfg)
    p3 = cfg["phase3"]
    labels = cohort.labels(cohort.patient_ids)
    assert len(cohort.bags) == int(p3["expected_patients"])
    assert int((labels == 1).sum()) == int(p3["expected_adc"])
    assert int((labels == 0).sum()) == int(p3["expected_scc"])
    log("cohort: %d bags, %d ADC / %d SCC, %d dimensions, %d instances"
        % (len(cohort.bags), int((labels == 1).sum()), int((labels == 0).sum()),
           len(cohort.feature_names), sum(b.n_patches for b in cohort.bags.values())))

    preflight = verify_frozen_inputs(cfg, cohort)
    for k in sorted(preflight):
        log("  preflight %s = %s" % (k, preflight[k]))

    # the frozen Phase-6B baseline this phase is judged against is READ, never retrained
    ref_rel = str(cfg["comparison"]["primary"]["reference_predictions"])
    with open(os.path.join(root, "reports", "phase6c_pre_snapshot.json"),
              "r", encoding="utf-8") as fh:
        snap = json.load(fh)["files"]
    ref_sha = sha256_file(os.path.join(root, *ref_rel.split("/")))
    if snap.get(ref_rel) != ref_sha:
        raise RuntimeError("the frozen Phase-6B baseline %s is not byte-identical to "
                           "the pre-Phase-6C snapshot" % ref_rel)
    log("  baseline %s sha256 %s… verified frozen (never retrained)"
        % (ref_rel, ref_sha[:16]))

    fold_results = [process_fold(cfg, cohort, split_df, f, log) for f in args.folds]
    write_instance_diagnostics(cfg, cohort, fold_results)

    manifest_after = input_manifest(cfg)
    total_seconds = time.time() - t_start

    rows = []
    for fr in fold_results:
        for pid, prob in fr["test_probabilities"].items():
            bag = cohort.bags[pid]
            rows.append({"PatientID": pid, "histology": bag.histology,
                         "true_label": int(bag.label),
                         "predicted_probability_ADC": float(prob),
                         "predicted_label": int(
                             prob >= float(cfg["evaluation"]["threshold"])),
                         "outer_fold": int(fr["fold"]), "model": MODEL_NAME,
                         "selected_epoch": int(fr["selected_epoch"]),
                         "selected_lambda_instance": float(fr["selected_lambda_instance"]),
                         "n_patches": bag.n_patches})
    df = pd.DataFrame(rows).sort_values("PatientID")
    pred_dir = ppath(cfg, cfg["outputs"]["predictions_dir"])
    os.makedirs(pred_dir, exist_ok=True)
    tag = model_tag(cfg, MODEL_NAME)
    df.to_csv(os.path.join(pred_dir, "%s_oof.csv" % tag), index=False)
    log("wrote %s/%s_oof.csv  (%d rows)" % (cfg["outputs"]["predictions_dir"], tag, len(df)))

    meta = {
        "phase": "6c",
        "roi": str(cfg["phase3"]["roi"]),
        "model": MODEL_NAME,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "procedure": ("stratified 3-fold inner CV, JOINT (lambda_instance x epoch) "
                      "selection from the mean inner-CV ROC-AUC surface, then a "
                      "from-scratch refit on all outer-training patients"),
        "changed_variable": "instance-level auxiliary loss only",
        "clam_is_reference_only": True,
        "lambda_grid": lambda_grid(cfg),
        "k_rule": cfg["instance_supervision"]["k_rule"],
        "folds": list(args.folds),
        "total_runtime_seconds": total_seconds,
        "selection_seconds": sum(fr["selection_seconds"] for fr in fold_results),
        "refit_seconds": sum(fr["refit_seconds"] for fr in fold_results),
        "selected_lambda_instance": {fr["fold"]: fr["selected_lambda_instance"]
                                     for fr in fold_results},
        "selected_epochs": {MODEL_NAME: {fr["fold"]: fr["selected_epoch"]
                                         for fr in fold_results}},
        "frozen_inputs_unchanged": bool(manifest_before == manifest_after),
        "frozen_baseline": {ref_rel: ref_sha, "retrained": False},
        "preflight_verification": preflight,
        "input_manifest": manifest_before,
        "folds_detail": [{k: v for k, v in fr.items() if k != "test_probabilities"}
                         for fr in fold_results],
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "torch": torch.__version__, "numpy": np.__version__,
                        "pandas": pd.__version__,
                        "device": str(cfg["training"]["device"]),
                        "torch_threads": torch.get_num_threads()},
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "config_path": cfg["_config_path"],
        "radiomics_imported_during_training": ("radiomics" in sys.modules
                                               or "SimpleITK" in sys.modules),
        "torchvision_imported_during_training": ("torchvision" in sys.modules),
    }
    with open(ppath(cfg, str(cfg["outputs"]["run_json"])), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)

    log("-" * 78)
    log("selected lambda_instance: %s" % json.dumps(meta["selected_lambda_instance"]))
    log("selected epochs: %s" % json.dumps(meta["selected_epochs"]))
    log("frozen inputs unchanged: %s" % meta["frozen_inputs_unchanged"])
    log("radiomics/SimpleITK imported: %s   torchvision imported: %s"
        % (meta["radiomics_imported_during_training"],
           meta["torchvision_imported_during_training"]))
    log("total runtime %.1f s" % total_seconds)
    log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
