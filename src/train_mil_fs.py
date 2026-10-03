"""Phase 5C - feature-selection rescue test for the GTV+Rim local-radiomics MIL.

Phase 5 trained on all 74 features and reached pooled OOF ROC-AUC 0.5702 (mean
pooling) / 0.5767 (gated attention).  Phase 5C asks whether redundant or noisy
radiomic dimensions are what stops the MIL model generalising, by testing
exactly two selectors - mRMR and LASSO - inside the nested CV.

Everything except the feature set is held at its Phase-5 value: the frozen 202
GTV+Rim bags, the frozen 5-fold outer split, the 3-fold inner CV (seed
3000+fold), the encoder (K_selected -> 64 -> 32, ReLU, dropout 0.2), Adam
lr 1e-3 / weight_decay 1e-4, weighted BCE with pos_weight from TRAINING PATIENT
counts, a 100-epoch budget with no early stopping, one bag per batch, threshold
0.5 and ADC positive.

Per outer fold, per selector, per model:

  1. outer-test patients are untouched;
  2. stratified 3-fold inner CV over the outer-TRAINING patients;
  3. for every inner split the selector is fitted on the INNER-TRAINING
     patients' 74-D patient-median summaries and their labels - one row per
     patient, never one row per patch;
  4. the selected ORIGINAL feature indices are applied back to the untouched
     patch bags; the patch scaler is fitted on the INNER-TRAINING PATCHES over
     those dimensions alone; pos_weight comes from inner-training PATIENT counts;
  5. the MIL model trains from scratch for the full 100 epochs and its
     patient-level inner-validation ROC-AUC is recorded after every epoch;
  6. the three curves are averaged, giving a (candidate x epoch) surface;
  7. the (selector setting, epoch) pair with the best mean inner-CV ROC-AUC is
     chosen - ties go to the smaller selected feature count, then the lower
     epoch, then configuration order;
  8. the selector is refitted FROM SCRATCH on ALL outer-training patients with
     that setting, the patch scaler and pos_weight are recomputed from all
     outer-training patients, the model is retrained from scratch for exactly
     the selected epoch, and the outer-test fold is scored ONCE.

No outer-test patient enters steps 2-8.  No radiomics is recomputed.  No
Phase-3/3B/4/5 artefact is read for anything but comparison, or written at all.

Usage:
    python src/train_mil_fs.py --config config/phase5c.yaml
    python src/train_mil_fs.py --config config/phase5c.yaml --folds 1 --pipelines mrmr_mean
    python src/train_mil_fs.py --config config/phase5c.yaml --status
    python src/train_mil_fs.py --config config/phase5c.yaml --assemble-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_selection import (build_patient_summaries, feature_family,  # noqa: E402
                               fit_selected_patch_scaler, lasso_select,
                               load_patient_summaries, mrmr_rank,
                               save_patient_summaries)
from mil_data import (Cohort, load_cohort, load_outer_split,  # noqa: E402
                      load_phase3_config, outer_fold_patients,
                      pos_weight_from_patients, ppath, sha256_file)
from mil_metrics import patient_metrics  # noqa: E402
from mil_models import build_model  # noqa: E402
from train_mil import predict, train_one_epoch  # noqa: E402

DEFAULT_CONFIG = "config/phase5c.yaml"
MODEL_OFFSET = {"mean_mil": 0, "attention_mil": 1}


# --------------------------------------------------------------------- helpers
class _BagSource:
    """Adapter so the frozen train_one_epoch/predict consume our tensors unchanged."""

    def __init__(self, tensors: Dict[str, np.ndarray]):
        self.tensors = tensors


def set_seeds(cfg: dict, seed: int) -> int:
    scfg = cfg["seeds"]
    random.seed(int(scfg["python"]) + seed)
    np.random.seed(int(scfg["numpy"]) + seed)
    torch.manual_seed(int(scfg["torch"]) + seed)
    if bool(scfg.get("deterministic_torch", True)):
        torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    return seed


def pipelines(cfg: dict) -> List[dict]:
    return [dict(p) for p in cfg["pipelines"]]


def pipeline_by_tag(cfg: dict, tag: str) -> dict:
    for p in pipelines(cfg):
        if p["tag"] == tag:
            return p
    raise ValueError("unknown pipeline %r" % tag)


def unit_path(cfg: dict, tag: str, fold: int) -> str:
    return os.path.join(ppath(cfg, cfg["outputs"]["checkpoint_dir"]),
                        "%s_fold_%d.json" % (tag, fold))


def training_fingerprint(cfg: dict) -> str:
    blob = json.dumps({"model": cfg["model"], "training": cfg["training"],
                       "preprocessing": cfg["preprocessing"], "seeds": cfg["seeds"],
                       "threshold": cfg["evaluation"]["threshold"]},
                      sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def inner_cv_assignment(cfg: dict, cohort: Cohort, fold: int,
                        train_pids: Sequence[str]) -> Dict[str, int]:
    """The Phase-5 inner-CV structure, verbatim: stratified K-fold, seed 3000+fold."""
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
        if len(np.unique(cohort.labels(members))) < 2:
            raise RuntimeError("split-generation failure: inner fold %d of outer fold %d "
                               "is single-class" % (j, fold))
    return assign


# ------------------------------------------------------- selector fitting layer
def fit_selector_inner(cfg: dict, summaries, selector: str, fit_pids: Sequence[str]) -> dict:
    """Fit `selector` on EXACTLY the patients in `fit_pids` (one summary row each).

    Returns a JSON-serialisable record with, per candidate, the ORIGINAL feature
    indices to retain.  Nothing outside `fit_pids` is read.
    """
    scfg = cfg["selectors"][selector]
    x, y = summaries.subset(fit_pids)               # one row per fitting patient
    names = summaries.feature_names
    rec = {"selector": selector, "n_fit_patients": int(x.shape[0]),
           "n_fit_adc": int((y == 1).sum()), "n_fit_scc": int((y == 0).sum()),
           "fit_patients": list(fit_pids),
           "candidates_key": str(scfg["candidates_key"]), "candidates": {}}

    if selector == "mrmr":
        k_max = int(max(scfg["candidates"]))
        rank = mrmr_rank(x, y, k_max,
                         float(scfg["zero_variance_relevance"]),
                         float(scfg["zero_variance_correlation"]))
        rec["ranking"] = list(rank.order)
        rec["ranking_names"] = [names[i] for i in rank.order]
        # ALL 74 original dimensions are scored and offered to the greedy step, so
        # the audit record keeps the whole relevance vector, not the ranked prefix
        rec["relevance_f"] = {names[i]: float(rank.relevance[i])
                              for i in range(len(names))}
        rec["steps"] = rank.steps
        rec["criterion"] = str(scfg["criterion"])
        for k in scfg["candidates"]:
            sel = rank.select(int(k))
            rec["candidates"][str(k)] = {
                "setting": int(k), "valid": True, "reason": "ok",
                "selected_indices": sorted(int(i) for i in sel),
                "selected_names": [names[i] for i in sorted(sel)],
                "selection_order": [int(i) for i in sel],
                "selection_order_names": [names[i] for i in sel],
                "n_selected": len(sel)}
        return rec

    if selector == "lasso":
        for c in scfg["candidates"]:
            ls = lasso_select(x, y, float(c), scfg)
            rec["candidates"]["%g" % c] = {
                "setting": float(c), "valid": bool(ls.valid), "reason": ls.reason,
                "selected_indices": sorted(int(i) for i in ls.selected),
                "selected_names": [names[i] for i in sorted(ls.selected)],
                "n_selected": int(ls.n_selected),
                "coefficients": {names[i]: float(ls.coefficients[i])
                                 for i in sorted(ls.selected)},
                "intercept": float(ls.intercept), "n_iter": int(ls.n_iter),
                "summary_scaler_mean": [float(v) for v in ls.scaler_mean],
                "summary_scaler_std": [float(v) for v in ls.scaler_std],
                "summary_scaler_source": "fitting_patient_summaries_only"}
        return rec

    raise ValueError("unknown selector %r" % selector)


# ------------------------------------------------------------ MIL training layer
def run_cache_key(cfg: dict, fold: int, model_name: str, train_pids: Sequence[str],
                  eval_pids: Sequence[str], selected: Sequence[int], seed: int,
                  n_epochs: int) -> str:
    blob = json.dumps({
        "fold": int(fold), "model": model_name, "seed": int(seed),
        "n_epochs": int(n_epochs),
        "train": sorted(train_pids), "eval": sorted(eval_pids),
        "selected": sorted(int(i) for i in selected),
        "training_fingerprint": training_fingerprint(cfg),
    }, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def train_mil_run(cfg: dict, cohort: Cohort, model_name: str, train_pids: Sequence[str],
                  eval_pids: Sequence[str], selected: Sequence[int], seed: int,
                  n_epochs: int, eval_every_epoch: bool):
    """Fit the patch scaler on TRAINING patches over the selected dimensions, then
    train from scratch for exactly `n_epochs`."""
    tcfg = cfg["training"]
    device = torch.device(str(tcfg.get("device", "cpu")))
    scaler = fit_selected_patch_scaler(cohort, train_pids, selected,
                                       float(cfg["preprocessing"]["min_std"]))
    pw = pos_weight_from_patients(cohort.labels(train_pids))    # PATIENT counts
    tensors = {p: scaler.transform(cohort.bags[p].features)
               for p in list(train_pids) + list(eval_pids)}
    source = _BagSource(tensors)

    set_seeds(cfg, seed)
    model = build_model(model_name, cfg, input_dim=len(scaler.selected)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tcfg["learning_rate"]),
                                 weight_decay=float(tcfg["weight_decay"]))
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pw], dtype=torch.float32, device=device))
    rng = random.Random(seed)
    y_eval = cohort.labels(eval_pids) if eval_every_epoch else None
    thr = float(cfg["evaluation"]["threshold"])
    curve: List[dict] = []

    for epoch in range(1, int(n_epochs) + 1):
        order = list(train_pids)
        rng.shuffle(order)
        loss = train_one_epoch(model, source, cohort, order, optimizer, loss_fn,
                               device, int(tcfg["bags_per_batch"]))
        row = {"epoch": epoch, "train_loss": float(loss)}
        if eval_every_epoch:
            prob, _ = predict(model, source, eval_pids, device)
            if not np.all(np.isfinite(prob)):
                raise RuntimeError("non-finite validation probabilities at epoch %d" % epoch)
            if len(np.unique(y_eval)) < 2:
                raise RuntimeError("single-class inner validation set")
            m = patient_metrics(y_eval, prob, thr)
            row["val_roc_auc"] = m["roc_auc"]
            row["val_pr_auc"] = m["pr_auc"]
        curve.append(row)
    return model, curve, source, scaler, pw


def cached_inner_curve(cfg: dict, cohort: Cohort, fold: int, model_name: str,
                       train_pids: Sequence[str], eval_pids: Sequence[str],
                       selected: Sequence[int], seed: int, n_epochs: int,
                       log) -> Tuple[np.ndarray, np.ndarray, bool]:
    """The per-epoch inner-validation ROC-AUC curve, reused if already computed.

    Two candidate settings that select the IDENTICAL feature set produce an
    identical run (the seeds deliberately do not depend on the selector or the
    setting), so the cache is a pure speed-up, never a shortcut.
    """
    cdir = ppath(cfg, cfg["outputs"]["inner_cache_dir"])
    os.makedirs(cdir, exist_ok=True)
    key = run_cache_key(cfg, fold, model_name, train_pids, eval_pids, selected,
                        seed, n_epochs)
    path = os.path.join(cdir, "%s.npz" % key)
    if os.path.exists(path):
        try:
            with np.load(path, allow_pickle=True) as d:
                return (np.asarray(d["val_roc_auc"], dtype=float),
                        np.asarray(d["train_loss"], dtype=float), True)
        except Exception:                                        # noqa: BLE001
            os.remove(path)

    _, curve, _, _, _ = train_mil_run(cfg, cohort, model_name, train_pids, eval_pids,
                                      selected, seed, n_epochs, eval_every_epoch=True)
    auc = np.array([r["val_roc_auc"] for r in curve], dtype=float)
    lss = np.array([r["train_loss"] for r in curve], dtype=float)
    np.savez_compressed(path, val_roc_auc=auc, train_loss=lss,
                        val_pr_auc=np.array([r["val_pr_auc"] for r in curve], dtype=float),
                        fold=np.int32(fold), model=np.str_(model_name),
                        seed=np.int32(seed),
                        selected=np.array(sorted(int(i) for i in selected), dtype=np.int32),
                        train_patients=np.array(sorted(train_pids), dtype=object),
                        eval_patients=np.array(sorted(eval_pids), dtype=object),
                        training_fingerprint=np.str_(training_fingerprint(cfg)))
    return auc, lss, False


# -------------------------------------------------- joint selector/epoch choice
def choose_setting_and_epoch(surface: Dict[str, np.ndarray],
                             mean_n_selected: Dict[str, float],
                             order: List[str]) -> Tuple[str, int, float, dict]:
    """Argmax of the mean inner-CV ROC-AUC over the whole (candidate x epoch) grid.

    Ties, in order: (1) smaller selected feature count / smaller K,
    (2) lower epoch, (3) configuration order.  Purely inner-CV; the outer test
    never enters.
    """
    best = None
    for rank, key in enumerate(order):
        curve = surface[key]
        if not np.all(np.isfinite(curve)):
            raise RuntimeError("non-finite selection curve for candidate %s" % key)
        epoch = int(np.argmax(curve)) + 1            # first maximum -> lowest epoch
        auc = float(curve[epoch - 1])
        cand = (-auc, float(mean_n_selected[key]), epoch, rank, key)
        if best is None or cand < best:
            best = cand
    neg_auc, n_sel, epoch, rank, key = best
    detail = {"selected_candidate": key, "selected_epoch": int(epoch),
              "mean_inner_cv_roc_auc": float(-neg_auc),
              "mean_n_selected_features": float(n_sel),
              "tie_break": ["smaller_selected_feature_count", "lower_epoch",
                            "config_order"],
              "per_candidate_best": {k: {"epoch": int(np.argmax(surface[k])) + 1,
                                         "mean_inner_cv_roc_auc": float(surface[k].max()),
                                         "mean_n_selected_features":
                                             float(mean_n_selected[k])}
                                     for k in order}}
    return key, int(epoch), float(-neg_auc), detail


# ------------------------------------------------------------------- one unit
def process_unit(cfg: dict, cohort: Cohort, summaries, split_df: pd.DataFrame,
                 fold: int, pipe: dict, selector_cache: dict, log) -> dict:
    tag, selector, model_name = pipe["tag"], pipe["selector"], pipe["model"]
    scfg = cfg["selectors"][selector]
    names = cohort.feature_names
    n_inner = int(cfg["inner_cv"]["n_splits"])
    max_epochs = int(cfg["training"]["max_epochs"])
    offset = MODEL_OFFSET[model_name]
    t_unit = time.time()

    train_pids, test_pids = outer_fold_patients(split_df, fold)
    train_pids, test_pids = sorted(train_pids), sorted(test_pids)
    assign = inner_cv_assignment(cfg, cohort, fold, train_pids)
    y_tr = cohort.labels(train_pids)

    log("-" * 78)
    log("fold %d | %s | outer_train %d (%d ADC/%d SCC) outer_test %d"
        % (fold, tag, len(train_pids), int((y_tr == 1).sum()), int((y_tr == 0).sum()),
           len(test_pids)))

    # ---- selector fitted per inner split, on INNER-TRAINING patients only
    ckey = (fold, selector)
    if ckey not in selector_cache:
        inner_records = {}
        for j in range(1, n_inner + 1):
            itr = sorted(p for p in train_pids if assign[p] != j)
            iva = sorted(p for p in train_pids if assign[p] == j)
            if set(itr) & set(test_pids) or set(iva) & set(test_pids):
                raise RuntimeError("outer-test patient inside an inner split")
            rec = fit_selector_inner(cfg, summaries, selector, itr)
            rec["inner_fold"] = j
            rec["inner_val_patients"] = iva
            inner_records[j] = rec
        selector_cache[ckey] = inner_records
        save_inner_selection(cfg, fold, selector, inner_records, train_pids, test_pids)
    inner_records = selector_cache[ckey]

    # ---- candidate validity: a setting is evaluable only if valid in EVERY inner split
    order, invalid = [], {}
    for cand_key in inner_records[1]["candidates"]:
        states = [inner_records[j]["candidates"][cand_key] for j in range(1, n_inner + 1)]
        bad = [j for j, s in zip(range(1, n_inner + 1), states) if not s["valid"]]
        if bad:
            invalid[cand_key] = {
                "invalid_in_inner_folds": bad,
                "reasons": [states[j - 1]["reason"] for j in bad],
                "note": ("candidate excluded from selection for this fold/model; "
                         "features are never substituted")}
        else:
            order.append(cand_key)
    if not order:
        raise RuntimeError("every %s candidate is invalid in outer fold %d" % (selector, fold))
    if invalid:
        log("  invalid candidates: %s" % json.dumps(invalid, default=str)[:300])

    # ---- inner MIL runs -> (candidate x epoch) surface
    surface, per_candidate, mean_n_selected = {}, {}, {}
    n_trained = n_reused = 0
    for cand_key in order:
        curves, counts = [], []
        for j in range(1, n_inner + 1):
            itr = sorted(p for p in train_pids if assign[p] != j)
            iva = sorted(p for p in train_pids if assign[p] == j)
            sel = inner_records[j]["candidates"][cand_key]["selected_indices"]
            counts.append(len(sel))
            seed = int(cfg["seeds"]["inner_base_seed"]) + 100 * fold + 10 * j + offset
            auc, loss, reused = cached_inner_curve(cfg, cohort, fold, model_name, itr,
                                                   iva, sel, seed, max_epochs, log)
            n_reused += int(reused)
            n_trained += int(not reused)
            curves.append({"auc": auc, "loss": loss, "n_selected": len(sel), "seed": seed})
        aucs = np.vstack([c["auc"] for c in curves])
        surface[cand_key] = aucs.mean(axis=0)
        mean_n_selected[cand_key] = float(np.mean(counts))
        per_candidate[cand_key] = {
            "setting": inner_records[1]["candidates"][cand_key]["setting"],
            "n_selected_per_inner_fold": counts,
            "mean_n_selected": float(np.mean(counts)),
            "seeds": [c["seed"] for c in curves],
            "inner_auc": aucs, "inner_loss": np.vstack([c["loss"] for c in curves]),
            "mean_curve": surface[cand_key],
            "sd_curve": aucs.std(axis=0, ddof=1),
            "best_mean_inner_cv_roc_auc": float(surface[cand_key].max()),
            "best_epoch": int(np.argmax(surface[cand_key])) + 1,
            "per_inner_argmax": [int(np.argmax(c["auc"])) + 1 for c in curves]}
        log("    %-6s %-6s mean inner-CV ROC-AUC %.4f @ epoch %d (K=%s)"
            % (selector, cand_key, surface[cand_key].max(),
               int(np.argmax(surface[cand_key])) + 1, counts))

    chosen, sel_epoch, sel_auc, detail = choose_setting_and_epoch(
        surface, mean_n_selected, order)
    detail["invalid_candidates"] = invalid
    save_surface(cfg, fold, tag, per_candidate, order, detail)
    log("  -> chose %s=%s, epoch %d (mean inner-CV ROC-AUC %.4f)"
        % (scfg["candidates_key"], chosen, sel_epoch, sel_auc))

    # ---- FINAL: refit the selector from scratch on ALL outer-training patients
    final_rec = fit_selector_inner(cfg, summaries, selector, train_pids)
    final_cand = final_rec["candidates"][chosen]
    if not final_cand["valid"]:
        raise RuntimeError("the chosen %s candidate %s selects zero features on the full "
                           "outer-training set of fold %d" % (selector, chosen, fold))
    selected = sorted(int(i) for i in final_cand["selected_indices"])
    log("  final selector on all %d outer-train patients: %d features"
        % (len(train_pids), len(selected)))

    final_seed = int(cfg["seeds"]["final_base_seed"]) + 10 * fold + offset
    t1 = time.time()
    model, _, source, scaler, pw = train_mil_run(
        cfg, cohort, model_name, train_pids, test_pids, selected, final_seed,
        sel_epoch, eval_every_epoch=False)
    device = torch.device(str(cfg["training"]["device"]))
    test_prob, _ = predict(model, source, test_pids, device)
    refit_seconds = time.time() - t1
    if not np.all(np.isfinite(test_prob)):
        raise RuntimeError("non-finite test probabilities, fold %d, %s" % (fold, tag))
    y_test = cohort.labels(test_pids)
    fm = patient_metrics(y_test, test_prob, float(cfg["evaluation"]["threshold"]))
    log("  fold %d %s: test ROC-AUC %.4f PR-AUC %.4f balacc %.4f MCC %.4f"
        % (fold, tag, fm["roc_auc"], fm["pr_auc"], fm["balanced_accuracy"], fm["mcc"]))

    save_final_selection(cfg, fold, tag, selector, model_name, chosen, final_rec,
                         final_cand, selected, names, train_pids, test_pids)
    save_patch_scaler(cfg, fold, tag, scaler, pw, selected, names, train_pids, test_pids,
                      inner_records, assign, chosen)

    ck_dir = ppath(cfg, "%s/%s%s" % (cfg["outputs"]["models_dir"],
                                     cfg["outputs"]["model_prefix"], tag))
    os.makedirs(ck_dir, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(), "model_name": model_name,
        "pipeline": tag, "selector": selector, "fold": fold, "phase": "5c",
        "roi": str(cfg["phase3"]["roi"]),
        "selector_setting_key": str(scfg["candidates_key"]),
        "selector_setting": final_cand["setting"],
        "selected_indices": selected,
        "selected_features": [names[i] for i in selected],
        "input_dim": len(selected),
        "scaler_mean": scaler.mean, "scaler_std": scaler.std,
        "selected_epoch": sel_epoch, "selection_mean_inner_cv_roc_auc": sel_auc,
        "pos_weight": pw, "seed": final_seed,
        "inner_cv_seed": int(cfg["inner_cv"]["base_seed"]) + fold,
        "config": {"model": cfg["model"], "training": cfg["training"],
                   "inner_cv": cfg["inner_cv"], "seeds": cfg["seeds"],
                   "selector": scfg},
    }, os.path.join(ck_dir, "fold_%d.pt" % fold))

    unit = {
        "phase": "5c", "pipeline": tag, "selector": selector, "model": model_name,
        "fold": fold, "n_outer_train": len(train_pids), "n_outer_test": len(test_pids),
        "inner_cv_seed": int(cfg["inner_cv"]["base_seed"]) + fold,
        "selector_setting_key": str(scfg["candidates_key"]),
        "selected_candidate": chosen,
        "selected_setting": final_cand["setting"],
        "selected_epoch": sel_epoch,
        "selection_mean_inner_cv_roc_auc": sel_auc,
        "selection_detail": {k: v for k, v in detail.items()},
        "n_selected_features": len(selected),
        "selected_indices": selected,
        "selected_features": [names[i] for i in selected],
        "selected_families": {names[i]: feature_family(names[i]) for i in selected},
        "n_selected_per_inner_fold": per_candidate[chosen]["n_selected_per_inner_fold"],
        "final_pos_weight": float(pw),
        "final_pos_weight_definition":
            "n_scc_outer_train_patients / n_adc_outer_train_patients",
        "final_patch_scaler_fingerprint": scaler.fingerprint(),
        "final_scaler_source": "all_outer_train_patches_selected_dimensions_only",
        "final_seed": final_seed, "refit_seconds": refit_seconds,
        "unit_seconds": time.time() - t_unit,
        "inner_runs_trained": n_trained, "inner_runs_reused_from_cache": n_reused,
        "fold_metrics": fm,
        "test_probabilities": {p: float(v) for p, v in zip(test_pids, test_prob)},
        "outer_test_patients": test_pids,
        "training_fingerprint": training_fingerprint(cfg),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = unit_path(cfg, tag, fold)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(unit, fh, indent=2, default=float)
    return unit


# ------------------------------------------------------------------ persistence
def save_inner_selection(cfg: dict, fold: int, selector: str, inner_records: dict,
                         train_pids: Sequence[str], test_pids: Sequence[str]) -> None:
    out_dir = os.path.join(ppath(cfg, cfg["outputs"]["feature_selection_dir"]), selector)
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "phase": "5c", "selector": selector, "outer_fold": fold,
        "scope": "inner_cv",
        "leakage_statement": ("each record was fitted on that inner split's "
                              "INNER-TRAINING patients only; the outer-test patients "
                              "of this fold appear nowhere below"),
        "n_outer_train": len(train_pids), "n_outer_test": len(test_pids),
        "outer_test_patients": sorted(test_pids),
        "inner": {str(j): inner_records[j] for j in sorted(inner_records)},
    }
    with open(os.path.join(out_dir, "fold_%d_inner.json" % fold), "w",
              encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)


def save_final_selection(cfg: dict, fold: int, tag: str, selector: str, model_name: str,
                         chosen: str, final_rec: dict, final_cand: dict,
                         selected: Sequence[int], names: Sequence[str],
                         train_pids: Sequence[str], test_pids: Sequence[str]) -> None:
    out_dir = os.path.join(ppath(cfg, cfg["outputs"]["feature_selection_dir"]), selector)
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "phase": "5c", "selector": selector, "pipeline": tag, "model": model_name,
        "outer_fold": fold, "scope": "final_refit",
        "leakage_statement": ("refitted FROM SCRATCH on ALL outer-training patients; "
                              "no outer-test patient took part"),
        "chosen_candidate": chosen, "setting": final_cand["setting"],
        "n_fit_patients": final_rec["n_fit_patients"],
        "n_fit_adc": final_rec["n_fit_adc"], "n_fit_scc": final_rec["n_fit_scc"],
        "fit_patients": sorted(train_pids),
        "outer_test_patients": sorted(test_pids),
        "n_selected": len(selected),
        "selected_indices": list(selected),
        "selected_features": [names[i] for i in selected],
        "selected_families": {names[i]: feature_family(names[i]) for i in selected},
        "candidate_record": final_cand,
    }
    if selector == "mrmr":
        payload["ranking"] = final_rec["ranking"]
        payload["ranking_names"] = final_rec["ranking_names"]
        payload["relevance_f"] = final_rec["relevance_f"]
        payload["steps"] = final_rec["steps"]
    with open(os.path.join(out_dir, "fold_%d_%s_final.json" % (fold, tag)), "w",
              encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)


def save_patch_scaler(cfg: dict, fold: int, tag: str, scaler, pw: float,
                      selected: Sequence[int], names: Sequence[str],
                      train_pids: Sequence[str], test_pids: Sequence[str],
                      inner_records: dict, assign: Dict[str, int], chosen: str) -> None:
    prep = ppath(cfg, cfg["outputs"]["preprocessing_dir"])
    os.makedirs(prep, exist_ok=True)
    np.savez_compressed(
        os.path.join(prep, "%s_fold_%d_final_scaler.npz" % (tag, fold)),
        selected_indices=np.asarray(selected, dtype=np.int32),
        selected_features=np.array([names[i] for i in selected], dtype=object),
        mean=scaler.mean, std=scaler.std, pos_weight=np.float64(pw),
        fold=np.int32(fold), pipeline=np.str_(tag))
    meta = {
        "phase": "5c", "pipeline": tag, "fold": fold,
        "n_selected": len(selected),
        "selected_features": [names[i] for i in selected],
        "final_scaler_source": "all_outer_train_patches_selected_dimensions_only",
        "n_train_patients": scaler.n_train_patients,
        "n_train_patches": scaler.n_train_patches,
        "final_pos_weight": float(pw),
        "final_pos_weight_definition":
            "n_scc_outer_train_patients / n_adc_outer_train_patients",
        "fingerprint": scaler.fingerprint(),
        "inner": {str(j): {
            "inner_train_patients": [p for p in sorted(train_pids) if assign[p] != j],
            "inner_val_patients": [p for p in sorted(train_pids) if assign[p] == j],
            "n_selected_chosen_candidate":
                len(inner_records[j]["candidates"][chosen]["selected_indices"]),
            "selected_indices_chosen_candidate":
                inner_records[j]["candidates"][chosen]["selected_indices"],
        } for j in sorted(inner_records)},
    }
    with open(os.path.join(prep, "%s_fold_%d_scaler.json" % (tag, fold)), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=float)


def save_surface(cfg: dict, fold: int, tag: str, per_candidate: dict,
                 order: List[str], detail: dict) -> None:
    out_dir = ppath(cfg, cfg["outputs"]["surfaces_dir"])
    os.makedirs(out_dir, exist_ok=True)
    n_epochs = len(per_candidate[order[0]]["mean_curve"])
    df = pd.DataFrame({"epoch": np.arange(1, n_epochs + 1)})
    for key in order:
        pc = per_candidate[key]
        for j in range(pc["inner_auc"].shape[0]):
            df["%s_inner_%d_val_roc_auc" % (key, j + 1)] = pc["inner_auc"][j]
        df["%s_mean_val_roc_auc" % key] = pc["mean_curve"]
        df["%s_sd_val_roc_auc" % key] = pc["sd_curve"]
    df.to_csv(os.path.join(out_dir, "%s_fold_%d_surface.csv" % (tag, fold)), index=False)

    summary = {"phase": "5c", "pipeline": tag, "outer_fold": fold,
               "candidates_evaluated": order,
               "selection": {k: v for k, v in detail.items()},
               "per_candidate": {k: {
                   "setting": per_candidate[k]["setting"],
                   "n_selected_per_inner_fold": per_candidate[k]["n_selected_per_inner_fold"],
                   "mean_n_selected": per_candidate[k]["mean_n_selected"],
                   "seeds": per_candidate[k]["seeds"],
                   "best_mean_inner_cv_roc_auc": per_candidate[k]["best_mean_inner_cv_roc_auc"],
                   "best_epoch": per_candidate[k]["best_epoch"],
                   "per_inner_argmax": per_candidate[k]["per_inner_argmax"],
                   "mean_curve_at": {str(e): float(per_candidate[k]["mean_curve"][e - 1])
                                     for e in (1, 5, 10, 25, 50, 75, 100)
                                     if e <= n_epochs},
               } for k in order}}
    with open(os.path.join(out_dir, "%s_fold_%d_selection.json" % (tag, fold)), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)


def save_inner_split_table(cfg: dict, cohort: Cohort, split_df: pd.DataFrame,
                           fold: int) -> None:
    out_dir = ppath(cfg, cfg["outputs"]["inner_splits_dir"])
    os.makedirs(out_dir, exist_ok=True)
    train_pids, test_pids = outer_fold_patients(split_df, fold)
    assign = inner_cv_assignment(cfg, cohort, fold, train_pids)
    rows = []
    for pid in sorted(train_pids):
        rows.append({"fold": fold, "PatientID": pid,
                     "Histology": cohort.bags[pid].histology,
                     "label": cohort.bags[pid].label, "n_patches": cohort.bags[pid].n_patches,
                     "assignment": "outer_train", "inner_fold": assign[pid]})
    for pid in sorted(test_pids):
        rows.append({"fold": fold, "PatientID": pid,
                     "Histology": cohort.bags[pid].histology,
                     "label": cohort.bags[pid].label, "n_patches": cohort.bags[pid].n_patches,
                     "assignment": "outer_test", "inner_fold": 0})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "fold_%d.csv" % fold), index=False)


def input_manifest(cfg: dict) -> dict:
    feat_dir = ppath(cfg, cfg["phase3"]["features_dir"])
    bags = {f: sha256_file(os.path.join(feat_dir, f))
            for f in sorted(os.listdir(feat_dir)) if f.endswith(".npz")}
    base = cfg["baseline"]["model_files"]
    return {"n_bags": len(bags), "bag_sha256": bags,
            "splits_file_sha256": sha256_file(ppath(cfg, cfg["phase3"]["splits_file"])),
            "feature_names_sha256": sha256_file(
                ppath(cfg, cfg["phase3"]["feature_names_file"])),
            "baseline_predictions_sha256": {
                k: sha256_file(ppath(cfg, v)) for k, v in base.items()}}


# --------------------------------------------------------------------- assembly
def all_units(cfg: dict, folds: Sequence[int]) -> Dict[str, Dict[int, str]]:
    return {p["tag"]: {f: unit_path(cfg, p["tag"], f) for f in folds}
            for p in pipelines(cfg)}


def load_unit(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                                            # noqa: BLE001
        return None


def print_status(cfg: dict, folds: Sequence[int]) -> int:
    units = all_units(cfg, folds)
    total = done = 0
    print("PHASE 5C status  (%s)" % time.strftime("%Y-%m-%d %H:%M:%S"))
    for tag, byfold in units.items():
        marks = []
        for f in sorted(byfold):
            total += 1
            u = load_unit(byfold[f])
            ok = u is not None
            done += int(ok)
            marks.append("%d:%s" % (f, ("%.3f" % u["fold_metrics"]["roc_auc"]) if ok else "-"))
        print("  %-16s %s" % (tag, "  ".join(marks)))
    cdir = ppath(cfg, cfg["outputs"]["inner_cache_dir"])
    n_cache = len([f for f in os.listdir(cdir)]) if os.path.isdir(cdir) else 0
    print("  units complete: %d / %d   cached inner runs: %d" % (done, total, n_cache))
    return 0 if done == total else 1


def assemble(cfg: dict, folds: Sequence[int], log) -> Optional[dict]:
    units = all_units(cfg, folds)
    loaded, missing = {}, []
    for tag, byfold in units.items():
        loaded[tag] = {}
        for f in sorted(byfold):
            u = load_unit(byfold[f])
            if u is None:
                missing.append("%s/fold_%d" % (tag, f))
            else:
                loaded[tag][f] = u
    if missing:
        log("assembly skipped: %d unit(s) still missing (%s)"
            % (len(missing), ", ".join(missing[:8])))
        return None

    cohort_labels = None
    pred_dir = ppath(cfg, cfg["outputs"]["predictions_dir"])
    os.makedirs(pred_dir, exist_ok=True)
    prefix = str(cfg["outputs"]["model_prefix"])
    thr = float(cfg["evaluation"]["threshold"])
    # the frozen split is long-format (one row per patient PER FOLD), so the
    # per-patient label/histology must be de-duplicated before indexing
    bags = pd.read_csv(ppath(cfg, cfg["phase3"]["splits_file"]))
    bags = bags[["PatientID", "Histology", "label"]].drop_duplicates()
    if bags["PatientID"].duplicated().any():
        raise RuntimeError("the frozen split disagrees with itself about a patient label")
    bags = bags.set_index("PatientID")

    npz_dir = ppath(cfg, cfg["phase3"]["features_dir"])
    n_patches = {}
    for fname in sorted(os.listdir(npz_dir)):
        if fname.endswith(".npz"):
            with np.load(os.path.join(npz_dir, fname), allow_pickle=True) as d:
                n_patches[str(d["patient_id"])] = int(d["features"].shape[0])

    for tag in loaded:
        rows = []
        for f in sorted(loaded[tag]):
            u = loaded[tag][f]
            for pid, prob in u["test_probabilities"].items():
                rows.append({
                    "PatientID": pid, "histology": bags.loc[pid, "Histology"],
                    "true_label": int(bags.loc[pid, "label"]),
                    "predicted_probability_ADC": float(prob),
                    "predicted_label": int(float(prob) >= thr),
                    "outer_fold": int(f), "model": tag,
                    "selector": u["selector"], "mil_model": u["model"],
                    "selected_setting": u["selected_setting"],
                    "n_selected_features": int(u["n_selected_features"]),
                    "selected_epoch": int(u["selected_epoch"]),
                    "n_patches": int(n_patches[pid])})
        df = pd.DataFrame(rows).sort_values("PatientID")
        if df["PatientID"].duplicated().any():
            raise RuntimeError("duplicate out-of-fold predictions in %s" % tag)
        if len(df) != int(cfg["phase3"]["expected_patients"]):
            raise RuntimeError("%s produced %d predictions, expected %d"
                               % (tag, len(df), cfg["phase3"]["expected_patients"]))
        out = os.path.join(pred_dir, "%s%s_oof.csv" % (prefix, tag))
        df.to_csv(out, index=False)
        log("wrote %s  (%d rows)" % (os.path.relpath(out, cfg["_project_root"]), len(df)))
        cohort_labels = df["true_label"].values

    meta = {
        "phase": "5c",
        "roi": str(cfg["phase3"]["roi"]),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "procedure": ("patient-level feature selection (mRMR / LASSO) fitted inside the "
                      "Phase-5 nested CV, joint (selector setting, epoch) choice from the "
                      "mean inner-CV ROC-AUC surface, selector refitted from scratch on "
                      "all outer-training patients, MIL retrained from scratch, one "
                      "outer-test evaluation"),
        "pipelines": sorted(loaded),
        "folds": sorted(folds),
        "n_patients": int(cfg["phase3"]["expected_patients"]),
        "n_adc": int((cohort_labels == 1).sum()),
        "n_scc": int((cohort_labels == 0).sum()),
        "selected_settings": {t: {f: loaded[t][f]["selected_setting"] for f in sorted(loaded[t])}
                              for t in sorted(loaded)},
        "selected_epochs": {t: {f: loaded[t][f]["selected_epoch"] for f in sorted(loaded[t])}
                            for t in sorted(loaded)},
        "n_selected_features": {t: {f: loaded[t][f]["n_selected_features"]
                                    for f in sorted(loaded[t])} for t in sorted(loaded)},
        "unit_seconds": {t: {f: loaded[t][f]["unit_seconds"] for f in sorted(loaded[t])}
                         for t in sorted(loaded)},
        "inner_runs_trained": int(sum(loaded[t][f]["inner_runs_trained"]
                                      for t in loaded for f in loaded[t])),
        "inner_runs_reused_from_cache": int(sum(loaded[t][f]["inner_runs_reused_from_cache"]
                                                for t in loaded for f in loaded[t])),
        "training_fingerprint": training_fingerprint(cfg),
        "input_manifest": input_manifest(cfg),
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "torch": torch.__version__, "numpy": np.__version__,
                        "pandas": pd.__version__,
                        "device": str(cfg["training"]["device"]),
                        "torch_threads": torch.get_num_threads()},
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "config_path": cfg["_config_path"],
        "radiomics_imported_during_training": ("radiomics" in sys.modules
                                               or "SimpleITK" in sys.modules),
    }
    with open(ppath(cfg, cfg["outputs"]["run_json"]), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    log("wrote %s" % cfg["outputs"]["run_json"])
    return meta


# ---------------------------------------------------------------------- driver
def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5C feature-selection MIL")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--folds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    ap.add_argument("--pipelines", nargs="*", default=None)
    ap.add_argument("--status", action="store_true", help="print progress and exit")
    ap.add_argument("--assemble-only", action="store_true",
                    help="rebuild the OOF prediction files and run record, no training")
    ap.add_argument("--force", action="store_true", help="recompute completed units")
    ap.add_argument("--tag", default=None, help="suffix for the log file name")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_phase3_config(os.path.join(root, *args.config.split("/")))
    all_folds = [1, 2, 3, 4, 5]
    tags = ([p["tag"] for p in pipelines(cfg)] if args.pipelines is None
            else list(args.pipelines))
    for t in tags:
        pipeline_by_tag(cfg, t)

    if args.status:
        return print_status(cfg, all_folds)

    log_rel = str(cfg["outputs"]["training_log"])
    if args.tag:
        log_rel = log_rel.replace(".log", "_%s.log" % args.tag)
    log_path = ppath(cfg, log_rel)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_fh = open(log_path, "a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    t_start = time.time()
    log("=" * 78)
    log("PHASE 5C - feature-selection rescue test (ROI %s)  %s"
        % (cfg["phase3"]["roi"], time.strftime("%Y-%m-%d %H:%M:%S")))
    log("pipelines=%s folds=%s config=%s" % (tags, args.folds, args.config))

    if args.assemble_only:
        assemble(cfg, all_folds, log)
        log_fh.close()
        return 0

    # ---- frozen inputs, verified before anything is written
    meta = json.load(open(ppath(cfg, cfg["phase3"]["splits_meta_file"]), "r",
                          encoding="utf-8"))
    now = sha256_file(ppath(cfg, cfg["phase3"]["splits_file"]))
    if now != meta["splits_file_sha256"]:
        raise RuntimeError("the frozen outer split has changed: %s != %s"
                           % (now[:16], meta["splits_file_sha256"][:16]))
    log("frozen outer split sha256 %s - verified against its metadata" % now[:16])

    manifest_before = input_manifest(cfg)
    cohort = load_cohort(cfg)
    split_df = load_outer_split(cfg)
    p3 = cfg["phase3"]
    labels = cohort.labels(cohort.patient_ids)
    n_patches_total = sum(b.n_patches for b in cohort.bags.values())
    assert len(cohort.bags) == int(p3["expected_patients"])
    assert int((labels == 1).sum()) == int(p3["expected_adc"])
    assert int((labels == 0).sum()) == int(p3["expected_scc"])
    assert len(cohort.feature_names) == int(p3["expected_features"])
    assert n_patches_total == int(p3["expected_patches"])
    for b in cohort.bags.values():
        if b.config_signature != str(p3["expected_extraction_signature"]):
            raise RuntimeError("extraction signature mismatch on %s" % b.patient_id)
    log("cohort: %d bags, %d ADC / %d SCC, %d features, %d patches, signature %s"
        % (len(cohort.bags), int((labels == 1).sum()), int((labels == 0).sum()),
           len(cohort.feature_names), n_patches_total,
           p3["expected_extraction_signature"]))

    # ---- patient-level summaries: one row per patient, from that patient's bag alone
    summaries = build_patient_summaries(cohort, str(cfg["patient_summary"]["statistic"]))
    if summaries.matrix.shape != (int(p3["expected_patients"]), int(p3["expected_features"])):
        raise RuntimeError("patient summary matrix is %r" % (summaries.matrix.shape,))
    sm = save_patient_summaries(cfg, summaries, ppath, manifest_before["bag_sha256"])
    log("patient summaries: %d rows x %d features (%s), fingerprint %s, bags %d-%d patches"
        % (sm["n_patients"], sm["n_features"], sm["statistic"], sm["fingerprint"],
           sm["min_bag"], sm["max_bag"]))

    for f in args.folds:
        save_inner_split_table(cfg, cohort, split_df, f)

    selector_cache: dict = {}
    n_done = n_skipped = 0
    for f in args.folds:
        for t in tags:
            path = unit_path(cfg, t, f)
            if not args.force and load_unit(path) is not None:
                log("skip %s fold %d - already complete (%s)"
                    % (t, f, os.path.relpath(path, root)))
                n_skipped += 1
                continue
            process_unit(cfg, cohort, summaries, split_df, f,
                         pipeline_by_tag(cfg, t), selector_cache, log)
            n_done += 1

    manifest_after = input_manifest(cfg)
    if manifest_before != manifest_after:
        raise RuntimeError("a frozen input changed during the Phase-5C run")
    log("-" * 78)
    log("%d unit(s) computed, %d skipped as complete; frozen inputs unchanged: True"
        % (n_done, n_skipped))
    assemble(cfg, all_folds, log)
    log("total runtime %.1f s" % (time.time() - t_start))
    log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
