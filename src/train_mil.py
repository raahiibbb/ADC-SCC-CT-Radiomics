"""Phase 3 - 5-fold patient-level training of the two MIL models.

Usage:
    python src/train_mil.py                 # both models, folds 1..5
    python src/train_mil.py --folds 1       # a single fold
    python src/train_mil.py --models mean_mil

For every outer fold the fold data (inner split, variance screen, z-score
statistics, pos_weight) is prepared ONCE and handed to both models unchanged, so
model fairness is structural rather than merely asserted.

Never imports PyRadiomics or SimpleITK: features come only from the frozen bags.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import sys
import time
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mil_data import (Cohort, FoldData, load_cohort, load_outer_split,  # noqa: E402
                      load_phase3_config, ppath, prepare_fold, save_fold_artifacts,
                      sha256_file)
from mil_metrics import patient_metrics  # noqa: E402
from mil_models import build_model  # noqa: E402

MODELS = ("mean_mil", "attention_mil")


# ------------------------------------------------------------------ reproducibility
def set_seeds(cfg: dict, fold: int, model_name: str) -> int:
    """Fold- and model-specific but fully deterministic seeding."""
    scfg = cfg["seeds"]
    offset = 0 if model_name == "mean_mil" else 1
    seed = int(scfg["train_base_seed"]) + 10 * int(fold) + offset
    random.seed(int(scfg["python"]) + seed)
    np.random.seed(int(scfg["numpy"]) + seed)
    torch.manual_seed(int(scfg["torch"]) + seed)
    if bool(scfg.get("deterministic_torch", True)):
        torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    return seed


# --------------------------------------------------------------------------- epoch
def _bag_tensor(fd: FoldData, pid: str, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(fd.tensors[pid]).to(device)


def train_one_epoch(model: nn.Module, fd: FoldData, cohort: Cohort, order: Sequence[str],
                    optimizer, loss_fn, device: torch.device, bags_per_batch: int) -> float:
    model.train()
    total, n = 0.0, 0
    for start in range(0, len(order), bags_per_batch):
        batch = order[start:start + bags_per_batch]
        optimizer.zero_grad()
        logits, targets = [], []
        for pid in batch:                      # one patient -> one logit -> one loss term
            logit, _ = model(_bag_tensor(fd, pid, device))
            logits.append(logit)
            targets.append(float(cohort.bags[pid].label))
        logit_t = torch.stack(logits)
        target_t = torch.tensor(targets, dtype=torch.float32, device=device)
        loss = loss_fn(logit_t, target_t)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * len(batch)
        n += len(batch)
    return total / max(n, 1)


@torch.no_grad()
def predict(model: nn.Module, fd: FoldData, pids: Sequence[str],
            device: torch.device, want_attention: bool = False):
    model.eval()
    probs, attns = [], {}
    for pid in pids:
        logit, alpha = model(_bag_tensor(fd, pid, device))
        probs.append(float(torch.sigmoid(logit).item()))
        if want_attention and alpha is not None:
            attns[pid] = alpha.detach().cpu().numpy().astype(np.float64)
    return np.asarray(probs, dtype=float), attns


# -------------------------------------------------------------------------- 1 fold
def train_fold_model(cfg: dict, cohort: Cohort, fd: FoldData, model_name: str,
                     log) -> dict:
    tcfg = cfg["training"]
    device = torch.device(str(tcfg.get("device", "cpu")))
    seed = set_seeds(cfg, fd.fold, model_name)

    n_features = len(fd.scaler.kept_features)
    model = build_model(model_name, cfg, input_dim=n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tcfg["learning_rate"]),
                                 weight_decay=float(tcfg["weight_decay"]))
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([fd.pos_weight], dtype=torch.float32, device=device))

    y_val = cohort.labels(fd.inner_val)
    rng = random.Random(seed)
    best = {"val_roc_auc": -np.inf, "epoch": -1, "state": None}
    history: List[dict] = []
    epochs_without_improvement = 0
    t0 = time.time()

    for epoch in range(1, int(tcfg["max_epochs"]) + 1):
        order = list(fd.inner_train)
        rng.shuffle(order)
        train_loss = train_one_epoch(model, fd, cohort, order, optimizer, loss_fn,
                                     device, int(tcfg["bags_per_batch"]))
        val_prob, _ = predict(model, fd, fd.inner_val, device)
        if not np.all(np.isfinite(val_prob)):
            raise RuntimeError("non-finite validation probabilities, fold %d, %s"
                               % (fd.fold, model_name))
        if len(np.unique(y_val)) < 2:
            raise RuntimeError(
                "split-generation failure: fold %d inner validation is single-class; "
                "fix the stratified patient-level split, do not change the metric" % fd.fold)
        vm = patient_metrics(y_val, val_prob, float(cfg["evaluation"]["threshold"]))
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_roc_auc": vm["roc_auc"], "val_pr_auc": vm["pr_auc"],
                        "val_balanced_accuracy": vm["balanced_accuracy"]})

        if vm["roc_auc"] > best["val_roc_auc"] + 1e-12:
            best = {"val_roc_auc": vm["roc_auc"], "epoch": epoch,
                    "state": copy.deepcopy(model.state_dict())}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(tcfg["early_stopping_patience"]):
                log("    early stop at epoch %d (best epoch %d, val ROC-AUC %.4f)"
                    % (epoch, best["epoch"], best["val_roc_auc"]))
                break

    train_seconds = time.time() - t0
    if best["state"] is None:
        raise RuntimeError("no checkpoint was selected for fold %d, %s" % (fd.fold, model_name))
    model.load_state_dict(best["state"])

    # ---- final evaluation of the untouched outer-test patients
    want_att = model_name == "attention_mil"
    test_prob, test_att = predict(model, fd, fd.outer_test, device, want_attention=want_att)
    y_test = cohort.labels(fd.outer_test)
    if not np.all(np.isfinite(test_prob)):
        raise RuntimeError("non-finite test probabilities, fold %d, %s" % (fd.fold, model_name))
    fold_metrics = patient_metrics(y_test, test_prob, float(cfg["evaluation"]["threshold"]))

    # ---- persist
    ck_dir = ppath(cfg, "%s/%s" % (cfg["outputs"]["models_dir"], model_name))
    os.makedirs(ck_dir, exist_ok=True)
    ckpt_path = os.path.join(ck_dir, "fold_%d.pt" % fd.fold)
    torch.save({
        "model_state_dict": best["state"],
        "model_name": model_name,
        "fold": fd.fold,
        "input_dim": n_features,
        "kept_features": fd.scaler.kept_features,
        "scaler_mean": fd.scaler.mean,
        "scaler_std": fd.scaler.std,
        "best_epoch": best["epoch"],
        "best_val_roc_auc": best["val_roc_auc"],
        "pos_weight": fd.pos_weight,
        "seed": seed,
        "fold_fingerprint": fd.fingerprint(),
        "config": {"model": cfg["model"], "training": cfg["training"], "seeds": cfg["seeds"]},
    }, ckpt_path)
    with open(os.path.join(ck_dir, "fold_%d_history.json" % fd.fold), "w",
              encoding="utf-8") as fh:
        json.dump({"fold": fd.fold, "model": model_name, "seed": seed,
                   "epochs_run": len(history), "best_epoch": best["epoch"],
                   "best_val_roc_auc": best["val_roc_auc"],
                   "train_seconds": train_seconds, "history": history}, fh, indent=2)

    if want_att:
        save_attention(cfg, cohort, fd, test_att, test_prob)

    return {
        "model": model_name,
        "fold": fd.fold,
        "seed": seed,
        "epochs_run": len(history),
        "best_epoch": best["epoch"],
        "best_val_roc_auc": best["val_roc_auc"],
        "train_seconds": train_seconds,
        "pos_weight": fd.pos_weight,
        "n_features": n_features,
        "fold_fingerprint": fd.fingerprint(),
        "checkpoint": os.path.relpath(ckpt_path, cfg["_project_root"]).replace("\\", "/"),
        "test_probabilities": {p: float(v) for p, v in zip(fd.outer_test, test_prob)},
        "fold_metrics": fold_metrics,
    }


def save_attention(cfg: dict, cohort: Cohort, fd: FoldData,
                   attns: Dict[str, np.ndarray], test_prob: np.ndarray) -> None:
    """Raw attention for Phase 4.  No interpretation is produced here."""
    out_dir = ppath(cfg, cfg["outputs"]["attention_dir"])
    os.makedirs(out_dir, exist_ok=True)
    prob_of = {p: float(v) for p, v in zip(fd.outer_test, test_prob)}
    thr = float(cfg["evaluation"]["threshold"])
    for pid in fd.outer_test:
        alpha = attns[pid]
        bag = cohort.bags[pid]
        if alpha.shape[0] != bag.n_patches:
            raise RuntimeError("attention length %d != %d patches for %s"
                               % (alpha.shape[0], bag.n_patches, pid))
        if not np.all(np.isfinite(alpha)):
            raise RuntimeError("non-finite attention for %s" % pid)
        np.savez_compressed(
            os.path.join(out_dir, "%s.npz" % pid),
            patient_id=np.str_(pid),
            histology=np.str_(bag.histology),
            true_label=np.int8(bag.label),
            attention=alpha.astype(np.float64),
            patch_index=np.arange(bag.n_patches, dtype=np.int32),
            coords_index=bag.coords_index,
            coords_world=bag.coords_world,
            roi_voxels=bag.roi_voxels,
            outer_fold=np.int32(fd.fold),
            predicted_probability_adc=np.float64(prob_of[pid]),
            predicted_label=np.int8(1 if prob_of[pid] >= thr else 0),
            model=np.str_("attention_mil"),
            roi_name=np.str_(cfg["phase3"]["roi"]),
            extraction_signature=np.str_(bag.config_signature),
            attention_sum=np.float64(float(alpha.sum())),
            npz_version=np.int32(1),
        )


# ---------------------------------------------------------------------------- main
def input_manifest(cfg: dict) -> dict:
    feat_dir = ppath(cfg, cfg["phase3"]["features_dir"])
    bags = {f: sha256_file(os.path.join(feat_dir, f))
            for f in sorted(os.listdir(feat_dir)) if f.endswith(".npz")}
    return {
        "features_dir": cfg["phase3"]["features_dir"],
        "n_bags": len(bags),
        "bag_sha256": bags,
        "splits_file_sha256": sha256_file(ppath(cfg, cfg["phase3"]["splits_file"])),
        "feature_names_sha256": sha256_file(ppath(cfg, cfg["phase3"]["feature_names_file"])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3 MIL training")
    ap.add_argument("--config", default=None)
    ap.add_argument("--folds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    ap.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    args = ap.parse_args()

    cfg = load_phase3_config(args.config) if args.config else load_phase3_config()
    log_path = ppath(cfg, "logs/phase3_training.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_fh = open(log_path, "a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    t_start = time.time()
    log("=" * 78)
    log("PHASE 3 - MIL training  %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    log("models=%s folds=%s" % (args.models, args.folds))

    manifest_before = input_manifest(cfg)
    cohort = load_cohort(cfg)
    split_df = load_outer_split(cfg)
    p3 = cfg["phase3"]
    labels = cohort.labels(cohort.patient_ids)
    assert len(cohort.bags) == int(p3["expected_patients"]), "unexpected bag count"
    assert int((labels == 1).sum()) == int(p3["expected_adc"])
    assert int((labels == 0).sum()) == int(p3["expected_scc"])
    log("cohort: %d bags, %d ADC / %d SCC, %d features, %d patches"
        % (len(cohort.bags), int((labels == 1).sum()), int((labels == 0).sum()),
           len(cohort.feature_names), sum(b.n_patches for b in cohort.bags.values())))

    runs: List[dict] = []
    fold_info: List[dict] = []
    for fold in args.folds:
        fd = prepare_fold(cfg, cohort, split_df, fold)
        save_fold_artifacts(cfg, cohort, fd)
        y_it = cohort.labels(fd.inner_train)
        y_iv = cohort.labels(fd.inner_val)
        log("-" * 78)
        log("fold %d: inner_train %d (%d ADC/%d SCC), inner_val %d (%d ADC/%d SCC), "
            "outer_test %d | pos_weight %.4f | features kept %d/%d | fp %s"
            % (fold, len(fd.inner_train), int((y_it == 1).sum()), int((y_it == 0).sum()),
               len(fd.inner_val), int((y_iv == 1).sum()), int((y_iv == 0).sum()),
               len(fd.outer_test), fd.pos_weight, len(fd.scaler.kept_features),
               len(cohort.feature_names), fd.fingerprint()))
        fold_info.append({
            "fold": fold, "inner_seed": fd.inner_seed, "fold_fingerprint": fd.fingerprint(),
            "scaler_fingerprint": fd.scaler.fingerprint(), "pos_weight": fd.pos_weight,
            "n_inner_train": len(fd.inner_train), "n_inner_val": len(fd.inner_val),
            "n_outer_test": len(fd.outer_test),
            "n_features_kept": len(fd.scaler.kept_features),
            "dropped_features": fd.scaler.dropped_features,
        })
        for model_name in args.models:
            log("  training %s ..." % model_name)
            res = train_fold_model(cfg, cohort, fd, model_name, log)
            m = res["fold_metrics"]
            log("    %s fold %d: test ROC-AUC %.4f  PR-AUC %.4f  balacc %.4f  MCC %.4f "
                "(%.1f s, %d epochs)"
                % (model_name, fold, m["roc_auc"], m["pr_auc"], m["balanced_accuracy"],
                   m["mcc"], res["train_seconds"], res["epochs_run"]))
            runs.append(res)

    manifest_after = input_manifest(cfg)
    frozen_ok = manifest_before == manifest_after
    total_seconds = time.time() - t_start

    run_meta = {
        "phase": 3,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models": list(args.models),
        "folds": list(args.folds),
        "total_runtime_seconds": total_seconds,
        "training_seconds_by_model": {
            m: sum(r["train_seconds"] for r in runs if r["model"] == m) for m in args.models},
        "frozen_inputs_unchanged": bool(frozen_ok),
        "input_manifest": manifest_before,
        "fold_info": fold_info,
        "runs": [{k: v for k, v in r.items() if k != "test_probabilities"} for r in runs],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(cfg["training"]["device"]),
            "deterministic_algorithms": bool(cfg["seeds"].get("deterministic_torch", True)),
            "torch_threads": torch.get_num_threads(),
        },
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "config_path": cfg["_config_path"],
        "radiomics_imported_during_training": ("radiomics" in sys.modules
                                               or "SimpleITK" in sys.modules),
    }
    out = ppath(cfg, "reports/phase3_training_run.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(run_meta, fh, indent=2, default=str)

    # per-patient out-of-fold probabilities, one row per (patient, model)
    rows = []
    for r in runs:
        for pid, prob in r["test_probabilities"].items():
            bag = cohort.bags[pid]
            rows.append({"PatientID": pid, "histology": bag.histology,
                         "true_label": int(bag.label),
                         "predicted_probability_ADC": float(prob),
                         "predicted_label": int(prob >= float(cfg["evaluation"]["threshold"])),
                         "outer_fold": int(r["fold"]), "model": r["model"],
                         "n_patches": bag.n_patches})
    import pandas as pd
    pred_dir = ppath(cfg, cfg["outputs"]["predictions_dir"])
    os.makedirs(pred_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    for model_name in args.models:
        sub = df[df["model"] == model_name].sort_values("PatientID")
        sub.to_csv(os.path.join(pred_dir, "%s_oof.csv" % model_name), index=False)
        log("wrote predictions/%s_oof.csv  (%d rows)" % (model_name, len(sub)))

    log("-" * 78)
    log("frozen inputs unchanged: %s" % frozen_ok)
    log("radiomics/SimpleITK imported during training: %s"
        % run_meta["radiomics_imported_during_training"])
    log("total runtime %.1f s" % total_seconds)
    log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
