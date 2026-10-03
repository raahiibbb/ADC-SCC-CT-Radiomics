"""Phase 3 - frozen-bag loading, fold assembly, inner splits and fold-local scaling.

Nothing in this module reads DICOM/NIfTI or touches PyRadiomics: the Phase-2 bags
are the only source of features.  Every statistic used to transform features is
derived from INNER TRAINING patients only.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_PHASE3_CONFIG = os.path.join(PROJECT_ROOT, "config", "phase3.yaml")


def load_phase3_config(path: str = DEFAULT_PHASE3_CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = os.path.abspath(path)
    cfg["_project_root"] = PROJECT_ROOT
    return cfg


def ppath(cfg: dict, rel: str) -> str:
    return os.path.join(cfg["_project_root"], *rel.split("/"))


def model_tag(cfg: dict, model_name: str) -> str:
    """Namespaced artefact name for a model.

    Phase 3   -> "mean_mil"                  (no prefix, no suffix)
    Phase 3B  -> "mean_mil_3b"               (suffix only)
    Phase 5   -> "gtv_rim_mean_mil"          (prefix only)
    Phase 6B  -> "phase6b_cnn_mean"          (explicit outputs.model_names map)

    All three keys default to absent/"", so every earlier phase keeps its exact
    file names.
    """
    out = cfg.get("outputs", {})
    explicit = out.get("model_names") or {}
    if model_name in explicit:
        return str(explicit[model_name])
    return "%s%s%s" % (str(out.get("model_prefix", "")), model_name,
                       str(out.get("model_suffix", "")))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1048576)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- bags
@dataclass
class Bag:
    patient_id: str
    histology: str
    label: int
    features: np.ndarray          # [N_patches, F] float32
    coords_index: np.ndarray      # [N_patches, 3] int32
    coords_world: np.ndarray      # [N_patches, 3] float64
    roi_voxels: Optional[np.ndarray]   # [N_patches] int32, or None (CNN embeddings)
    config_signature: str
    provenance: Dict[str, str] = field(default_factory=dict)

    @property
    def n_patches(self) -> int:
        return int(self.features.shape[0])


@dataclass
class Cohort:
    bags: Dict[str, Bag]
    feature_names: List[str]
    patient_ids: List[str] = field(default_factory=list)

    def labels(self, pids: Sequence[str]) -> np.ndarray:
        return np.array([self.bags[p].label for p in pids], dtype=np.int64)


# Which kind of frozen per-patient instance file `features_dir` holds.
#   radiomics_bag  -> patch_features/<roi>/<PID>.npz   (Phases 3, 3B, 5, 5C)
#   cnn_embedding  -> cnn_embeddings/<enc>/<PID>.npz   (Phase 6B)
# Both are read-only inputs produced by an earlier, frozen phase.  Nothing here
# extracts radiomics, opens a CT, or runs a CNN: the file on disk is the only
# source of instance features in either case.
FEATURES_FORMATS = ("radiomics_bag", "cnn_embedding")


def features_format(cfg: dict) -> str:
    fmt = str(cfg["phase3"].get("features_format", "radiomics_bag"))
    if fmt not in FEATURES_FORMATS:
        raise ValueError("unknown phase3.features_format %r" % fmt)
    return fmt


def load_cohort(cfg: dict) -> Cohort:
    """Load the frozen per-patient bags.  Read-only: files are opened, never written."""
    feat_dir = ppath(cfg, cfg["phase3"]["features_dir"])
    names_path = ppath(cfg, cfg["phase3"]["feature_names_file"])
    with open(names_path, "r", encoding="utf-8") as fh:
        feature_names = [ln.strip() for ln in fh if ln.strip()]
    fmt = features_format(cfg)

    files = sorted(f for f in os.listdir(feat_dir) if f.endswith(".npz"))
    bags: Dict[str, Bag] = {}
    for fname in files:
        with np.load(os.path.join(feat_dir, fname), allow_pickle=True) as d:
            pid = str(d["patient_id"])
            if fmt == "radiomics_bag":
                names = [str(x) for x in d["feature_names"]]
                if names != feature_names:
                    raise ValueError("feature name/order mismatch in %s" % fname)
                x = np.asarray(d["features"], dtype=np.float32)
                roi_voxels = np.asarray(d["roi_voxels"], dtype=np.int32)
                signature = str(d["config_signature"])
                prov = {}
            else:                                        # cnn_embedding
                x = np.asarray(d["embeddings"], dtype=np.float32)
                if x.shape[1] != len(feature_names):
                    raise ValueError("embedding width %d != %d declared names in %s"
                                     % (x.shape[1], len(feature_names), fname))
                if int(d["embedding_dim"]) != x.shape[1]:
                    raise ValueError("embedding_dim disagrees with the array in %s" % fname)
                if not bool(d["encoder_frozen"]):
                    raise RuntimeError("%s was produced by a non-frozen encoder" % fname)
                roi_voxels = None
                signature = str(d["source_extraction_signature"])
                prov = {
                    "encoder_architecture": str(d["encoder_architecture"]),
                    "encoder_weights_enum": str(d["encoder_weights_enum"]),
                    "encoder_state_fingerprint": str(d["encoder_state_fingerprint"]),
                    "encoder_head": str(d["encoder_head"]),
                    "transform_signature": str(d["transform_signature"]),
                    "image_crop_signature": str(d["image_crop_signature"]),
                    "source_coords_sha256": str(d["source_coords_sha256"]),
                    "source_extraction_signature": str(d["source_extraction_signature"]),
                }
            if not np.all(np.isfinite(x)):
                raise ValueError("non-finite instance features in %s" % fname)
            bags[pid] = Bag(
                patient_id=pid,
                histology=str(d["histology"]),
                label=int(d["label"]),
                features=x,
                coords_index=np.asarray(d["coords_index"], dtype=np.int32),
                coords_world=np.asarray(d["coords_world"], dtype=np.float64),
                roi_voxels=roi_voxels,
                config_signature=signature,
                provenance=prov,
            )
    return Cohort(bags=bags, feature_names=feature_names, patient_ids=sorted(bags))


# ------------------------------------------------------------------------- splits
def load_outer_split(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(ppath(cfg, cfg["phase3"]["splits_file"]))
    required = {"fold", "PatientID", "Histology", "label", "split"}
    if not required.issubset(df.columns):
        raise ValueError("unexpected split file columns: %s" % list(df.columns))
    return df


def outer_fold_patients(split_df: pd.DataFrame, fold: int) -> Tuple[List[str], List[str]]:
    sub = split_df[split_df["fold"] == fold]
    train = sorted(sub.loc[sub["split"] == "train", "PatientID"].tolist())
    test = sorted(sub.loc[sub["split"] == "test", "PatientID"].tolist())
    return train, test


def make_inner_split(cfg: dict, fold: int, train_pids: Sequence[str],
                     labels: np.ndarray) -> Tuple[List[str], List[str], int]:
    """Reproducible stratified inner train/validation split of the OUTER TRAIN patients.

    Deterministic: seed = base_seed + fold.  Both classes are guaranteed to be
    present on both sides; a violation raises rather than being worked around.
    """
    icfg = cfg["inner_split"]
    seed = int(icfg["base_seed"]) + int(fold)
    frac = float(icfg["val_fraction"])
    rng = np.random.RandomState(seed)

    train_pids = list(train_pids)
    label_of = {p: int(y) for p, y in zip(train_pids, labels)}
    val: List[str] = []
    for cls in (0, 1):
        members = sorted(p for p in train_pids if label_of[p] == cls)
        n_val = int(round(frac * len(members)))
        n_val = max(n_val, int(icfg["min_per_class_in_val"]))
        n_val = min(n_val, len(members) - int(icfg["min_per_class_in_train"]))
        picked = rng.choice(np.array(members), size=n_val, replace=False)
        val.extend(sorted(picked.tolist()))
    val_set = set(val)
    inner_train = sorted(p for p in train_pids if p not in val_set)
    inner_val = sorted(val)

    y_tr = np.array([label_of[p] for p in inner_train])
    y_va = np.array([label_of[p] for p in inner_val])
    for name, y in (("inner_train", y_tr), ("inner_val", y_va)):
        if len(np.unique(y)) < 2:
            raise RuntimeError(
                "split-generation failure: %s of fold %d contains a single class"
                % (name, fold))
    return inner_train, inner_val, seed


# -------------------------------------------------------------------- preprocessing
@dataclass
class FoldScaler:
    keep_idx: np.ndarray
    kept_features: List[str]
    dropped_features: List[str]
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x[:, self.keep_idx] - self.mean) / self.std).astype(np.float32)

    def fingerprint(self) -> str:
        blob = json.dumps({
            "kept": self.kept_features,
            "dropped": self.dropped_features,
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
        }, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


def fit_fold_scaler(cfg: dict, cohort: Cohort, inner_train_pids: Sequence[str]) -> FoldScaler:
    """Variance screen + z-score statistics from INNER TRAINING PATCHES ONLY."""
    x = np.concatenate([cohort.bags[p].features for p in inner_train_pids],
                       axis=0).astype(np.float64)
    vcfg = cfg["preprocessing"]["variance"]
    eps = float(vcfg["eps"])
    std = x.std(axis=0, ddof=0)
    mean = x.mean(axis=0)
    rel = std / (np.abs(mean) + eps)
    drop = ((std <= float(vcfg["absolute_std_threshold"]))
            | (rel <= float(vcfg["relative_std_threshold"])))
    keep_idx = np.where(~drop)[0]
    if keep_idx.size == 0:
        raise RuntimeError("all features were screened out")
    names = cohort.feature_names
    return FoldScaler(
        keep_idx=keep_idx,
        kept_features=[names[i] for i in keep_idx],
        dropped_features=[names[i] for i in np.where(drop)[0]],
        mean=mean[keep_idx],
        std=std[keep_idx],
    )


def pos_weight_from_patients(labels: np.ndarray) -> float:
    """N_SCC_patients / N_ADC_patients, from PATIENT counts (never patch counts)."""
    n_adc = int((labels == 1).sum())
    n_scc = int((labels == 0).sum())
    if n_adc == 0:
        raise RuntimeError("no ADC patients in the training set")
    return float(n_scc) / float(n_adc)


# ------------------------------------------------------------------- fold container
@dataclass
class FoldData:
    fold: int
    inner_train: List[str]
    inner_val: List[str]
    outer_test: List[str]
    inner_seed: int
    scaler: FoldScaler
    pos_weight: float
    tensors: Dict[str, np.ndarray]     # pid -> scaled [N, F] float32

    def fingerprint(self) -> str:
        blob = json.dumps({
            "fold": self.fold,
            "inner_train": self.inner_train,
            "inner_val": self.inner_val,
            "outer_test": self.outer_test,
            "inner_seed": self.inner_seed,
            "scaler": self.scaler.fingerprint(),
            "pos_weight": round(self.pos_weight, 12),
        }, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


def prepare_fold(cfg: dict, cohort: Cohort, split_df: pd.DataFrame, fold: int) -> FoldData:
    train_pids, test_pids = outer_fold_patients(split_df, fold)
    y_train = cohort.labels(train_pids)
    inner_train, inner_val, seed = make_inner_split(cfg, fold, train_pids, y_train)

    overlap = ((set(inner_train) & set(inner_val))
               | (set(inner_train) & set(test_pids))
               | (set(inner_val) & set(test_pids)))
    if overlap:
        raise RuntimeError("patient overlap in fold %d: %s" % (fold, sorted(overlap)))

    scaler = fit_fold_scaler(cfg, cohort, inner_train)
    pw = pos_weight_from_patients(cohort.labels(inner_train))
    tensors = {p: scaler.transform(cohort.bags[p].features)
               for p in list(inner_train) + list(inner_val) + list(test_pids)}
    return FoldData(fold=fold, inner_train=inner_train, inner_val=inner_val,
                    outer_test=test_pids, inner_seed=seed, scaler=scaler,
                    pos_weight=pw, tensors=tensors)


def save_fold_artifacts(cfg: dict, cohort: Cohort, fd: FoldData) -> None:
    inner_dir = ppath(cfg, cfg["outputs"]["inner_splits_dir"])
    prep_dir = ppath(cfg, cfg["outputs"]["preprocessing_dir"])
    os.makedirs(inner_dir, exist_ok=True)
    os.makedirs(prep_dir, exist_ok=True)

    rows = []
    for pid in fd.inner_train:
        rows.append((pid, cohort.bags[pid].histology, cohort.bags[pid].label, "inner_train"))
    for pid in fd.inner_val:
        rows.append((pid, cohort.bags[pid].histology, cohort.bags[pid].label, "inner_val"))
    for pid in fd.outer_test:
        rows.append((pid, cohort.bags[pid].histology, cohort.bags[pid].label, "outer_test"))
    df = pd.DataFrame(rows, columns=["PatientID", "Histology", "label", "assignment"])
    df.insert(0, "fold", fd.fold)
    df.to_csv(os.path.join(inner_dir, "fold_%d.csv" % fd.fold), index=False)

    np.savez_compressed(
        os.path.join(prep_dir, "fold_%d_scaler.npz" % fd.fold),
        keep_idx=fd.scaler.keep_idx, mean=fd.scaler.mean, std=fd.scaler.std,
        kept_features=np.array(fd.scaler.kept_features, dtype=object),
        dropped_features=np.array(fd.scaler.dropped_features, dtype=object),
        fold=np.int32(fd.fold), pos_weight=np.float64(fd.pos_weight),
    )
    meta = {
        "fold": fd.fold,
        "inner_seed": fd.inner_seed,
        "n_inner_train_patients": len(fd.inner_train),
        "n_inner_val_patients": len(fd.inner_val),
        "n_outer_test_patients": len(fd.outer_test),
        "n_inner_train_adc": int((cohort.labels(fd.inner_train) == 1).sum()),
        "n_inner_train_scc": int((cohort.labels(fd.inner_train) == 0).sum()),
        "n_inner_val_adc": int((cohort.labels(fd.inner_val) == 1).sum()),
        "n_inner_val_scc": int((cohort.labels(fd.inner_val) == 0).sum()),
        "n_inner_train_patches": int(sum(cohort.bags[p].n_patches for p in fd.inner_train)),
        "n_inner_val_patches": int(sum(cohort.bags[p].n_patches for p in fd.inner_val)),
        "n_outer_test_patches": int(sum(cohort.bags[p].n_patches for p in fd.outer_test)),
        "pos_weight": fd.pos_weight,
        "pos_weight_definition": "n_scc_inner_train_patients / n_adc_inner_train_patients",
        "n_features_kept": len(fd.scaler.kept_features),
        "n_features_dropped": len(fd.scaler.dropped_features),
        "dropped_features": fd.scaler.dropped_features,
        "kept_features": fd.scaler.kept_features,
        "scaler_fingerprint": fd.scaler.fingerprint(),
        "fold_fingerprint": fd.fingerprint(),
        "variance_rule": cfg["preprocessing"]["variance"],
        "scaling_source": "inner_train_patches_only",
    }
    with open(os.path.join(prep_dir, "fold_%d_scaler.json" % fd.fold), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
