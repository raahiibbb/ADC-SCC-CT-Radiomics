"""Phase 7A - the frozen, training-only modelling primitives.

Everything in this module takes the training rows EXPLICITLY.  Nothing here can
see a held-out patient unless the caller hands it one, and the caller
(src/phase7a_cv.py) never does.

Contents
--------
TrainingOnlyPreprocessor   median impute -> near-zero-variance screen -> z-score,
                           all three fitted on the given training rows alone
ClinicalEncoder            age / sex / stage design matrix; medians, modes and
                           one-hot categories all learned from training rows only
CappedElasticNet           elastic-net logistic + the deterministic <= 8 feature
                           cap and the same-hyperparameter refit
fit_ridge / fit_stacker    the A0 control and the A3 two-input ridge stacker
platt_fit / platt_apply    Platt calibration from inner-OOF predictions only
select_threshold           balanced-accuracy threshold from calibrated inner-OOF
                           predictions only
mrmr_select                the SECONDARY Ding & Peng selector (reused verbatim
                           from src/feature_selection.py)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from feature_selection import mrmr_rank


# ---------------------------------------------------------------------------
# 1.  Preprocessing - fitted on the training rows, applied unchanged elsewhere
# ---------------------------------------------------------------------------

@dataclass
class TrainingOnlyPreprocessor:
    """drop-non-features -> median impute -> variance screen -> z-score.

    `fit` may only ever be handed the rows of the current training partition.
    """
    abs_tol: float = 1e-8
    rel_tol: float = 1e-8
    all_nan_fill: float = 0.0

    medians: Optional[np.ndarray] = None
    keep: Optional[np.ndarray] = None          # boolean mask over the input columns
    mean: Optional[np.ndarray] = None
    scale: Optional[np.ndarray] = None
    n_fit_rows: int = 0
    n_input: int = 0

    def fit(self, x: np.ndarray) -> "TrainingOnlyPreprocessor":
        x = np.asarray(x, dtype=np.float64)
        x = np.where(np.isfinite(x), x, np.nan)          # +/-inf -> NaN, then imputed
        self.n_fit_rows, self.n_input = x.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN column
            med = np.nanmedian(x, axis=0)
        med = np.where(np.isfinite(med), med, float(self.all_nan_fill))
        self.medians = med
        xi = np.where(np.isnan(x), med, x)
        mean = xi.mean(axis=0)
        std = xi.std(axis=0, ddof=0)
        drop = (std <= self.abs_tol) | (std / (np.abs(mean) + 1e-12) <= self.rel_tol)
        self.keep = ~drop
        self.mean = mean[self.keep]
        scale = std[self.keep]
        scale[scale == 0.0] = 1.0
        self.scale = scale
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x = np.where(np.isfinite(x), x, np.nan)
        xi = np.where(np.isnan(x), self.medians, x)
        return (xi[:, self.keep] - self.mean) / self.scale

    @property
    def kept_indices(self) -> np.ndarray:
        return np.flatnonzero(self.keep)

    def state(self) -> dict:
        return {"n_fit_rows": int(self.n_fit_rows), "n_input": int(self.n_input),
                "n_kept": int(self.keep.sum()), "n_dropped": int((~self.keep).sum()),
                "abs_tol": self.abs_tol, "rel_tol": self.rel_tol}

    def arrays(self) -> dict:
        return {"medians": self.medians, "keep": self.keep.astype(np.uint8),
                "mean": self.mean, "scale": self.scale}


# ---------------------------------------------------------------------------
# 2.  A0 clinical design matrix
# ---------------------------------------------------------------------------

@dataclass
class ClinicalEncoder:
    """age (numeric) + sex and stage (one-hot).

    Medians, modes and the one-hot category lists are all learned from the
    TRAINING rows handed to `fit`.  A category unseen in training encodes as
    all-zero; a missing value additionally raises its own indicator column, so
    "missing" is never silently mapped onto the training mode's dummy.
    """
    categorical: Tuple[str, ...] = ("sex", "stage")
    numeric: Tuple[str, ...] = ("age",)
    categories: Dict[str, List[str]] = field(default_factory=dict)
    modes: Dict[str, str] = field(default_factory=dict)
    medians: Dict[str, float] = field(default_factory=dict)
    columns: List[str] = field(default_factory=list)

    @staticmethod
    def _norm(v):
        if v is None:
            return None
        s = str(v).strip()
        if s == "" or s.lower() in ("nan", "none", "na", "unknown"):
            return None
        return s

    def fit(self, rows: List[dict]) -> "ClinicalEncoder":
        for c in self.numeric:
            vals = [float(r[c]) for r in rows
                    if r.get(c) is not None and np.isfinite(_safe_float(r[c]))]
            self.medians[c] = float(np.median(vals)) if vals else 0.0
        for c in self.categorical:
            vals = [self._norm(r.get(c)) for r in rows]
            present = [v for v in vals if v is not None]
            self.categories[c] = sorted(set(present))
            if present:
                counts = {v: present.count(v) for v in set(present)}
                best = max(counts.values())
                self.modes[c] = sorted(v for v, n in counts.items() if n == best)[0]
            else:
                self.modes[c] = ""
        cols = list(self.numeric) + ["%s_missing" % c for c in self.numeric]
        for c in self.categorical:
            cols += ["%s=%s" % (c, v) for v in self.categories[c]]
            cols += ["%s_missing" % c]
        self.columns = cols
        return self

    def transform(self, rows: List[dict]) -> np.ndarray:
        out = np.zeros((len(rows), len(self.columns)), dtype=np.float64)
        idx = {c: i for i, c in enumerate(self.columns)}
        for i, r in enumerate(rows):
            for c in self.numeric:
                v = _safe_float(r.get(c))
                if not np.isfinite(v):
                    out[i, idx[c]] = self.medians[c]
                    out[i, idx["%s_missing" % c]] = 1.0
                else:
                    out[i, idx[c]] = v
            for c in self.categorical:
                v = self._norm(r.get(c))
                if v is None:
                    out[i, idx["%s_missing" % c]] = 1.0
                elif v in self.categories[c]:
                    out[i, idx["%s=%s" % (c, v)]] = 1.0
                # a category unseen in training stays all-zero, by design
        return out


def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# 3.  Estimators
# ---------------------------------------------------------------------------

@dataclass
class CappedElasticNet:
    """Elastic-net logistic regression + the frozen <= 8 feature cap.

    Both stages are fitted on the SAME training rows.  The ranking that chooses
    the retained features is computed from the first-stage coefficients of those
    rows alone - no held-out patient participates.
    """
    C: float
    l1_ratio: float
    max_features: int
    est_cfg: dict

    selected: List[int] = field(default_factory=list)     # indices into the SCALED matrix
    n_nonzero_stage1: int = 0
    coefficients: Optional[np.ndarray] = None             # stage-2 coefs, len(selected)
    intercept: float = 0.0
    model: Optional[LogisticRegression] = None
    constant_prior: Optional[float] = None
    converged_stage1: bool = True
    converged_stage2: bool = True
    n_iter_stage1: int = 0
    n_iter_stage2: int = 0

    def _make(self):
        return LogisticRegression(
            penalty=self.est_cfg["penalty"], solver=self.est_cfg["solver"],
            C=float(self.C), l1_ratio=float(self.l1_ratio),
            class_weight=self.est_cfg["class_weight"],
            max_iter=int(self.est_cfg["max_iter"]), tol=float(self.est_cfg["tol"]),
            random_state=int(self.est_cfg["random_state"]))

    def fit(self, xs: np.ndarray, y: np.ndarray) -> "CappedElasticNet":
        with warnings.catch_warnings(record=True) as w1:
            warnings.simplefilter("always", ConvergenceWarning)
            stage1 = self._make().fit(xs, y)
        self.converged_stage1 = not any(issubclass(x.category, ConvergenceWarning) for x in w1)
        self.n_iter_stage1 = int(np.max(stage1.n_iter_))
        coef = stage1.coef_[0]
        nz = np.flatnonzero(coef != 0.0)
        self.n_nonzero_stage1 = int(nz.size)
        order = sorted(nz.tolist(), key=lambda j: (-abs(float(coef[j])), int(j)))
        self.selected = sorted(int(j) for j in order[: int(self.max_features)])

        if not self.selected:
            # recorded, never substituted: the fold's model is the prior
            self.constant_prior = float(np.mean(y))
            self.coefficients = np.zeros(0)
            self.intercept = 0.0
            self.model = None
            return self

        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter("always", ConvergenceWarning)
            stage2 = self._make().fit(xs[:, self.selected], y)
        self.converged_stage2 = not any(issubclass(x.category, ConvergenceWarning) for x in w2)
        self.n_iter_stage2 = int(np.max(stage2.n_iter_))
        self.model = stage2
        self.coefficients = stage2.coef_[0].copy()
        self.intercept = float(stage2.intercept_[0])
        return self

    def predict_proba(self, xs: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.full(xs.shape[0], float(self.constant_prior))
        return self.model.predict_proba(xs[:, self.selected])[:, 1]


def fit_logistic(xs: np.ndarray, y: np.ndarray, cfg: dict, C: Optional[float] = None):
    est = LogisticRegression(
        penalty=cfg["penalty"], solver=cfg["solver"],
        C=float(cfg["C"] if C is None else C),
        class_weight=cfg.get("class_weight"),
        max_iter=int(cfg["max_iter"]), tol=float(cfg.get("tol", 1e-4)),
        random_state=int(cfg["random_state"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        est.fit(xs, y)
    return est


# ---------------------------------------------------------------------------
# 4.  Calibration and threshold - inner-OOF only
# ---------------------------------------------------------------------------

@dataclass
class PlattCalibrator:
    a: float = 1.0
    b: float = 0.0
    clip: float = 1e-6
    n_fit: int = 0

    def fit(self, p: np.ndarray, y: np.ndarray, cfg: dict) -> "PlattCalibrator":
        self.clip = float(cfg["clip"])
        z = _logit(np.asarray(p, dtype=np.float64), self.clip).reshape(-1, 1)
        est = LogisticRegression(solver=cfg["solver"], C=float(cfg["C"]),
                                 class_weight=cfg.get("class_weight"),
                                 max_iter=int(cfg["max_iter"]),
                                 random_state=int(cfg["random_state"]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            est.fit(z, y)
        self.a = float(est.coef_[0][0])
        self.b = float(est.intercept_[0])
        self.n_fit = int(len(y))
        return self

    def apply(self, p: np.ndarray) -> np.ndarray:
        z = _logit(np.asarray(p, dtype=np.float64), self.clip)
        return 1.0 / (1.0 + np.exp(-(self.a * z + self.b)))


def _logit(p, clip):
    p = np.clip(np.asarray(p, dtype=np.float64), clip, 1.0 - clip)
    return np.log(p / (1.0 - p))


def select_threshold(p_cal: np.ndarray, y: np.ndarray, tie_break: str = "closest_to_0.5"):
    """Balanced-accuracy-maximising threshold over the calibrated inner-OOF
    probabilities.  Deterministic: candidates are the sorted unique values, the
    predicted label is `p >= t`, ties go to the candidate closest to 0.5."""
    p = np.asarray(p_cal, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    cands = np.unique(p)
    best_t, best_v = 0.5, -np.inf
    scores = []
    for t in cands:
        yhat = (p >= t).astype(int)
        tp = int(((yhat == 1) & (y == 1)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
        tn = int(((yhat == 0) & (y == 0)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        v = 0.5 * (sens + spec)
        scores.append((float(t), float(v)))
        if v > best_v + 1e-15 or (abs(v - best_v) <= 1e-15 and abs(t - 0.5) < abs(best_t - 0.5)):
            best_t, best_v = float(t), float(v)
    return best_t, best_v, scores


# ---------------------------------------------------------------------------
# 5.  SECONDARY - mRMR
# ---------------------------------------------------------------------------

def mrmr_select(xs: np.ndarray, y: np.ndarray, k: int, scfg: dict) -> List[int]:
    """Ding & Peng FCD ranking over the PATIENT rows of `xs`; first k indices."""
    rank = mrmr_rank(xs, y, n_select=int(k),
                     zero_variance_relevance=float(scfg["zero_variance_relevance"]),
                     zero_variance_correlation=float(scfg["zero_variance_correlation"]))
    return [int(i) for i in rank.select(int(k))]
