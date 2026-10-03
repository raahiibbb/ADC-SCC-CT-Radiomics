"""Phase 8B - the low-capacity, weight-aware modelling primitives.

Everything here takes the training rows and the training weights EXPLICITLY.
Nothing can see a held-out patient unless the caller hands it one, and no
caller ever does.

What is REUSED verbatim from the frozen Phase-7A module, by import rather than
by re-implementation, so identity is a fact about the import graph:

    TrainingOnlyPreprocessor   median impute -> near-zero-variance screen -> z-score
    PlattCalibrator            Platt calibration from inner-OOF predictions only
    select_threshold           balanced-accuracy threshold from calibrated inner-OOF

and from the frozen Phase-5C selector module:

    mrmr_rank                  Ding & Peng (2003) continuous FCD ranking

What Phase 8B ADDS, and only this:

    weighted fits              sample_weight support for the four-stratum
                               source-balanced weights (protocol section 7)
    the two-family candidate set  elastic-net x {l1_ratio, C}  +  mRMR x {k}
    the frozen tie-break       protocol section 3.2, executable

There is ONE radiomics procedure.  A1 and A2 both call it; only the region
changes.  No CNN, no MIL, no attention, no CLAM, no Transformer, no RF, no
XGBoost, no SVM, no SMOTE, no harmonisation.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# --- frozen primitives, imported verbatim ---------------------------------
from feature_selection import mrmr_rank                                  # noqa: F401
from phase7a_models import (PlattCalibrator, TrainingOnlyPreprocessor,   # noqa: F401
                            select_threshold)

ELASTIC_NET = "elastic_net"
MRMR = "mrmr"


# ---------------------------------------------------------------------------
# candidate enumeration
# ---------------------------------------------------------------------------

def candidates(cfg) -> List[dict]:
    """The frozen candidate set: 32 elastic-net + 2 mRMR, in deterministic order."""
    r = cfg["models"]["radiomics"]
    out = []
    for l1 in r["elastic_net"]["grid"]["l1_ratio"]:
        for C in r["elastic_net"]["grid"]["C"]:
            out.append({"id": "elastic_net|l1=%.2f|C=%g" % (float(l1), float(C)),
                        "family": ELASTIC_NET, "l1_ratio": float(l1), "C": float(C),
                        "k": None})
    for k in r["mrmr"]["k"]:
        out.append({"id": "mrmr|k=%d|C=%g" % (int(k), float(r["mrmr"]["C"])),
                    "family": MRMR, "l1_ratio": None, "C": float(r["mrmr"]["C"]),
                    "k": int(k)})
    return out


# ---------------------------------------------------------------------------
# weighted estimators
# ---------------------------------------------------------------------------

def _fit(est, xs, y, w):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        est.fit(xs, y, sample_weight=w)
    converged = not any(issubclass(c.category, ConvergenceWarning) for c in caught)
    return est, converged


@dataclass
class WeightedCappedElasticNet:
    """Elastic-net logistic + the frozen <= 8 feature cap, both weighted.

    The ranking that chooses the retained features is computed from the
    first-stage coefficients of the SAME training rows with the SAME weights.
    No held-out patient participates in either stage.
    """
    C: float
    l1_ratio: float
    max_features: int
    est_cfg: dict
    class_weight: Optional[str] = None

    selected: List[int] = field(default_factory=list)
    n_nonzero_stage1: int = 0
    coefficients: Optional[np.ndarray] = None
    intercept: float = 0.0
    model: Optional[LogisticRegression] = None
    constant_prior: Optional[float] = None
    converged_stage1: bool = True
    converged_stage2: bool = True

    def _make(self):
        return LogisticRegression(
            penalty=self.est_cfg["penalty"], solver=self.est_cfg["solver"],
            C=float(self.C), l1_ratio=float(self.l1_ratio),
            class_weight=self.class_weight,
            max_iter=int(self.est_cfg["max_iter"]), tol=float(self.est_cfg["tol"]),
            random_state=int(self.est_cfg["random_state"]))

    def fit(self, xs: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None):
        stage1, self.converged_stage1 = _fit(self._make(), xs, y, w)
        coef = stage1.coef_[0]
        nz = np.flatnonzero(coef != 0.0)
        self.n_nonzero_stage1 = int(nz.size)
        order = sorted(nz.tolist(), key=lambda j: (-abs(float(coef[j])), int(j)))
        self.selected = sorted(int(j) for j in order[: int(self.max_features)])

        if not self.selected:
            # the weighted prior-probability constant - recorded, never substituted
            ww = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
            self.constant_prior = float(np.sum(ww * (np.asarray(y) == 1)) / np.sum(ww))
            self.coefficients = np.zeros(0)
            self.intercept = 0.0
            self.model = None
            return self

        stage2, self.converged_stage2 = _fit(self._make(), xs[:, self.selected], y, w)
        self.model = stage2
        self.coefficients = stage2.coef_[0].copy()
        self.intercept = float(stage2.intercept_[0])
        return self

    def predict_proba(self, xs: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.full(xs.shape[0], float(self.constant_prior))
        return self.model.predict_proba(xs[:, self.selected])[:, 1]


@dataclass
class WeightedMrmrRidge:
    """mRMR (training-only FCD ranking) followed by L2 logistic, both weighted.

    The frozen implementation from src/feature_selection.py is used verbatim;
    only the downstream ridge sees the weights, because the FCD criterion is a
    property of the training feature matrix and its labels.
    """
    k: int
    est_cfg: dict
    class_weight: Optional[str] = None

    selected: List[int] = field(default_factory=list)
    coefficients: Optional[np.ndarray] = None
    intercept: float = 0.0
    model: Optional[LogisticRegression] = None
    converged: bool = True

    def fit(self, xs: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None):
        rank = mrmr_rank(xs, y, n_select=int(self.k),
                         zero_variance_relevance=float(self.est_cfg["zero_variance_relevance"]),
                         zero_variance_correlation=float(self.est_cfg["zero_variance_correlation"]))
        self.selected = [int(i) for i in rank.select(int(self.k))]
        est = LogisticRegression(penalty=self.est_cfg["penalty"], solver=self.est_cfg["solver"],
                                 C=float(self.est_cfg["C"]), class_weight=self.class_weight,
                                 max_iter=int(self.est_cfg["max_iter"]),
                                 tol=float(self.est_cfg["tol"]),
                                 random_state=int(self.est_cfg["random_state"]))
        self.model, self.converged = _fit(est, xs[:, self.selected], y, w)
        self.coefficients = self.model.coef_[0].copy()
        self.intercept = float(self.model.intercept_[0])
        return self

    def predict_proba(self, xs: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(xs[:, self.selected])[:, 1]


def fit_stacker(cfg, z: np.ndarray, y: np.ndarray, w: Optional[np.ndarray],
                class_weight: Optional[str] = None):
    """The A3 ridge-logistic stacker.  Exactly two inputs, ever."""
    c = cfg["models"]["A3"]
    if z.shape[1] != 2:
        raise RuntimeError("A3 must have exactly 2 inputs, got %d" % z.shape[1])
    est = LogisticRegression(penalty=c["penalty"], solver=c["solver"], C=float(c["C"]),
                             class_weight=class_weight, max_iter=int(c["max_iter"]),
                             tol=float(c["tol"]), random_state=int(c["random_state"]))
    est, _ = _fit(est, z, y, w)
    return est


# ---------------------------------------------------------------------------
# ONE procedure, used by both A1 (gtv) and A2 (rim)
# ---------------------------------------------------------------------------

def make_preprocessor(cfg):
    return TrainingOnlyPreprocessor(
        abs_tol=float(cfg["pipeline"]["variance_filter"]["abs_tol"]),
        rel_tol=float(cfg["pipeline"]["variance_filter"]["rel_tol"]),
        all_nan_fill=float(cfg["pipeline"]["imputer"]["all_nan_fill"]))


def fit_candidate(cfg, x_tr, y_tr, w_tr, cand, class_weight=None):
    """Fit ONE candidate on ONE training partition.  Returns (preprocessor, model).

    `w_tr` carries the four-stratum source-balanced weights for the pooled
    model; `class_weight` carries ordinary balancing for the single-cohort
    transfer experiments.  Exactly one of them is ever non-None.
    """
    if w_tr is not None and class_weight is not None:
        raise RuntimeError("source weights and class_weight are mutually exclusive")
    r = cfg["models"]["radiomics"]
    pre = make_preprocessor(cfg).fit(x_tr)
    xs = pre.transform(x_tr)
    if cand["family"] == ELASTIC_NET:
        est = WeightedCappedElasticNet(
            C=cand["C"], l1_ratio=cand["l1_ratio"],
            max_features=int(r["elastic_net"]["feature_cap"]["max_features"]),
            est_cfg=r["elastic_net"], class_weight=class_weight).fit(xs, y_tr, w_tr)
    elif cand["family"] == MRMR:
        est = WeightedMrmrRidge(k=cand["k"], est_cfg=r["mrmr"],
                                class_weight=class_weight).fit(xs, y_tr, w_tr)
    else:
        raise RuntimeError("unknown selector family %r" % cand["family"])
    return pre, est


def auc(y, p):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def inner_surface(cfg, x, y, strata_weights, inner, class_weight=None, progress=None):
    """Evaluate every candidate by mean inner-CV ROC-AUC on ONE training partition.

    `x`, `y` are the training partition's rows.  `strata_weights` is a callable
    idx -> weights for that inner-training subset, or None when ordinary class
    balancing is used.  Returns (surface, inner_oof_by_candidate).
    """
    surface, oof_cache = [], {}
    for ci, cand in enumerate(candidates(cfg)):
        oof = np.full(len(y), np.nan)
        aucs, n_sel = [], []
        for a, b in inner:
            w = None if strata_weights is None else strata_weights(a)
            pre, est = fit_candidate(cfg, x[a], y[a], w, cand, class_weight=class_weight)
            oof[b] = est.predict_proba(pre.transform(x[b]))
            aucs.append(auc(y[b], oof[b]))
            n_sel.append(len(est.selected))
        surface.append({"candidate_id": cand["id"], "family": cand["family"],
                        "l1_ratio": cand["l1_ratio"], "C": cand["C"], "k": cand["k"],
                        "mean_inner_roc_auc": float(np.mean(aucs)),
                        "sd_inner_roc_auc": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                        "inner_roc_auc": [float(v) for v in aucs],
                        "mean_n_selected": float(np.mean(n_sel)),
                        "config_order": ci})
        oof_cache[cand["id"]] = oof
        if progress is not None:
            progress(ci + 1, len(candidates(cfg)), cand["id"])
    return surface, oof_cache


def choose_candidate(cfg, surface):
    """The frozen tie-break of protocol section 3.2, applied mechanically.

    Primary criterion is the mean inner-CV ROC-AUC.  Candidates within
    `tie_tolerance` of the maximum form the tie group; inside it the four
    documented rules decide, in order.
    """
    tol = float(cfg["models"]["radiomics"]["selection"]["tie_tolerance"])
    best = max(s["mean_inner_roc_auc"] for s in surface
               if np.isfinite(s["mean_inner_roc_auc"]))
    group = [s for s in surface if np.isfinite(s["mean_inner_roc_auc"])
             and best - s["mean_inner_roc_auc"] <= tol]
    family_rank = {MRMR: 0, ELASTIC_NET: 1}
    group_sorted = sorted(group, key=lambda s: (
        s["mean_n_selected"],                                   # 1 fewer features
        family_rank[s["family"]],                               # 2 simpler selector
        s["C"], s["l1_ratio"] if s["l1_ratio"] is not None else -1.0,   # 3 stronger reg.
        s["candidate_id"]))                                     # 4 lexical
    chosen = group_sorted[0]
    return chosen, {"max_mean_inner_roc_auc": float(best),
                    "tie_tolerance": tol,
                    "tie_group_size": len(group),
                    "tie_group_ids": [s["candidate_id"] for s in group_sorted],
                    "chosen_candidate_id": chosen["candidate_id"],
                    "rule": ["fewer_selected_features", "mrmr_before_elastic_net",
                             "smaller_C_then_smaller_l1_ratio", "lexical_candidate_id"]}


def calibrate_and_threshold(cfg, inner_oof, y_tr, p_test):
    """Platt + balanced-accuracy threshold, from inner-OOF TRAINING rows only.

    The calibrator is never class-reweighted and never stratum-weighted
    (protocol section 9.2).  Validity is reported, never silently repaired.
    """
    valid = bool(len(np.unique(np.asarray(y_tr))) == 2 and np.all(np.isfinite(inner_oof)))
    if valid:
        cal = PlattCalibrator().fit(inner_oof, y_tr, cfg["calibration"])
        valid = bool(np.isfinite(cal.a) and np.isfinite(cal.b))
    if not valid:
        cal = PlattCalibrator()
        cal.a, cal.b, cal.clip, cal.n_fit = 1.0, 0.0, float(cfg["calibration"]["clip"]), 0
    cal_inner = cal.apply(inner_oof)
    thr, bal, _ = select_threshold(cal_inner, y_tr, cfg["threshold"]["tie_break"])
    return {"calibrator": cal, "calibrated_inner_oof": cal_inner,
            "threshold": float(thr), "inner_balanced_accuracy": float(bal),
            "calibrated_test": cal.apply(p_test),
            "platt_a": float(cal.a), "platt_b": float(cal.b),
            "platt_valid": bool(valid), "n_calibration_rows": int(cal.n_fit)}
