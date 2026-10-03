"""Phase 5C - patient-level feature selection for local-radiomics MIL.

THE MIL RULE THIS MODULE EXISTS TO ENFORCE
------------------------------------------
The ADC/SCC label is known only at PATIENT level.  Copying it onto the 37 464
individual patches and running a supervised selector on that matrix would
pseudo-replicate patients, weight a 900-patch bag 112x more heavily than an
8-patch bag, and violate the multiple-instance assumption outright.

So every supervised statistic in this module consumes a matrix whose rows are
PATIENTS.  For each patient and each of the 74 original radiomic dimensions:

    patient_summary[p, j] = median over that patient's retained patches of
                            features[i, j]

One patient -> exactly one row, whatever the bag size.  The summary of patient p
is a function of patient p's own frozen bag alone: it uses no label, no other
patient, and no scaling, so building all 202 summaries up front is not a
cross-patient operation and cannot leak.  What must never leak - and what this
module keeps under the caller's explicit control - is WHICH PATIENTS are used to
FIT a selector.  Every fitting function here takes the patient row-subset
explicitly and computes every statistic from that subset alone.

The summaries are used ONLY to decide which original radiomic dimensions to
retain.  The selected indices are then applied back to the untouched patch bags
([N_patches, 74] -> [N_patches, K_selected]) and the final classifier remains
MIL.  The patient-summary classifier is never the model.

SELECTORS
---------
mRMR - the deterministic CONTINUOUS formulation of Ding & Peng (2003),
"Minimum Redundancy Feature Selection from Microarray Gene Expression Data",
Table 1 / section 3.2, criterion FCD ("F-test correlation difference"):

    relevance   F(i, h)   one-way ANOVA F statistic of feature i against the
                          class variable h, their Eq. (8)
    redundancy  |c(i,j)|  absolute Pearson correlation between features i and j

    first feature   argmax_i F(i, h)
    then, greedily  argmax over unselected i of
                        F(i, h) - (1/|S|) * sum over j in S of |c(i, j)|

    Both statistics are scale invariant, so mRMR fits no scaler at all.  The
    greedy order does not depend on K, so one ranking serves every K and the
    selected sets for K = 8, 12, 16, 24, 32 are nested prefixes of it.

LASSO - L1-penalised logistic regression (liblinear, class_weight="balanced")
on the patient summaries, after a StandardScaler fitted on the FITTING
PATIENTS' summaries only.  That scaler exists for feature selection alone and
never touches a patch.  A feature is selected iff |coefficient| > 1e-8.  A
candidate that selects zero features is reported invalid; features are never
silently substituted.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 1.  Patient summaries
# ---------------------------------------------------------------------------

SUMMARY_STATISTICS = {"median": lambda x: np.median(x, axis=0)}


def patient_summary(features: np.ndarray, statistic: str = "median") -> np.ndarray:
    """One 74-D vector from one patient's [N_patches, 74] frozen bag.

    Uses only this patient's own patches.  No label, no other patient, no
    scaling.  Bag size does not change the shape of the output: an 8-patch bag
    and a 900-patch bag both produce exactly one row.
    """
    if statistic not in SUMMARY_STATISTICS:
        raise ValueError("unsupported patient summary statistic %r" % statistic)
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError("expected a non-empty [N_patches, F] bag, got %r" % (x.shape,))
    out = SUMMARY_STATISTICS[statistic](x)
    if not np.all(np.isfinite(out)):
        raise RuntimeError("non-finite patient summary")
    return out.astype(np.float64)


@dataclass
class PatientSummaries:
    """One row per patient; `matrix[i]` summarises `patient_ids[i]`."""
    patient_ids: List[str]
    matrix: np.ndarray            # [n_patients, 74] float64
    labels: np.ndarray            # [n_patients] int64
    n_patches: np.ndarray         # [n_patients] int64
    feature_names: List[str]
    statistic: str = "median"

    def index_of(self, pids: Sequence[str]) -> np.ndarray:
        pos = {p: i for i, p in enumerate(self.patient_ids)}
        missing = [p for p in pids if p not in pos]
        if missing:
            raise KeyError("no patient summary for %s" % missing[:5])
        return np.asarray([pos[p] for p in pids], dtype=int)

    def subset(self, pids: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """The (X, y) a selector may see.  Exactly len(pids) rows, one each."""
        pids = list(pids)
        if len(set(pids)) != len(pids):
            raise ValueError("duplicate patients in a feature-selection subset")
        idx = self.index_of(pids)
        return self.matrix[idx].copy(), self.labels[idx].copy()

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(("|".join(self.patient_ids)).encode("utf-8"))
        h.update(np.ascontiguousarray(self.matrix, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(self.labels, dtype=np.int64).tobytes())
        return h.hexdigest()[:16]


def build_patient_summaries(cohort, statistic: str = "median") -> PatientSummaries:
    pids = list(cohort.patient_ids)
    rows = np.vstack([patient_summary(cohort.bags[p].features, statistic) for p in pids])
    labels = np.asarray([int(cohort.bags[p].label) for p in pids], dtype=np.int64)
    npatch = np.asarray([int(cohort.bags[p].n_patches) for p in pids], dtype=np.int64)
    return PatientSummaries(patient_ids=pids, matrix=rows, labels=labels,
                            n_patches=npatch, feature_names=list(cohort.feature_names),
                            statistic=statistic)


# ---------------------------------------------------------------------------
# 2.  mRMR - Ding & Peng (2003), continuous variables, FCD criterion
# ---------------------------------------------------------------------------

def anova_f_statistic(x: np.ndarray, y: np.ndarray,
                      zero_variance_value: float = 0.0) -> np.ndarray:
    """Ding & Peng Eq. (8), written out rather than called from a library.

        F(g, h) = [ sum_k n_k (mean_k - mean)^2 / (K - 1) ] / pooled_variance
        pooled_variance = [ sum_k (n_k - 1) var_k ] / (n - K)

    with var_k the unbiased within-class variance.  For two classes this is the
    square of the two-sample t statistic.  A feature with zero pooled variance
    carries no class information and is given `zero_variance_value` instead of
    a NaN, so the ranking stays total and deterministic.

    `x` is [n_patients, n_features]; `y` is [n_patients].  Rows are patients.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y).ravel()
    if x.shape[0] != y.shape[0]:
        raise ValueError("x has %d rows but y has %d" % (x.shape[0], y.shape[0]))
    classes = np.unique(y)
    k = classes.size
    n = x.shape[0]
    if k < 2:
        raise ValueError("the F statistic needs at least two classes")
    if n <= k:
        raise ValueError("not enough samples (%d) for %d classes" % (n, k))

    grand = x.mean(axis=0)
    between = np.zeros(x.shape[1], dtype=np.float64)
    within = np.zeros(x.shape[1], dtype=np.float64)
    for c in classes:
        xc = x[y == c]
        nc = xc.shape[0]
        if nc < 2:
            raise ValueError("class %r has %d members; the F statistic needs >= 2" % (c, nc))
        between += nc * (xc.mean(axis=0) - grand) ** 2
        within += (nc - 1) * xc.var(axis=0, ddof=1)

    msb = between / float(k - 1)
    pooled = within / float(n - k)
    f = np.full(x.shape[1], float(zero_variance_value), dtype=np.float64)
    ok = pooled > 0.0
    f[ok] = msb[ok] / pooled[ok]
    f[~np.isfinite(f)] = float(zero_variance_value)
    return f


def absolute_pearson_matrix(x: np.ndarray,
                            zero_variance_value: float = 0.0) -> np.ndarray:
    """|c(i, j)| between feature columns, computed on the given patient rows."""
    x = np.asarray(x, dtype=np.float64)
    sd = x.std(axis=0, ddof=1)
    ok = sd > 0.0
    corr = np.full((x.shape[1], x.shape[1]), float(zero_variance_value), dtype=np.float64)
    if ok.sum() >= 2:
        sub = np.corrcoef(x[:, ok], rowvar=False)
        sub = np.nan_to_num(sub, nan=float(zero_variance_value))
        idx = np.where(ok)[0]
        corr[np.ix_(idx, idx)] = np.abs(sub)
    np.fill_diagonal(corr, 1.0)
    return corr


@dataclass
class MrmrRanking:
    """The greedy mRMR order, plus the trace that makes it auditable."""
    order: List[int]                       # feature indices, best first
    relevance: np.ndarray                  # F statistic per feature
    steps: List[dict] = field(default_factory=list)
    n_fit_patients: int = 0
    n_fit_adc: int = 0
    n_fit_scc: int = 0

    def select(self, k: int) -> List[int]:
        if k > len(self.order):
            raise ValueError("K=%d exceeds the %d ranked features" % (k, len(self.order)))
        return list(self.order[:k])


def mrmr_rank(x: np.ndarray, y: np.ndarray, n_select: int,
              zero_variance_relevance: float = 0.0,
              zero_variance_correlation: float = 0.0) -> MrmrRanking:
    """Greedy FCD mRMR over the PATIENT rows in `x`.

    Every statistic - the F values and the correlation matrix alike - is
    computed from exactly these rows.  Ties go to the lowest feature index, so
    the ranking is deterministic.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y).ravel()
    n_features = x.shape[1]
    n_select = int(n_select)
    if not 1 <= n_select <= n_features:
        raise ValueError("n_select=%d outside 1..%d" % (n_select, n_features))

    rel = anova_f_statistic(x, y, zero_variance_relevance)
    corr = absolute_pearson_matrix(x, zero_variance_correlation)

    first = int(np.argmax(rel))            # np.argmax returns the FIRST maximum
    order = [first]
    steps = [{"step": 1, "selected_index": first, "relevance": float(rel[first]),
              "mean_redundancy": None, "score": float(rel[first]),
              "criterion": "max_relevance"}]

    while len(order) < n_select:
        remaining = np.array([j for j in range(n_features) if j not in set(order)], dtype=int)
        red = corr[np.ix_(remaining, np.asarray(order, dtype=int))].mean(axis=1)
        score = rel[remaining] - red
        best = int(np.argmax(score))       # first maximum -> lowest feature index
        pick = int(remaining[best])
        order.append(pick)
        steps.append({"step": len(order), "selected_index": pick,
                      "relevance": float(rel[pick]),
                      "mean_redundancy": float(red[best]),
                      "score": float(score[best]),
                      "criterion": "FCD_difference"})

    return MrmrRanking(order=[int(i) for i in order], relevance=rel, steps=steps,
                       n_fit_patients=int(x.shape[0]),
                       n_fit_adc=int((y == 1).sum()), n_fit_scc=int((y == 0).sum()))


# ---------------------------------------------------------------------------
# 3.  LASSO - L1 logistic regression on the patient summaries
# ---------------------------------------------------------------------------

@dataclass
class LassoSelection:
    C: float
    selected: List[int]                    # ORIGINAL feature indices, ascending
    coefficients: np.ndarray               # [74], in the ORIGINAL feature order
    intercept: float
    n_selected: int
    valid: bool
    reason: str
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    n_fit_patients: int
    n_fit_adc: int
    n_fit_scc: int
    n_iter: int


def lasso_select(x: np.ndarray, y: np.ndarray, c: float, scfg: dict) -> LassoSelection:
    """Fit L1 logistic regression on the PATIENT rows in `x` and read off the support.

    The StandardScaler is fitted here, on these rows only, and is a
    feature-selection artefact: it is never applied to a patch.
    """
    from sklearn.linear_model import LogisticRegression

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y).ravel().astype(int)
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)
    safe = np.where(std > 0.0, std, 1.0)          # a constant column becomes all-zero
    xs = (x - mean) / safe

    clf = LogisticRegression(penalty=str(scfg["penalty"]), solver=str(scfg["solver"]),
                             class_weight=str(scfg["class_weight"]), C=float(c),
                             max_iter=int(scfg["max_iter"]), tol=float(scfg["tol"]),
                             random_state=int(scfg["random_state"]))
    clf.fit(xs, y)
    coef = np.asarray(clf.coef_, dtype=np.float64).ravel()
    tol = float(scfg["coefficient_tolerance"])
    selected = [int(i) for i in np.where(np.abs(coef) > tol)[0]]
    valid = len(selected) > 0
    return LassoSelection(
        C=float(c), selected=selected, coefficients=coef,
        intercept=float(np.asarray(clf.intercept_).ravel()[0]),
        n_selected=len(selected), valid=valid,
        reason="ok" if valid else "selected zero features at C=%g (candidate invalid)" % c,
        scaler_mean=mean, scaler_std=std,
        n_fit_patients=int(x.shape[0]), n_fit_adc=int((y == 1).sum()),
        n_fit_scc=int((y == 0).sum()),
        n_iter=int(np.asarray(clf.n_iter_).ravel()[0]))


# ---------------------------------------------------------------------------
# 4.  The patch-level scaler over the SELECTED dimensions
# ---------------------------------------------------------------------------

@dataclass
class SelectedPatchScaler:
    """Z-scoring of the selected patch dimensions, fitted on training patches only."""
    selected: np.ndarray                   # indices into the ORIGINAL 74
    selected_names: List[str]
    mean: np.ndarray
    std: np.ndarray
    n_train_patients: int
    n_train_patches: int

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float64)[:, self.selected] - self.mean)
                / self.std).astype(np.float32)

    def fingerprint(self) -> str:
        blob = json.dumps({"selected": [int(i) for i in self.selected],
                           "names": list(self.selected_names),
                           "mean": [float(v) for v in self.mean],
                           "std": [float(v) for v in self.std]},
                          sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


def fit_selected_patch_scaler(cohort, train_pids: Sequence[str],
                              selected: Sequence[int], min_std: float = 1e-8
                              ) -> SelectedPatchScaler:
    """Mean/std of the selected dimensions over the TRAINING patients' patches only."""
    sel = np.asarray(sorted(int(i) for i in selected), dtype=int)
    if sel.size == 0:
        raise ValueError("cannot fit a patch scaler on zero selected dimensions")
    x = np.concatenate([cohort.bags[p].features[:, sel] for p in train_pids],
                       axis=0).astype(np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)
    if np.any(std <= float(min_std)):
        bad = [cohort.feature_names[sel[i]] for i in np.where(std <= float(min_std))[0]]
        raise RuntimeError("selected dimension(s) with near-zero training variance: %s" % bad)
    return SelectedPatchScaler(
        selected=sel, selected_names=[cohort.feature_names[i] for i in sel],
        mean=mean, std=std, n_train_patients=len(list(train_pids)),
        n_train_patches=int(x.shape[0]))


# ---------------------------------------------------------------------------
# 5.  Persistence of the patient summaries
# ---------------------------------------------------------------------------

def save_patient_summaries(cfg, ps: PatientSummaries, ppath, bag_manifest: dict) -> dict:
    import pandas as pd

    pcfg = cfg["patient_summary"]
    out_dir = ppath(cfg, pcfg["dir"])
    os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        ppath(cfg, pcfg["npz"]),
        patient_ids=np.array(ps.patient_ids, dtype=object),
        summaries=ps.matrix.astype(np.float64),
        labels=ps.labels.astype(np.int64),
        n_patches=ps.n_patches.astype(np.int64),
        feature_names=np.array(ps.feature_names, dtype=object),
        statistic=np.str_(ps.statistic))

    df = pd.DataFrame(ps.matrix, columns=ps.feature_names)
    df.insert(0, "n_patches", ps.n_patches)
    df.insert(0, "label", ps.labels)
    df.insert(0, "PatientID", ps.patient_ids)
    df.to_csv(ppath(cfg, pcfg["csv"]), index=False)

    meta = {
        "phase": "5c",
        "statistic": ps.statistic,
        "definition": ("patient_summary[p, j] = median over patient p's retained "
                       "patches of the frozen feature j"),
        "rows": "one per patient, regardless of bag size",
        "n_patients": len(ps.patient_ids),
        "n_adc": int((ps.labels == 1).sum()),
        "n_scc": int((ps.labels == 0).sum()),
        "n_features": int(ps.matrix.shape[1]),
        "total_patches_summarised": int(ps.n_patches.sum()),
        "min_bag": int(ps.n_patches.min()), "max_bag": int(ps.n_patches.max()),
        "all_finite": bool(np.all(np.isfinite(ps.matrix))),
        "fingerprint": ps.fingerprint(),
        "uses_labels": False,
        "uses_other_patients": False,
        "scaled": False,
        "note": ("built once for all patients because it is a per-patient transform "
                 "of that patient's own frozen bag; WHICH patients a selector is "
                 "FITTED on is controlled by the caller and is inner-training only"),
        "source_bags": bag_manifest,
    }
    with open(ppath(cfg, pcfg["json"]), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def load_patient_summaries(cfg, ppath) -> PatientSummaries:
    with np.load(ppath(cfg, cfg["patient_summary"]["npz"]), allow_pickle=True) as d:
        return PatientSummaries(
            patient_ids=[str(p) for p in d["patient_ids"]],
            matrix=np.asarray(d["summaries"], dtype=np.float64),
            labels=np.asarray(d["labels"], dtype=np.int64),
            n_patches=np.asarray(d["n_patches"], dtype=np.int64),
            feature_names=[str(f) for f in d["feature_names"]],
            statistic=str(d["statistic"]))


# ---------------------------------------------------------------------------
# 6.  Family of a radiomic feature name (reporting only)
# ---------------------------------------------------------------------------

def feature_family(name: str) -> str:
    n = name.lower()
    for fam in ("firstorder", "glcm", "glrlm", "glszm"):
        if "_%s_" % fam in n:
            return fam
    return "other"


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(int(i) for i in a), set(int(i) for i in b)
    union = sa | sb
    return float(len(sa & sb) / len(union)) if union else 1.0
