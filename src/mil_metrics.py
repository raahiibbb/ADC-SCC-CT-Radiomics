"""Phase 3 - patient-level metrics.

Every metric here consumes ONE probability per PATIENT.  Nothing patch-level is
ever scored.  ADC (label 1) is the positive class; the reported
threshold-dependent metrics use a fixed threshold of 0.5.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             matthews_corrcoef, precision_recall_curve,
                             roc_auc_score, roc_curve)


def patient_metrics(y_true: Sequence[int], y_prob: Sequence[float],
                    threshold: float = 0.5) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    if y.size == 0:
        raise ValueError("no patients to score")
    if not np.all(np.isfinite(p)):
        raise ValueError("non-finite predicted probabilities")

    yhat = (p >= threshold).astype(int)
    both_classes = len(np.unique(y)) == 2

    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * prec * sens / (prec + sens)) if (prec + sens) else 0.0
    bal_acc = np.nanmean([sens, spec])
    mcc = matthews_corrcoef(y, yhat) if len(np.unique(yhat)) > 1 or both_classes else float("nan")

    return {
        "n_patients": int(y.size),
        "n_adc": int((y == 1).sum()),
        "n_scc": int((y == 0).sum()),
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if both_classes else float("nan"),
        "balanced_accuracy": float(bal_acc),
        "sensitivity_adc": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "mcc": float(mcc),
        "accuracy": float((yhat == y).mean()),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "threshold": float(threshold),
    }


def curve_points(y_true: Sequence[int], y_prob: Sequence[float]) -> Dict[str, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    fpr, tpr, roc_thr = roc_curve(y, p)
    prec, rec, pr_thr = precision_recall_curve(y, p)
    return {"fpr": fpr, "tpr": tpr, "roc_thresholds": roc_thr,
            "precision": prec, "recall": rec, "pr_thresholds": pr_thr}


def mean_sd(values: Sequence[float]) -> Dict[str, float]:
    v = np.asarray([x for x in values], dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    return {"mean": float(finite.mean()),
            "sd": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
            "n": int(finite.size)}
