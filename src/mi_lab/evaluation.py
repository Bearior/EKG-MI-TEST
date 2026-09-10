"""Binary metrics with STEMI as positive class and patient bootstrap intervals."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def _validate(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError('Both binary classes and matching one-dimensional arrays required')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Probabilities must be finite and between zero and one')
    return y, p


def binary_metrics(y, p, threshold=0.5):
    y, p = _validate(y, p)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must be in [0, 1]')
    predicted = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        'auroc': float(roc_auc_score(y, p)),
        'average_precision': float(average_precision_score(y, p)),
        'accuracy': float(np.mean(y == predicted)),
        'balanced_accuracy': float(balanced_accuracy_score(y, predicted)),
        'sensitivity': float(tp / (tp + fn)),
        'specificity': float(tn / (tn + fp)),
        'precision': float(tp / (tp + fp)) if tp + fp else 0.0,
        'f1': float(f1_score(y, predicted, zero_division=0)),
        'brier_score': float(brier_score_loss(y, p)),
        'threshold': float(threshold),
        'n_patients': len(y),
        'stemi_prevalence': float(np.mean(y)),
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def choose_threshold(y, p):
    """Maximize validation Youden J; ties choose threshold nearest 0.5."""
    y, p = _validate(y, p)
    fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    valid = np.isfinite(thresholds) & (thresholds <= 1)
    thresholds, scores = thresholds[valid], (tpr - fpr)[valid]
    candidates = thresholds[np.isclose(scores, scores.max(), rtol=0, atol=1e-12)]
    return float(candidates[np.argmin(abs(candidates - 0.5))])


def bootstrap_intervals(y, p, threshold, n_bootstrap=1000, seed=42):
    """Percentile intervals from resampling patients, one record per patient."""
    y, p = _validate(y, p)
    if n_bootstrap < 1:
        raise ValueError('n_bootstrap must be positive')
    rng = np.random.default_rng(seed)
    keys = ['auroc', 'average_precision', 'balanced_accuracy', 'sensitivity', 'specificity']
    samples = {key: [] for key in keys}
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) != 2:
            continue
        metrics = binary_metrics(y[index], p[index], threshold)
        for key in keys:
            samples[key].append(metrics[key])
    if not samples['auroc']:
        raise ValueError('No bootstrap sample contained both classes')
    return {key: np.quantile(values, [0.025, 0.975]).tolist() for key, values in samples.items()}
