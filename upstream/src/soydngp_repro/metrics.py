"""Epoch-level metrics.

Computed once on the full validation set after the epoch finishes. Never
per-batch, never averaged over cumulative batches (the legacy error).
"""
import numpy as np


def pcc(y_true, y_pred):
    """Pearson correlation coefficient.

    Degenerate input (zero variance on either side) returns NaN, not 0.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.std() == 0.0 or y_pred.std() == 0.0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def mse(y_true, y_pred):
    d = np.asarray(y_true, dtype=np.float64).ravel() - np.asarray(y_pred, dtype=np.float64).ravel()
    return float((d ** 2).mean())


def smooth_l1(y_true, y_pred, beta=0.1):
    d = np.abs(np.asarray(y_true, dtype=np.float64).ravel()
               - np.asarray(y_pred, dtype=np.float64).ravel())
    return float(np.where(d <= beta, 0.5 * d ** 2, d - 0.5 * beta ** 2).mean())


def _class_counts(y_true, y_pred, labels):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)
    match = y_true[:, None] == labels[None, :]
    pred_match = y_pred[:, None] == labels[None, :]
    tp = (match & pred_match).sum(axis=0)
    fp = (~match & pred_match).sum(axis=0)
    fn = (match & ~pred_match).sum(axis=0)
    return labels, tp, fp, fn


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float((y_true == y_pred).mean())


def macro_f1(y_true, y_pred, labels=None):
    _, tp, fp, fn = _class_counts(y_true, y_pred, labels)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)
    return float(f1.mean())


def macro_precision(y_true, y_pred, labels=None):
    _, tp, fp, _ = _class_counts(y_true, y_pred, labels)
    denom = tp + fp
    prec = np.where(denom > 0, tp / np.where(denom > 0, denom, 1), 0.0)
    return float(prec.mean())


def macro_recall(y_true, y_pred, labels=None):
    _, tp, _, fn = _class_counts(y_true, y_pred, labels)
    denom = tp + fn
    rec = np.where(denom > 0, tp / np.where(denom > 0, denom, 1), 0.0)
    return float(rec.mean())


def confusion_matrix(y_true, y_pred, labels=None):
    from sklearn.metrics import confusion_matrix as _cm
    return _cm(y_true, y_pred, labels=labels)
