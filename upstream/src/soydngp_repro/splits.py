"""Split protocols, one function per Protocol member.

PACKAGE_LEGACY mirrors upstream DataProcess.to_dataset:
    regression -> 70/30 random holdout
    quality    -> 70/30 stratified holdout (upstream stratify=trait_value)
Upstream did not fix the random seed; this trainer requires an explicit
seed for reproducibility.

PAPER_REPRO is the scientific protocol: 10-fold CV, KFold(shuffle=True).
"""
import numpy as np
from sklearn.model_selection import KFold, train_test_split


def package_legacy_holdout(n_samples, train_fraction=0.7, seed=0, y=None):
    """70/30 holdout. Stratified when y (labels) is given, plain otherwise.

    Returns (train_idx, val_idx) as int64 arrays.
    """
    idx = np.arange(n_samples)
    if y is None:
        tr, va = train_test_split(idx, train_size=train_fraction, random_state=seed)
    else:
        y_arr = np.asarray(y)
        tr, va, _, _ = train_test_split(idx, y_arr, train_size=train_fraction,
                                         random_state=seed, stratify=y_arr)
    return np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64)


def paper_repro_folds(n_samples, n_splits=10, seed=0):
    """10-fold CV. Returns [(train_idx, val_idx)] x n_splits."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64))
            for tr, va in kf.split(np.arange(n_samples))]
