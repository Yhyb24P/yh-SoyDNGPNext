"""G4 T4: split protocol contract.

The two protocols are physically separated functions:
    PACKAGE_LEGACY -> 70/30 holdout (stratified when labels are given)
    PAPER_REPRO    -> 10-fold CV
Run:  python tests/test_split_contract.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import numpy as np

from soydngp_repro import splits
from soydngp_repro.protocols import Protocol


def test_protocol_members_separated():
    assert Protocol.PACKAGE_LEGACY.value == "package_legacy"
    assert Protocol.PAPER_REPRO.value == "paper_repro"
    assert len(Protocol) == 2


def test_package_legacy_70_30_holdout():
    n = 1000
    tr, va = splits.package_legacy_holdout(n, train_fraction=0.7, seed=0)
    assert len(tr) == 700 and len(va) == 300
    assert len(set(tr.tolist()) & set(va.tolist())) == 0
    assert set(tr.tolist()) | set(va.tolist()) == set(range(n))


def test_package_legacy_stratified_for_labels():
    y = np.repeat(np.arange(3), 100)  # 300 samples, 100 per class
    tr, va = splits.package_legacy_holdout(len(y), train_fraction=0.7, seed=0, y=y)
    assert len(tr) == 210 and len(va) == 90
    for lab in range(3):
        n_lab = int((y == lab).sum())
        assert int((y[tr] == lab).sum()) == round(n_lab * 0.7)
        assert int((y[va] == lab).sum()) == n_lab - round(n_lab * 0.7)


def test_paper_repro_10_fold_cv():
    n = 100
    folds = splits.paper_repro_folds(n, n_splits=10, seed=0)
    assert len(folds) == 10
    seen = []
    for tr, va in folds:
        assert len(tr) == 90 and len(va) == 10
        assert len(set(tr.tolist()) & set(va.tolist())) == 0
        seen.extend(va.tolist())
    assert sorted(seen) == list(range(n))  # each sample validated exactly once


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
        print('PASS', fn.__name__)
    print('all split contract tests passed')
