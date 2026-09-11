"""G4 T3: metric contract.

Metrics are computed once on the full validation set; PCC is checked
against the numpy reference, macro-F1 against the sklearn reference, and
degenerate PCC must return NaN, not 0.

Run:  python tests/test_metrics_contract.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from soydngp_repro import metrics
from soydngp_repro.trainer import evaluate_epoch_regression


def test_pcc_matches_numpy_reference():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=500)
    y_pred = 0.8 * y_true + rng.normal(scale=0.5, size=500)
    ref = np.corrcoef(y_true, y_pred)[0, 1]
    assert abs(metrics.pcc(y_true, y_pred) - ref) < 1e-12


def test_pcc_degenerate_returns_nan():
    assert np.isnan(metrics.pcc(np.ones(10), np.ones(10)))
    assert np.isnan(metrics.pcc(np.ones(10), np.arange(10, dtype=float)))
    assert np.isnan(metrics.pcc(np.arange(10, dtype=float), np.zeros(10)))


def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 4, size=400)
    y_pred = np.where(rng.random(400) < 0.7, y_true, rng.integers(0, 4, size=400))
    ref = f1_score(y_true, y_pred, average='macro', zero_division=0)
    assert abs(metrics.macro_f1(y_true, y_pred) - ref) < 1e-12
    # zero-support class: class 0 is never predicted
    yt = np.array([0, 0, 1, 1, 2, 2])
    yp = np.array([0, 1, 1, 2, 2, 2])
    ref2 = f1_score(yt, yp, labels=[0, 1, 2], average='macro', zero_division=0)
    assert abs(metrics.macro_f1(yt, yp, labels=[0, 1, 2]) - ref2) < 1e-12


def test_accuracy_precision_recall_match_sklearn():
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 3, size=300)
    y_pred = np.where(rng.random(300) < 0.6, y_true, rng.integers(0, 3, size=300))
    assert abs(metrics.accuracy(y_true, y_pred) - accuracy_score(y_true, y_pred)) < 1e-12
    assert abs(metrics.macro_precision(y_true, y_pred)
               - precision_score(y_true, y_pred, average='macro', zero_division=0)) < 1e-12
    assert abs(metrics.macro_recall(y_true, y_pred)
               - recall_score(y_true, y_pred, average='macro', zero_division=0)) < 1e-12


def test_mse_smooth_l1():
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([0.1, 0.9, 2.5])
    assert metrics.mse(y_true, y_pred) == float(np.mean((y_true - y_pred) ** 2))
    # |d| = [0.1, 0.1, 0.5]; smooth_l1: 0.5*d^2 if d<=0.1 else d-0.005
    expect = (0.5 * 0.01 + 0.5 * 0.01 + (0.5 - 0.005)) / 3
    assert abs(metrics.smooth_l1(y_true, y_pred) - expect) < 1e-12


def test_full_epoch_collection():
    """evaluate_epoch covers the WHOLE validation set across batches,
    so metrics are computed once on the full set, not per batch."""
    import torch.nn as nn

    class MeanHead(nn.Module):
        def forward(self, x):
            return x.mean(dim=(1, 2, 3)).unsqueeze(1)

    model = MeanHead()
    x = torch.randn(10, 3, 4, 4)
    y = torch.rand(10)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y),
                                         batch_size=4)  # 3 batches
    y_true, y_pred = evaluate_epoch_regression(model, loader, "cpu")
    assert len(y_true) == 10 and len(y_pred) == 10
    assert np.allclose(y_true, y.numpy())


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
        print('PASS', fn.__name__)
    print('all metric contract tests passed')
