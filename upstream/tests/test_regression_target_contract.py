"""G4 T1: regression target contract.

The trainer must pass float32 (B,1) targets to the loss, value-preserving
([0.13, 0.47, 0.89] in, same out; NOT zeroed by an int cast), with the
prediction shape asserted equal (no implicit broadcasting).

Run:  python tests/test_regression_target_contract.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import torch
import torch.nn as nn

from soydngp_repro.trainer import train_epoch_regression


class MeanHead(nn.Module):
    """(B, C, H, W) -> (B, 1)."""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x.mean(dim=(1, 2, 3)).unsqueeze(1) + self.bias


class RecordingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, pred, target):
        self.calls.append((pred.detach().clone(), target.detach().clone()))
        return (pred - target).pow(2).mean()


def _loader(x, y, batch_size):
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y),
                                       batch_size=batch_size)


def test_target_values_preserved_not_zeroed():
    """Hard test: [0.13, 0.47, 0.89] must reach the loss unchanged."""
    torch.manual_seed(0)
    model = MeanHead()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(3, 3, 4, 4)
    y = torch.tensor([0.13, 0.47, 0.89])  # float64, like a numpy label array
    loss = RecordingLoss()
    train_epoch_regression(model, _loader(x, y, 3), loss, opt, "cpu")
    pred, target = loss.calls[0]
    assert target.dtype == torch.float32, target.dtype
    assert target.shape == (3, 1), target.shape
    assert pred.shape == target.shape
    torch.testing.assert_close(target.squeeze(1),
                               torch.tensor([0.13, 0.47, 0.89]), rtol=0.0, atol=1e-6)


def test_no_broadcasting_across_batch_sizes():
    """(B,1) pred vs (B,1) target; the loss must never see (B,B)."""
    for b in (2, 5, 8):
        torch.manual_seed(b)
        model = MeanHead()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(b, 3, 4, 4)
        y = torch.rand(b)
        loss = RecordingLoss()
        train_epoch_regression(model, _loader(x, y, b), loss, opt, "cpu")
        for pred, target in loss.calls:
            assert pred.shape == target.shape == (b, 1), (pred.shape, target.shape)


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
        print('PASS', fn.__name__)
    print('all regression target contract tests passed')
