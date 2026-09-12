"""G4 T2: classification target contract.

Targets must reach the loss as int64 (B,), even when the source array is
float (legacy label arrays are float). Logits must be 2-D with a matching
batch size. Regression and classification never share a dtype-guessing
path.

Run:  python tests/test_classification_target_contract.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import torch
import torch.nn as nn

from soydngp_repro.trainer import train_epoch_classification


class LogitStub(nn.Module):
    """(B, C, H, W) -> (B, 3) logits."""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        return x.mean(dim=(1, 2, 3)).unsqueeze(1).repeat_interleave(3, dim=1) + self.bias


class RecordingCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, logits, target):
        self.calls.append((logits.detach().clone(), target.detach().clone()))
        return nn.functional.cross_entropy(logits, target)


def _loader(x, y, batch_size):
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y),
                                       batch_size=batch_size)


def test_integer_labels_reach_loss_as_int64():
    torch.manual_seed(0)
    model = LogitStub()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(4, 3, 4, 4)
    y = torch.tensor([0, 1, 2, 1])
    loss = RecordingCE()
    train_epoch_classification(model, _loader(x, y, 4), loss, opt, "cpu")
    logits, target = loss.calls[0]
    assert target.dtype == torch.int64, target.dtype
    assert target.shape == (4,), target.shape
    assert logits.ndim == 2
    assert logits.shape[0] == target.shape[0]
    torch.testing.assert_close(target, torch.tensor([0, 1, 2, 1], dtype=torch.int64))


def test_float_labels_cast_to_int64():
    """Legacy label arrays are float; the trainer must cast, not guess."""
    torch.manual_seed(1)
    model = LogitStub()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(4, 3, 4, 4)
    y = torch.tensor([0.0, 1.0, 2.0, 1.0])  # float64, like upstream .float() labels
    loss = RecordingCE()
    train_epoch_classification(model, _loader(x, y, 4), loss, opt, "cpu")
    logits, target = loss.calls[0]
    assert target.dtype == torch.int64, target.dtype
    assert target.shape == (4,), target.shape
    assert logits.ndim == 2
    assert logits.shape[0] == target.shape[0]
    torch.testing.assert_close(target, torch.tensor([0, 1, 2, 1], dtype=torch.int64))


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
        print('PASS', fn.__name__)
    print('all classification target contract tests passed')
