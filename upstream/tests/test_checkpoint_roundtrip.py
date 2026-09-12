"""G4: checkpoint save/reload exactness.

A checkpoint is a dict with model_state_dict, optimizer_state_dict,
epoch, best_metric, protocol, config. Reloading into a fresh model must
reproduce the state and the predictions exactly. No torch.save(net, ...).

Run:  python tests/test_checkpoint_roundtrip.py
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import torch
import torch.nn as nn

from soydngp_repro.protocols import Protocol
from soydngp_repro.trainer import load_checkpoint, save_checkpoint


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 1)

    def forward(self, x):
        return self.lin(x)


def test_checkpoint_roundtrip_exact():
    torch.manual_seed(0)
    model = Tiny()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    x = torch.randn(4, 4)
    (model(x) ** 2).mean().backward()
    opt.step()  # move optimizer state away from its initial values

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ck.pt")
        save_checkpoint(path, model, opt, epoch=7, best_metric=0.42,
                        protocol=Protocol.PAPER_REPRO, config={"lr": 1e-3})
        ckpt = torch.load(path, map_location="cpu")
        assert set(ckpt) == {"model_state_dict", "optimizer_state_dict",
                             "epoch", "best_metric", "protocol", "config"}
        assert ckpt["epoch"] == 7
        assert ckpt["best_metric"] == 0.42
        assert ckpt["protocol"] == "paper_repro"
        assert ckpt["config"] == {"lr": 1e-3}

        model2 = Tiny()
        opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3, weight_decay=1e-5)
        load_checkpoint(path, model2, opt2)
        for k, v in model.state_dict().items():
            assert torch.equal(model2.state_dict()[k], v), k

        def eq(a, b):
            if torch.is_tensor(a) and torch.is_tensor(b):
                return torch.equal(a, b)
            if isinstance(a, dict) and isinstance(b, dict):
                return set(a) == set(b) and all(eq(a[k], b[k]) for k in a)
            if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                return len(a) == len(b) and all(eq(p, q) for p, q in zip(a, b))
            return a == b

        for k in opt.state_dict():
            assert eq(opt2.state_dict()[k], opt.state_dict()[k]), k
        with torch.no_grad():
            assert torch.equal(model(x), model2(x))


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
        print('PASS', fn.__name__)
    print('all checkpoint roundtrip tests passed')
