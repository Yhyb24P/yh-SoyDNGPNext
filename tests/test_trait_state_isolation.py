"""G4: multi-trait state isolation.

Each trait must get its own model, optimizer, epoch counter, best score,
and history. Nothing is shared across the trait loop.

Run:  python tests/test_trait_state_isolation.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import torch
import torch.nn as nn

from soydngp_repro.protocols import Protocol, Task
from soydngp_repro.trainer import build_optimizer, train_traits


def _loaders():
    loaders = {}
    for trait in ("trait_a", "trait_b"):
        x = torch.randn(8, 4)
        y = torch.rand(8)
        ld = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y),
                                          batch_size=8)
        loaders[trait] = (ld, ld)
    return loaders


def test_one_fresh_model_per_trait():
    created = []

    def build_model(trait):
        m = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        created.append(m)
        return m

    results = train_traits(Task.REGRESSION, ["trait_a", "trait_b"], build_model,
                           lambda m: build_optimizer(m, lr=1e-3), _loaders(), "cpu",
                           epochs=2, loss_fn=torch.nn.SmoothL1Loss(),
                           protocol=Protocol.PACKAGE_LEGACY, config={})
    assert len(created) == 2
    assert results["trait_a"]["model"] is created[0]
    assert results["trait_b"]["model"] is created[1]


def test_optimizers_bound_to_own_models():
    def build_model(trait):
        return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))

    results = train_traits(Task.REGRESSION, ["trait_a", "trait_b"], build_model,
                           lambda m: build_optimizer(m, lr=1e-3), _loaders(), "cpu",
                           epochs=1, loss_fn=torch.nn.SmoothL1Loss(),
                           protocol=Protocol.PACKAGE_LEGACY, config={})
    for trait in ("trait_a", "trait_b"):
        m = results[trait]["model"]
        opt = results[trait]["optimizer"]
        opt_ids = {id(p) for pg in opt.param_groups for p in pg["params"]}
        model_ids = {id(p) for p in m.parameters()}
        assert opt_ids == model_ids, trait


def test_histories_and_best_scores_are_per_trait():
    def build_model(trait):
        return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))

    results = train_traits(Task.REGRESSION, ["trait_a", "trait_b"], build_model,
                           lambda m: build_optimizer(m, lr=1e-3), _loaders(), "cpu",
                           epochs=3, loss_fn=torch.nn.SmoothL1Loss(),
                           protocol=Protocol.PACKAGE_LEGACY, config={})
    ha = results["trait_a"]["history"]
    hb = results["trait_b"]["history"]
    assert ha is not hb
    assert [r["epoch"] for r in ha] == [1, 2, 3]
    assert [r["epoch"] for r in hb] == [1, 2, 3]
    assert "best_metric" in results["trait_a"]
    assert "best_metric" in results["trait_b"]


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
        print('PASS', fn.__name__)
    print('all trait state isolation tests passed')
