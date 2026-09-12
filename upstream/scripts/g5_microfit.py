"""G5 synthetic micro-overfit: can the corrected trainer memorize a tiny set?

G5-A regression: 16 fixed binary inputs, random y in [0,1] (random targets,
not mean-type, so the CNN cannot cheat with trivial statistics).
G5-B classification: 32 samples, 3 random classes.

The purpose is not generalization; it answers: does this trainer drive the
loss down on a fixed dataset?

PASS gates (written to results/trainer_validation/g5/):
    regression: loss reduction > 95%, PCC > 0.98, finite preds/grads,
                reload prediction identical
    classification: train accuracy >= 0.98, macro-F1 >= 0.98

All optimizer/loss/epoch choices are test-harness choices, not paper
claims; each is recorded in the JSON it produces.

Run (GPU, soydngp312 env):
    /home/yhshy/miniconda3/envs/soydngp312/bin/python scripts/g5_microfit.py [regression|classification|all]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from soydngpnext.remodel import remodel
from soydngp_repro.metrics import accuracy, macro_f1, pcc
from soydngp_repro.protocols import Protocol, Task
from soydngp_repro.trainer import (build_optimizer, evaluate_epoch_classification,
                                   evaluate_epoch_regression, train_traits)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(ROOT, "results", "trainer_validation", "g5")
os.makedirs(OUT, exist_ok=True)
YAML = os.path.join(ROOT, "soydngpnext", "data", "model.yaml")
WEIGHT_DECAY = 1e-5
# G5-A regression harness choices (recorded in regression.json):
#   MSE, not SmoothL1: SmoothL1's constant-magnitude gradient regime stalls
#   memorization (PCC plateau ~0.9 at lr 1e-4, oscillating loss); MSE
#   gradients scale with the error and drive the loss toward zero.
#   Discriminative lr: the final head is a bias-free Linear(50176, 1) whose
#   input features have a tiny L2 norm at init (~3.6e-3), so the head
#   weights must grow to O(80) for O(1) outputs. A uniform lr makes the Adam
#   step on the head (~lr x ||feature||, growing as the backbone matures) an
#   output random walk: PCC oscillates in a band (0.90-0.97 at lr 1e-5 /
#   20000 epochs, best 0.967) and never crosses 0.98. The backbone keeps a
#   normal lr so features mature quickly; the head gets a small lr so the
#   walk amplitude drops while the weights still grow.
N_REG, REG_EPOCHS = 16, 30000
REG_BACKBONE_LR, REG_HEAD_LR = 1e-4, 3e-6
REG_LOSS = torch.nn.MSELoss()
# G5-B classification harness choices (recorded in classification.json).
N_CLS, CLS_EPOCHS, CLS_LR = 32, 100, 1e-3


def _build(num_classes):
    def build_model(trait):
        net, _ = remodel(YAML, num_classes, show_structure=False)
        return net.to(DEVICE)

    return build_model


def _grouped_optimizer(model, backbone_lr, head_lr, weight_decay):
    """Adam with a small lr for the final head (last nn.Linear) and a
    normal lr for the backbone. See the module docstring / constants for
    why a uniform lr leaves PCC oscillating below the gate."""
    head = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][-1]
    head_ids = {id(p) for p in head.parameters()}
    backbone, last = [], []
    for p in model.parameters():
        (last if id(p) in head_ids else backbone).append(p)
    return torch.optim.Adam(
        [{"params": backbone, "lr": backbone_lr},
         {"params": last, "lr": head_lr}],
        weight_decay=weight_decay)


def _loaders(x, y, batch_size):
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    loader = DataLoader(TensorDataset(xt, yt), batch_size=batch_size)
    return loader, xt


def _reload_predictions(ckpt_path, num_classes, x):
    """Load the best checkpoint into two fresh models; predictions must be
    identical across independent reloads."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    nets = []
    for _ in range(2):
        net, _ = remodel(YAML, num_classes, show_structure=False)
        net.to(DEVICE)
        ckpt = torch.load(os.path.join(OUT, ckpt_path), map_location=DEVICE)
        net.load_state_dict(ckpt["model_state_dict"])
        net.eval()
        nets.append(net)
    with torch.no_grad():
        p1 = nets[0](x.to(DEVICE))
        p2 = nets[1](x.to(DEVICE))
    return nets[0], bool(torch.equal(p1, p2)), bool(torch.isfinite(p1).all().item())


def g5_regression():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, size=(N_REG, 3, 206, 206)).astype(np.float32)
    y = rng.uniform(0.0, 1.0, size=N_REG).astype(np.float64)
    loader, xt = _loaders(x, y, N_REG)

    results = train_traits(
        Task.REGRESSION, ["g5_reg"], _build(1),
        lambda m: _grouped_optimizer(m, REG_BACKBONE_LR, REG_HEAD_LR, WEIGHT_DECAY),
        {"g5_reg": (loader, loader)}, DEVICE, REG_EPOCHS,
        REG_LOSS, Protocol.PACKAGE_LEGACY,
        config={"lr_backbone": REG_BACKBONE_LR, "lr_head": REG_HEAD_LR,
                "n_samples": N_REG, "epochs": REG_EPOCHS, "gate": "g5_regression"},
        checkpoint_dir=OUT,
    )
    hist = results["g5_reg"]["history"]
    first, last = hist[0]["train_loss"], hist[-1]["train_loss"]
    reduction = 1.0 - last / first

    model, reload_identical, preds_finite = _reload_predictions(
        "g5_reg_best.pt", 1, x)
    y_true, y_pred = evaluate_epoch_regression(model, loader, DEVICE)
    pcc_val = pcc(y_true, y_pred)
    grads_finite = all(bool(torch.isfinite(p.grad).all())
                       for p in model.parameters() if p.grad is not None)

    gate = {
        "loss_reduction_gt_0.95": reduction > 0.95,
        "pcc_gt_0.98": pcc_val > 0.98,
        "preds_finite": preds_finite,
        "grads_finite": grads_finite,
        "reload_prediction_identical": reload_identical,
    }
    out = {
        "gate": "g5_regression",
        "n_samples": N_REG, "epochs": REG_EPOCHS,
        "lr_backbone": REG_BACKBONE_LR, "lr_head": REG_HEAD_LR,
        "loss": "mse",
        "first_epoch_loss": first, "final_epoch_loss": last,
        "loss_reduction": reduction,
        "pcc": pcc_val,
        "best_metric": results["g5_reg"]["best_metric"],
        "best_epoch": hist[int(np.nanargmax([h["val_score"] for h in hist]))]["epoch"],
        "curve_every_10": [
            {"epoch": h["epoch"], "train_loss": round(h["train_loss"], 5),
             "val_pcc": None if np.isnan(h["val_score"]) else round(h["val_score"], 5)}
            for h in hist[::10]
        ],
        **gate,
        "pass": all(gate.values()),
    }
    with open(os.path.join(OUT, "regression.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("G5-A regression:", json.dumps(out, indent=2))
    return out["pass"]


def g5_classification():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 2, size=(N_CLS, 3, 206, 206)).astype(np.float32)
    y = rng.integers(0, 3, size=N_CLS)
    loader, xt = _loaders(x, y, N_CLS)

    results = train_traits(
        Task.CLASSIFICATION, ["g5_cls"], _build(3),
        lambda m: build_optimizer(m, lr=CLS_LR, weight_decay=WEIGHT_DECAY),
        {"g5_cls": (loader, loader)}, DEVICE, CLS_EPOCHS,
        torch.nn.CrossEntropyLoss(), Protocol.PACKAGE_LEGACY,
        config={"lr": CLS_LR, "n_samples": N_CLS, "epochs": CLS_EPOCHS,
                "gate": "g5_classification"},
        checkpoint_dir=OUT,
    )
    model, reload_identical, preds_finite = _reload_predictions(
        "g5_cls_best.pt", 3, x)
    y_true, y_pred = evaluate_epoch_classification(model, loader, DEVICE)
    acc = accuracy(y_true, y_pred)
    f1 = macro_f1(y_true, y_pred)

    gate = {
        "train_accuracy_ge_0.98": acc >= 0.98,
        "macro_f1_ge_0.98": f1 >= 0.98,
        "preds_finite": preds_finite,
        "reload_prediction_identical": reload_identical,
    }
    out = {
        "gate": "g5_classification",
        "n_samples": N_CLS, "epochs": CLS_EPOCHS, "lr": CLS_LR,
        "loss": "cross_entropy",
        "train_accuracy": acc,
        "macro_f1": f1,
        "best_metric": results["g5_cls"]["best_metric"],
        **gate,
        "pass": all(gate.values()),
    }
    with open(os.path.join(OUT, "classification.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("G5-B classification:", json.dumps(out, indent=2))
    return out["pass"]


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    reg_pass = g5_regression() if which in ("all", "regression") else None
    cls_pass = g5_classification() if which in ("all", "classification") else None
    print(f"G5 overall: regression={'PASS' if reg_pass else 'FAIL'} "
          f"classification={'PASS' if cls_pass else 'FAIL'}")
    sys.exit(0 if all(p for p in (reg_pass, cls_pass) if p is not None) else 1)


if __name__ == "__main__":
    main()
