"""Corrected trainer (G4).

Fixes over upstream soydngpnext/train.py:
  T1 regression targets: float32, reshaped to (B, 1); prediction shape is
     asserted equal, so no implicit broadcasting can slip in.
  T2 classification targets: int64, reshaped to (B,); separate epoch loop,
     logits asserted 2-D with matching batch size.
  T3 metrics: computed once after the full validation epoch, never
     per-batch or averaged over cumulative batches.
  T4 splits: come from protocols.Protocol, physically separated.
  checkpoints: dict with model/optimizer state, epoch, best_metric,
     protocol, config. Selection by max validation PCC (regression) or
     max validation macro-F1 (classification); no "mean of four metrics".
  multi-trait: model, optimizer, counters, best, history are created
     inside the per-trait loop.

The trainer accepts plain nn.Modules. Abnormal interfaces (the verbatim
paper model) go through adapters.unwrap_model / adapters.PaperSoyDNGP.
"""
import os

import numpy as np
import torch

from soydngp_repro import metrics
from soydngp_repro.protocols import Protocol, Task


def build_optimizer(model, lr=1e-3, weight_decay=1e-5):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


# ------------------------------------------------------------- T1 / T2

def train_epoch_regression(model, loader, loss_fn, optimizer, device):
    """One training epoch for regression (T1 contract).

    target is cast to float32 and reshaped to (B, 1); the prediction shape
    is asserted equal to the target shape.
    """
    model.train()
    total, n = 0.0, 0
    for x, target in loader:
        x = x.to(device, dtype=torch.float32)
        target = target.to(device, dtype=torch.float32).reshape(-1, 1)
        pred = model(x)
        assert pred.shape == target.shape, \
            f"pred {tuple(pred.shape)} != target {tuple(target.shape)}"
        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


def train_epoch_classification(model, loader, loss_fn, optimizer, device):
    """One training epoch for classification (T2 contract).

    target is cast to int64 and reshaped to (B,); logits must be 2-D with
    a matching batch size.
    """
    model.train()
    total, n = 0.0, 0
    for x, target in loader:
        x = x.to(device, dtype=torch.float32)
        target = target.to(device, dtype=torch.long).reshape(-1)
        logits = model(x)
        assert logits.ndim == 2, f"logits ndim {logits.ndim}"
        assert logits.shape[0] == target.shape[0], \
            f"logits {tuple(logits.shape)} vs target {tuple(target.shape)}"
        loss = loss_fn(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


# ------------------------------------------------------- T3 evaluation

@torch.no_grad()
def evaluate_epoch_regression(model, loader, device):
    """Collect (y_true, y_pred) over the FULL epoch (T3 contract)."""
    model.eval()
    truths, preds = [], []
    for x, target in loader:
        x = x.to(device, dtype=torch.float32)
        target = target.to(device, dtype=torch.float32).reshape(-1, 1)
        pred = model(x)
        assert pred.shape == target.shape
        truths.append(target.cpu().numpy().ravel())
        preds.append(pred.cpu().numpy().ravel())
    return np.concatenate(truths), np.concatenate(preds)


@torch.no_grad()
def evaluate_epoch_classification(model, loader, device):
    """Collect (y_true, y_pred_argmax) over the FULL epoch (T3 contract)."""
    model.eval()
    truths, preds = [], []
    for x, target in loader:
        x = x.to(device, dtype=torch.float32)
        target = target.to(device, dtype=torch.long).reshape(-1)
        logits = model(x)
        assert logits.ndim == 2
        truths.append(target.cpu().numpy())
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(truths), np.concatenate(preds)


# --------------------------------------------------------- checkpoints

def save_checkpoint(path, model, optimizer, epoch, best_metric, protocol, config):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "protocol": protocol.value if isinstance(protocol, Protocol) else protocol,
        "config": config,
    }, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# -------------------------------------------------------- trait driver

def train_trait(task, trait, model, optimizer, train_loader, val_loader, device,
                epochs, loss_fn, protocol, config, checkpoint_dir=None):
    """Train one trait with a fresh model+optimizer.

    Selection: regression -> max validation PCC; classification -> max
    validation macro-F1. Metrics are computed once per full epoch (T3).
    Returns (history, best_metric).
    """
    best_metric = -np.inf
    history = []
    for epoch in range(1, epochs + 1):
        if task is Task.REGRESSION:
            train_loss = train_epoch_regression(model, train_loader, loss_fn, optimizer, device)
            y_true, y_pred = evaluate_epoch_regression(model, val_loader, device)
            score = metrics.pcc(y_true, y_pred)
            extra = {"val_mse": metrics.mse(y_true, y_pred),
                     "val_smooth_l1": metrics.smooth_l1(y_true, y_pred)}
        else:
            train_loss = train_epoch_classification(model, train_loader, loss_fn, optimizer, device)
            y_true, y_pred = evaluate_epoch_classification(model, val_loader, device)
            score = metrics.macro_f1(y_true, y_pred)
            extra = {"val_accuracy": metrics.accuracy(y_true, y_pred)}
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_score": score, **extra})
        if not np.isnan(score) and score > best_metric:
            best_metric = score
            if checkpoint_dir is not None:
                os.makedirs(checkpoint_dir, exist_ok=True)
                save_checkpoint(os.path.join(checkpoint_dir, f"{trait}_best.pt"),
                                model, optimizer, epoch, best_metric, protocol, config)
    return history, best_metric


def train_traits(task, traits, build_model, build_optimizer, loaders, device,
                 epochs, loss_fn, protocol, config, checkpoint_dir=None):
    """Driver: one fresh model+optimizer+counters+best+history per trait.

    loaders: dict trait -> (train_loader, val_loader).
    Returns dict trait -> {"model", "optimizer", "history", "best_metric"}.
    """
    results = {}
    for trait in traits:
        model = build_model(trait)
        optimizer = build_optimizer(model)
        train_loader, val_loader = loaders[trait]
        history, best = train_trait(task, trait, model, optimizer, train_loader, val_loader,
                                     device, epochs, loss_fn, protocol, config, checkpoint_dir)
        results[trait] = {"model": model, "optimizer": optimizer,
                          "history": history, "best_metric": best}
    return results
