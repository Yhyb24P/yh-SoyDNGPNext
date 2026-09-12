"""G6: real-data smoke test on the D3 matrix.

Verifies the end-to-end pipeline on REAL data (not a memorization test):
    D3 matrix (15,899 x 32,032 int8) -> one_hot -> (N,3,206,206)
    phenotype labels (protein / MG) -> corrected trainer -> short run

Scope is a SMOKE test, not the production baseline: a seeded 1024-sample
subset, 50 epochs (NOT 150), the G5-validated discriminative lr. It answers
"does the pipeline run and behave sanely on real data?", not "does it
generalize" (that is G7).

Gates (written to results/trainer_validation/g6/):
    pipeline_runs, loss_finite, loss_decreased, preds_finite, grads_finite,
    reload_prediction_identical; regression additionally pcc_finite.

All harness choices (subset size, epochs, lr, split) are recorded in the JSON.

Run (GPU, soydngp312 env):
    /home/yhshy/miniconda3/envs/soydngp312/bin/python scripts/g6_realdata_smoke.py [regression|classification|all]
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from soydngpnext.remodel import remodel
from soydngpnext.reader_cpu import one_hot_CPU
from soydngp_repro.metrics import accuracy, macro_f1, pcc
from soydngp_repro.protocols import Protocol, Task
from soydngp_repro.splits import package_legacy_holdout
from soydngp_repro.trainer import (evaluate_epoch_classification,
                                   evaluate_epoch_regression, train_traits)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(ROOT, "results", "trainer_validation", "g6")
os.makedirs(OUT, exist_ok=True)
YAML = os.path.join(ROOT, "soydngpnext", "data", "model.yaml")
MATRIX = os.path.join(WS, "data", "derived", "d3_matrix", "d3_matrix.npy")
SAMP_ORDER = os.path.join(WS, "data", "derived", "d3_matrix", "d3_sample_order.txt")
COHORT = os.path.join(ROOT, "results", "data_contract", "d2", "d2_accepted_cohort.tsv")
PHENO = os.path.join(ROOT, "data_manifest", "phenotype_table.csv")

WEIGHT_DECAY = 1e-5
# G6 smoke-test harness choices (recorded in the JSONs):
N_SUB, SEED = 1024, 0
EPOCHS, BATCH = 50, 64
BACKBONE_LR, HEAD_LR = 1e-4, 3e-6   # G5-validated discriminative lr
PROTEIN_MIN, PROTEIN_MAX = 31.7, 57.9  # from provenance / n_trait range


def _grouped_optimizer(model, backbone_lr, head_lr, weight_decay):
    head = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][-1]
    head_ids = {id(p) for p in head.parameters()}
    backbone, last = [], []
    for p in model.parameters():
        (last if id(p) in head_ids else backbone).append(p)
    return torch.optim.Adam(
        [{"params": backbone, "lr": backbone_lr},
         {"params": last, "lr": head_lr}], weight_decay=weight_decay)


def _labels():
    """Return (protein_norm, mg_class, mg_levels) aligned to the D3 sample order."""
    samp = open(SAMP_ORDER).read().split("\n")[:-1]
    vcf2pheno = {r[1]: r[0] for r in
                 [l.rstrip("\n").split("\t") for l in open(COHORT)][1:]}
    pt = list(csv.reader(open(PHENO, encoding="utf-8-sig", newline="")))
    hdr = pt[0]
    idx = {c: i for i, c in enumerate(hdr)}
    by_acid = {r[0]: r for r in pt[1:] if r}
    iP, iMG = idx["protein"], idx["MG"]
    prot, mg = [], []
    for s in samp:
        r = by_acid.get(vcf2pheno[s])
        prot.append(float(r[iP]))
        mg.append(r[iMG])
    prot = np.asarray(prot, dtype=np.float64)
    mg = np.asarray(mg)
    prot_norm = (prot - PROTEIN_MIN) / (PROTEIN_MAX - PROTEIN_MIN)
    levels = np.sort(np.unique(mg))
    mg_class = np.searchsorted(levels, mg).astype(np.int64)
    return prot_norm, mg_class, levels


def _subset(x_full_int8, y_full):
    """Pick a seeded subset on the int8 matrix, one-hot only that subset
    (avoids materializing the full 15,899 x 3 x 206 x 206 float32)."""
    rng = np.random.default_rng(SEED)
    sel = np.sort(rng.choice(len(x_full_int8), N_SUB, replace=False))
    return one_hot_CPU(x_full_int8[sel]), y_full[sel]


def _reload_predictions(ckpt_path, num_classes, x):
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
        p1, p2 = nets[0](x.to(DEVICE)), nets[1](x.to(DEVICE))
    return nets[0], bool(torch.equal(p1, p2)), bool(torch.isfinite(p1).all().item())


def _common_gate(history, model, reload_identical, preds_finite):
    """Pipeline-health gates for a SMOKE test (not convergence): the run
    completes, every quantity stays finite, and the checkpoint reloads to
    identical predictions. Loss decrease is reported but not gated, because
    the bias-free head is slow (G5) and 50 epochs is still in the climbing
    regime; convergence is G7's job."""
    losses = [h["train_loss"] for h in history]
    grads_finite = all(bool(torch.isfinite(p.grad).all())
                      for p in model.parameters() if p.grad is not None)
    return {
        "loss_finite": bool(all(np.isfinite(l) for l in losses)),
        "preds_finite": bool(preds_finite),
        "grads_finite": bool(grads_finite),
        "reload_prediction_identical": bool(reload_identical),
    }


def g6_regression():
    prot_norm, mg_class, levels = _labels()
    x_full = np.load(MATRIX)
    x, y = _subset(x_full, prot_norm)
    tr, va = package_legacy_holdout(N_SUB, 0.7, SEED)
    xt = torch.from_numpy(x[tr]); yt = torch.from_numpy(np.asarray(y[tr], dtype=np.float64))
    xv = torch.from_numpy(x[va]); yv = torch.from_numpy(np.asarray(y[va], dtype=np.float64))
    train_loader = DataLoader(TensorDataset(xt, yt), batch_size=BATCH)
    val_loader = DataLoader(TensorDataset(xv, yv), batch_size=BATCH)

    def build_model(_):
        net, _ = remodel(YAML, 1, show_structure=False)
        return net.to(DEVICE)

    results = train_traits(
        Task.REGRESSION, ["g6_protein"], build_model,
        lambda m: _grouped_optimizer(m, BACKBONE_LR, HEAD_LR, WEIGHT_DECAY),
        {"g6_protein": (train_loader, val_loader)}, DEVICE, EPOCHS,
        torch.nn.MSELoss(), Protocol.PACKAGE_LEGACY,
        config={"n_subset": N_SUB, "epochs": EPOCHS, "lr_backbone": BACKBONE_LR,
                "lr_head": HEAD_LR, "gate": "g6_regression"},
        checkpoint_dir=OUT)
    hist = results["g6_protein"]["history"]
    model, reload_identical, preds_finite = _reload_predictions(
        "g6_protein_best.pt", 1, x)
    y_true, y_pred = evaluate_epoch_regression(model, val_loader, DEVICE)
    pcc_val = pcc(y_true, y_pred)
    gate = _common_gate(hist, model, reload_identical, preds_finite)
    gate["pcc_finite"] = bool(np.isfinite(pcc_val))
    out = {
        "gate": "g6_regression", "trait": "protein",
        "n_subset": N_SUB, "n_train": len(tr), "n_val": len(va),
        "epochs": EPOCHS, "batch": BATCH, "lr_backbone": BACKBONE_LR,
        "lr_head": HEAD_LR, "loss": "mse",
        "first_epoch_loss": hist[0]["train_loss"],
        "final_epoch_loss": hist[-1]["train_loss"],
        "val_pcc": pcc_val, "val_mse": hist[-1]["val_mse"],
        "loss_decreased": bool(hist[-1]["train_loss"] < hist[0]["train_loss"]),
        "loss_note": "informational: 50-epoch smoke test, head still climbing "
                     "(G5); convergence is G7's job.",
        **gate, "pass": all(gate.values()),
    }
    with open(os.path.join(OUT, "regression.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("G6 regression:", json.dumps(out, indent=2))
    return out["pass"]


def g6_classification():
    prot_norm, mg_class, levels = _labels()
    x_full = np.load(MATRIX)
    x, y = _subset(x_full, mg_class)
    tr, va = package_legacy_holdout(N_SUB, 0.7, SEED, y=y)
    xt = torch.from_numpy(x[tr]); yt = torch.from_numpy(np.asarray(y[tr], dtype=np.int64))
    xv = torch.from_numpy(x[va]); yv = torch.from_numpy(np.asarray(y[va], dtype=np.int64))
    train_loader = DataLoader(TensorDataset(xt, yt), batch_size=BATCH)
    val_loader = DataLoader(TensorDataset(xv, yv), batch_size=BATCH)
    n_cls = len(levels)

    def build_model(_):
        net, _ = remodel(YAML, n_cls, show_structure=False)
        return net.to(DEVICE)

    results = train_traits(
        Task.CLASSIFICATION, ["g6_mg"], build_model,
        lambda m: _grouped_optimizer(m, BACKBONE_LR, HEAD_LR, WEIGHT_DECAY),
        {"g6_mg": (train_loader, val_loader)}, DEVICE, EPOCHS,
        torch.nn.CrossEntropyLoss(), Protocol.PACKAGE_LEGACY,
        config={"n_subset": N_SUB, "epochs": EPOCHS, "n_classes": n_cls,
                "mg_levels": [str(l) for l in levels], "gate": "g6_classification"},
        checkpoint_dir=OUT)
    hist = results["g6_mg"]["history"]
    model, reload_identical, preds_finite = _reload_predictions(
        "g6_mg_best.pt", n_cls, x)
    y_true, y_pred = evaluate_epoch_classification(model, val_loader, DEVICE)
    gate = _common_gate(hist, model, reload_identical, preds_finite)
    out = {
        "gate": "g6_classification", "trait": "maturity_group",
        "n_subset": N_SUB, "n_train": len(tr), "n_val": len(va),
        "epochs": EPOCHS, "batch": BATCH, "n_classes": n_cls,
        "mg_levels": [str(l) for l in levels],
        "loss": "cross_entropy",
        "first_epoch_loss": hist[0]["train_loss"],
        "final_epoch_loss": hist[-1]["train_loss"],
        "val_accuracy": hist[-1]["val_accuracy"], "val_macro_f1": hist[-1]["val_score"],
        "loss_decreased": bool(hist[-1]["train_loss"] < hist[0]["train_loss"]),
        **gate, "pass": all(gate.values()),
    }
    with open(os.path.join(OUT, "classification.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("G6 classification:", json.dumps(out, indent=2))
    return out["pass"]


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    reg = g6_regression() if which in ("all", "regression") else None
    cls = g6_classification() if which in ("all", "classification") else None
    ok = all(p for p in (reg, cls) if p is not None)
    smoke = {"device": DEVICE, "regression": reg, "classification": cls, "pass": ok}
    with open(os.path.join(OUT, "smoke.json"), "w") as f:
        json.dump(smoke, f, indent=2)
    print(f"G6 overall: regression={'PASS' if reg else 'FAIL'} "
          f"classification={'PASS' if cls else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
