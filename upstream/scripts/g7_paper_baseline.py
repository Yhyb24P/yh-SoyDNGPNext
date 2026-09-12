"""G7: paper-faithful baseline (10-fold CV x 150 epochs).

The production baseline on the full D3 matrix (15,899 x 32,032 int8):
    D3 matrix -> one_hot -> (N,3,206,206)
    10-fold CV (splits.paper_repro_folds, KFold shuffle=True, seed)
    per fold: train 150 epochs on 9/10, evaluate on the held-out 1/10
    aggregate: mean +/- std of the per-fold metric

Two pilot traits (user-approved scope):
    protein        -> regression  (target normalized (p-31.7)/26.2)
    maturity_group -> classification (10 levels, searchsorted index)

Model + optimizer: PAPER_MODEL_V1 (remodel), G5-validated discriminative lr
(backbone 1e-4 / head 3e-6), weight decay 1e-5. Protocol label PAPER_REPRO.

Per-fold metric: the held-out fold metric at the FINAL (150th) epoch
(faithful to the fixed 150-epoch schedule); the best-epoch metric is also
recorded for reference. No per-fold checkpoint is written (metrics only).

Modes (argv):
    g7_paper_baseline.py [quick|full] [regression|classification|all]
    quick: 2 folds x 10 epochs  (harness validation + per-epoch timing)
    full:  10 folds x 150 epochs (production baseline)

Run (GPU, soydngp312 env):
    /home/yhshy/miniconda3/envs/soydngp312/bin/python scripts/g7_paper_baseline.py quick all
"""
import csv
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from soydngpnext.remodel import remodel
from soydngpnext.reader_cpu import one_hot_CPU
from soydngp_repro.protocols import Protocol, Task
from soydngp_repro.splits import paper_repro_folds
from soydngp_repro.trainer import train_traits

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(ROOT, "results", "trainer_validation", "g7")
os.makedirs(OUT, exist_ok=True)
PROGRESS = os.path.join(OUT, "g7_progress.jsonl")
CURVES = os.path.join(OUT, "g7_curves.jsonl")
YAML = os.path.join(ROOT, "soydngpnext", "data", "model.yaml")
MATRIX = os.path.join(WS, "data", "derived", "d3_matrix", "d3_matrix.npy")
SAMP_ORDER = os.path.join(WS, "data", "derived", "d3_matrix", "d3_sample_order.txt")
COHORT = os.path.join(ROOT, "results", "data_contract", "d2", "d2_accepted_cohort.tsv")
PHENO = os.path.join(ROOT, "data_manifest", "phenotype_table.csv")

WEIGHT_DECAY = 1e-5
BATCH = 64
BACKBONE_LR, HEAD_LR = 1e-4, 3e-6   # G5-validated discriminative lr
PROTEIN_MIN, PROTEIN_MAX = 31.7, 57.9
SEED = 0
FULL_FOLDS, FULL_EPOCHS = 10, 150
QUICK_FOLDS, QUICK_EPOCHS = 2, 10
CURVE_FOLDS = 3   # "curves" mode: enough folds to show fold-to-fold variance


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


def _load_done(trait, mode):
    """Completed (trait, mode, fold) records from the progress file, so a
    restarted run skips folds that already finished (machine-suspend-safe)."""
    done = {}
    if os.path.exists(PROGRESS):
        for line in open(PROGRESS):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("trait") == trait and rec.get("mode") == mode:
                done[rec["fold"]] = rec
    return done


def _append_progress(rec):
    with open(PROGRESS, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _append_curves(trait, mode, fold, n_epochs, hist):
    """Persist the full per-epoch history of a fold (for plotting curves).
    Written once at fold end (not per-epoch) to avoid I/O stalling training."""
    rec = {"trait": trait, "mode": mode, "fold": fold, "n_epochs": n_epochs,
           "curves": [dict(h) for h in hist]}
    with open(CURVES, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()


def _run_trait(task, trait, x_full, y, num_classes, n_folds, epochs, out_json, mode):
    """Drive the 10-fold CV for one trait; write per-fold + aggregate JSON.

    Each completed fold is appended to PROGRESS (g7_progress.jsonl); a
    restarted run skips folds already recorded there."""
    folds = paper_repro_folds(len(x_full), n_folds, SEED)
    loss_fn = (torch.nn.MSELoss() if task is Task.REGRESSION
               else torch.nn.CrossEntropyLoss())
    ydtype = np.float64 if task is Task.REGRESSION else np.int64
    done = _load_done(trait, mode)
    per_fold = []
    t0 = time.time()
    for k, (tr, te) in enumerate(folds):
        if k in done:
            rec = done[k]
            per_fold.append({kk: vv for kk, vv in rec.items()
                             if kk not in ("trait", "mode")})
            print(f"[{trait}] fold {k+1}/{n_folds}: SKIPPED "
                  f"(resumed, PCC/macroF1={rec['final_epoch_metric']:.4f})",
                  flush=True)
            continue
        # Move this fold's data to the GPU ONCE: the trainer's per-batch
        # .to(device) then becomes a no-op and the DataLoader's per-batch
        # gathers happen on-GPU instead of the slow 7GB CPU fancy-index.
        xt = torch.from_numpy(x_full[tr]).to(DEVICE)
        yt = torch.from_numpy(np.asarray(y[tr], dtype=ydtype)).to(DEVICE)
        xv = torch.from_numpy(x_full[te]).to(DEVICE)
        yv = torch.from_numpy(np.asarray(y[te], dtype=ydtype)).to(DEVICE)
        train_loader = DataLoader(TensorDataset(xt, yt), batch_size=BATCH, shuffle=True)
        val_loader = DataLoader(TensorDataset(xv, yv), batch_size=BATCH)

        def build_model(_):
            net, _ = remodel(YAML, num_classes, show_structure=False)
            return net.to(DEVICE)

        fk = time.time()
        results = train_traits(
            task, [trait], build_model,
            lambda m: _grouped_optimizer(m, BACKBONE_LR, HEAD_LR, WEIGHT_DECAY),
            {trait: (train_loader, val_loader)}, DEVICE, epochs, loss_fn,
            Protocol.PAPER_REPRO,
            config={"trait": trait, "fold": k, "n_folds": n_folds,
                    "epochs": epochs, "seed": SEED, "batch": BATCH,
                    "lr_backbone": BACKBONE_LR, "lr_head": HEAD_LR,
                    "n_train": len(tr), "n_test": len(te),
                    "metric": "pcc" if task is Task.REGRESSION else "macro_f1"},
            checkpoint_dir=None)
        hist = results[trait]["history"]
        final_metric = hist[-1]["val_score"]          # held-out metric @ epoch `epochs`
        best_metric = results[trait]["best_metric"]    # best held-out metric across epochs
        rec = {
            "trait": trait, "mode": mode, "fold": k,
            "n_train": len(tr), "n_test": len(te),
            "final_epoch_metric": final_metric, "best_epoch_metric": best_metric,
            "first_epoch_loss": hist[0]["train_loss"],
            "final_epoch_loss": hist[-1]["train_loss"],
            "fold_seconds": round(time.time() - fk, 1),
        }
        per_fold.append({kk: vv for kk, vv in rec.items() if kk not in ("trait", "mode")})
        _append_progress(rec)   # checkpoint BEFORE releasing the fold (suspend-safe)
        _append_curves(trait, mode, k, epochs, hist)   # per-epoch curves for plotting
        print(f"[{trait}] fold {k+1}/{n_folds}: "
              f"{'PCC' if task is Task.REGRESSION else 'macroF1'} "
              f"@{epochs}ep={final_metric:.4f} best={best_metric:.4f} "
              f"({rec['fold_seconds']}s)", flush=True)
        # Release this fold's GPU tensors + model and shrink the caching pool,
        # so the pool does not accumulate across the 20 folds (OOM risk).
        del xt, xv, yt, yv, train_loader, val_loader, results
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    metric_key = "pcc" if task is Task.REGRESSION else "macro_f1"
    finals = [f["final_epoch_metric"] for f in per_fold]
    bests = [f["best_epoch_metric"] for f in per_fold]
    out = {
        "gate": f"g7_{trait}", "trait": trait,
        "task": task.value, "n_samples": len(x_full),
        "n_folds": n_folds, "epochs": epochs, "seed": SEED, "batch": BATCH,
        "lr_backbone": BACKBONE_LR, "lr_head": HEAD_LR,
        "loss": "mse" if task is Task.REGRESSION else "cross_entropy",
        "metric": metric_key,
        "per_fold": per_fold,
        "final_metric_mean": float(np.mean(finals)),
        "final_metric_std": float(np.std(finals)),
        "best_metric_mean": float(np.mean(bests)),
        "best_metric_std": float(np.std(bests)),
        "total_seconds": round(time.time() - t0, 1),
        "note": "per-fold metric is the held-out fold metric at the final "
                f"({epochs}) epoch (fixed schedule, paper-faithful); "
                "best_epoch_metric is the max over epochs (reference only).",
    }
    with open(os.path.join(OUT, out_json), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{trait}] {metric_key} @final: "
          f"{out['final_metric_mean']:.4f} +/- {out['final_metric_std']:.4f} "
          f"(best-epoch {out['best_metric_mean']:.4f} +/- {out['best_metric_std']:.4f}); "
          f"total {out['total_seconds']}s", flush=True)
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    if mode == "quick":
        n_folds, epochs = QUICK_FOLDS, QUICK_EPOCHS
    elif mode == "curves":
        n_folds, epochs = CURVE_FOLDS, FULL_EPOCHS   # curve capture: 3 folds x 150ep
    else:
        n_folds, epochs = FULL_FOLDS, FULL_EPOCHS
    tag = f"g7_{mode}"

    prot_norm, mg_class, levels = _labels()
    print(f"one_hot-ing full D3 matrix ({len(prot_norm)} samples)...", flush=True)
    x_full = one_hot_CPU(np.load(MATRIX))
    print(f"one_hot done: {x_full.shape} {x_full.dtype}", flush=True)

    results = {}
    if which in ("all", "regression"):
        results["regression"] = _run_trait(
            Task.REGRESSION, "protein", x_full, prot_norm, 1,
            n_folds, epochs, f"regression_{mode}.json", mode)
    if which in ("all", "classification"):
        results["classification"] = _run_trait(
            Task.CLASSIFICATION, "maturity_group", x_full, mg_class, len(levels),
            n_folds, epochs, f"classification_{mode}.json", mode)
        results["classification"]["mg_levels"] = [str(l) for l in levels]
        with open(os.path.join(OUT, f"classification_{mode}.json"), "w") as f:
            json.dump(results["classification"], f, indent=2)

    summary = {"mode": mode, "n_folds": n_folds, "epochs": epochs,
               "device": DEVICE, "traits": results}
    with open(os.path.join(OUT, f"{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{tag} done. " + ", ".join(
        f"{k}: {v['final_metric_mean']:.4f}+/-{v['final_metric_std']:.4f} "
        f"{v['metric']}" for k, v in results.items()))


if __name__ == "__main__":
    main()
