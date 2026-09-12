"""Plot G7 per-epoch training curves -> static PNG.

Reads the per-fold, per-epoch histories that g7_paper_baseline.py appends to
results/trainer_validation/g7/g7_curves.jsonl, groups them by (trait, mode),
and draws one panel per trait:
    - left axis  : train loss vs epoch (log scale), one thin line per fold
                   + mean (bold) + std band
    - right axis : validation metric vs epoch (linear), PCC for protein,
                   macro-F1 for maturity_group; same per-fold + mean + band

Safe to run while the capture is still going: it just plots whatever folds are
already in the JSONL.

Usage:
    python scripts/plot_g7_curves.py [mode]      # mode: curves (default) | full
    -> writes results/trainer_validation/g7/g7_curves_<mode>.png
"""
import json
import os
import sys
import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "trainer_validation", "g7")
CURVES = os.path.join(OUT, "g7_curves.jsonl")
TRAITS = [("protein", "PCC", "regression"),
          ("maturity_group", "macro-F1", "classification")]


def load(mode):
    """Return {trait: [(fold, epochs[], train_loss[], val_score[]), ...]}."""
    data = {}
    if not os.path.exists(CURVES):
        return data
    for line in open(CURVES):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("mode") != mode:
            continue
        curves = rec.get("curves", [])
        if not curves:
            continue
        data.setdefault(rec["trait"], []).append((
            rec["fold"],
            [h["epoch"] for h in curves],
            [h["train_loss"] for h in curves],
            [h["val_score"] for h in curves],
        ))
    return data


def _stack(arrays):
    """Stack a list of equal-length lists into a 2-D array (rows = folds)."""
    return np.vstack([np.asarray(a, dtype=float) for a in arrays])


def plot(data, mode):
    present = [t for t, _, _ in TRAITS if t in data]
    if not present:
        print(f"no curve data for mode={mode!r} in {CURVES}; nothing to plot")
        return None
    fig, axes = plt.subplots(len(present), 1, figsize=(9, 4.6 * len(present)),
                             sharex=False)
    if len(present) == 1:
        axes = [axes]
    for ax, trait in zip(axes, present):
        metric, task = next((m, tk) for t, m, tk in TRAITS if t == trait)
        folds = data[trait]
        ax2 = ax.twinx()
        for fold, epochs, tl, vs in folds:
            ax.plot(epochs, tl, color="tab:blue", alpha=0.20, lw=1.0)
            ax2.plot(epochs, vs, color="tab:red", alpha=0.20, lw=1.0)
        tl_all = _stack([tl for _, _, tl, _ in folds])
        vs_all = _stack([vs for _, _, _, vs in folds])
        ref = folds[0][1]
        tl_m, tl_s = tl_all.mean(0), tl_all.std(0)
        vs_m, vs_s = vs_all.mean(0), vs_all.std(0)
        ax.plot(ref, tl_m, color="tab:blue", lw=2.5, label="train loss (mean)")
        ax.fill_between(ref, np.clip(tl_m - tl_s, 1e-4, None), tl_m + tl_s,
                       color="tab:blue", alpha=0.15)
        ax2.plot(ref, vs_m, color="tab:red", lw=2.5,
                 label=f"{metric} (mean)")
        ax2.fill_between(ref, np.clip(vs_m - vs_s, 0, None), vs_m + vs_s,
                        color="tab:red", alpha=0.15)
        ax.set_yscale("log")
        ax.set_ylabel("train loss (log)", color="tab:blue")
        ax2.set_ylabel(metric, color="tab:red")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(f"{trait} ({task}) — {len(folds)} fold(s), "
                     f"mean {metric} @last = {vs_m[-1]:.3f}")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
    axes[-1].set_xlabel("epoch")
    fig.suptitle(f"G7 per-epoch curves ({mode} mode) — "
                 f"{datetime.date.today().isoformat()}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = os.path.join(OUT, f"g7_curves_{mode}.png")
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")
    return png


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "curves"
    plot(load(mode), mode)


if __name__ == "__main__":
    main()
