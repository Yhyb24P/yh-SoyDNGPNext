"""Generate the tidy rfig source CSV for the G7 training-curve figure.

Reads the per-fold, per-epoch histories (mode="full" by default) that
g7_paper_baseline.py appends to results/trainer_validation/g7/g7_curves.jsonl
and writes a long-format CSV (one row per (epoch, fold)) carrying both
traits' metrics, so the rfig contract's two panels (protein PCC,
maturity-group macro-F1) can both read the same source.

Folds that were resumed from checkpoint (no per-epoch re-capture) leave
their metric cells empty; the rfig line plot simply omits those points.

Usage:
    python scripts/make_g7_rfig_csv.py [mode]   # mode: full (default) | curves
    -> writes results/trainer_validation/g7/rfig/g7_curves.csv
       and co-locates it at rfig/.rfig/figures/g7_curves.csv
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "trainer_validation", "g7")
CURVES = os.path.join(OUT, "g7_curves.jsonl")
TARGETS = (
    os.path.join(OUT, "rfig", "g7_curves.csv"),
    os.path.join(OUT, "rfig", ".rfig", "figures", "g7_curves.csv"),
)

# output column -> (trait, jsonl curve key)
COLS = [
    ("protein_pcc", "protein", "val_score"),
    ("protein_loss", "protein", "train_loss"),
    ("mg_f1", "maturity_group", "val_score"),
    ("mg_loss", "maturity_group", "train_loss"),
]


def load(mode):
    """Return {trait: {fold: {epoch: {key: value}}}} for the given mode."""
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
        for h in rec.get("curves", []):
            data.setdefault(rec["trait"], {}).setdefault(rec["fold"], {})[h["epoch"]] = h
    return data


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    data = load(mode)
    if not data:
        print(f"no {mode!r}-mode curve data in {CURVES}; nothing to write")
        return
    # Intersection of folds across all traits: a multi-panel single-source
    # figure can only show folds that have data in every panel (else some
    # panel has missing values and rfig fails closed).
    folds = sorted(set.intersection(*[set(d) for d in data.values()]))
    epochs = sorted(set(e for d in data.values() for f in d.values() for e in f))
    header = ["epoch", "fold"] + [c for c, _, _ in COLS]
    rows = []
    for ep in epochs:
        for fo in folds:
            row = {"epoch": ep, "fold": fo}
            for col, trait, key in COLS:
                h = data.get(trait, {}).get(fo, {}).get(ep)
                row[col] = h[key] if h and key in h else ""
            rows.append(row)
    for path in TARGETS:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=header)
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {len(rows)} rows ({len(folds)} folds x {len(epochs)} epochs) "
          f"to {len(TARGETS)} target(s), mode={mode}")


if __name__ == "__main__":
    main()
