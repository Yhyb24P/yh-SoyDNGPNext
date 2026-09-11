"""D3: build the paper input matrix + verify historical preprocessing parity.

Assembles the (samples x 32,032) genotype matrix in the paper's SNP order:
  - 31,673 matched SNPs: GT read from the reduced VCF, legacy-encoded
    (1/1,1|1 -> 1; 0/1,0|1 -> 2; everything else incl './.' -> 3).
  - 359 missing SNPs: filled with code 3 ('./.'), author default no-Beagle.
Samples = the accepted D2 cohort restricted to non-missing phenotype (15,899).

The reduced VCF (15,899 x 31,673) is produced by bcftools from the raw
SoySNP50K gnm2 VCF using data/derived/d3_regions.txt (-R) and
data/derived/d3_samples.txt (-S). This script reads it, encodes, assembles,
and checks that the author's one_hot + 206x206 resize (wrap, since 32,032 <
42,436) behaves as documented.

Outputs:
    data/derived/d3_matrix/d3_matrix.npy        (15,899 x 32,032) int8
    data/derived/d3_matrix/d3_snp_order.txt     32,032 paper SNP ids (order)
    data/derived/d3_matrix/d3_sample_order.txt  15,899 sample ids (order)
    results/data_contract/d3/d3_manifest.json   provenance + counts + parity
"""
import gzip
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # upstream/
WORKSPACE = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

REDUCED = os.path.join(WORKSPACE, "data", "derived", "d3_reduced.vcf.gz")
D1_CONTRACT = os.path.join(ROOT, "results", "data_contract", "d1", "d1_contract.tsv")
OUT_M = os.path.join(WORKSPACE, "data", "derived", "d3_matrix")
OUT_R = os.path.join(ROOT, "results", "data_contract", "d3")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    import numpy as np

    os.makedirs(OUT_M, exist_ok=True)
    os.makedirs(OUT_R, exist_ok=True)

    # --- D1 contract: slot order + matched (raw_chrom,pos) -> slot ---
    rows = [l.rstrip("\n").split("\t") for l in open(D1_CONTRACT)][1:]
    n_slot = len(rows)
    assert n_slot == 32032, n_slot
    snp_order = [r[1] for r in rows]
    slot_of = {}
    n_matched = n_missing = 0
    for i, r in enumerate(rows):
        if r[4] == "MATCHED":
            slot_of[(r[5], r[3])] = i
            n_matched += 1
        else:
            n_missing += 1
    assert n_matched + n_missing == n_slot

    # --- stream the reduced VCF, encode, fill the matrix ---
    # bcftools 1.3 may write the .vcf.gz uncompressed; detect the magic bytes.
    with open(REDUCED, "rb") as _probe:
        is_gz = _probe.read(2) == b"\x1f\x8b"
    opener = gzip.open if is_gz else open
    mat = None
    sample_order = None
    n_filled = 0
    raw_gt = {"0/0": 0, "1/1": 0, "0/1": 0, "./.": 0, "other": 0}
    with opener(REDUCED, "rt") as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                sample_order = line.rstrip("\n").split("\t")[9:]
                mat = np.full((n_slot, len(sample_order)), 3, dtype=np.int8)
                continue
            fld = line.rstrip("\n").split("\t")
            slot = slot_of.get((fld[0], fld[1]))
            if slot is None:
                continue  # reduced VCF should only hold matched variants
            g3 = np.array([g[:3] for g in fld[9:]])
            codes = np.full(len(g3), 3, dtype=np.int8)
            codes[(g3 == "1/1") | (g3 == "1|1")] = 1
            codes[(g3 == "0/1") | (g3 == "0|1")] = 2
            mat[slot] = codes
            n_filled += 1
            raw_gt["0/0"] += int(np.sum(g3 == "0/0"))
            raw_gt["1/1"] += int(np.sum(g3 == "1/1"))
            raw_gt["0/1"] += int(np.sum((g3 == "0/1") | (g3 == "0|1")))
            raw_gt["./."] += int(np.sum(g3 == "./."))
            raw_gt["other"] += int(np.sum(~np.isin(g3, ["0/0", "1/1", "0/1", "0|1", "./."])))

    assert sample_order is not None and mat is not None
    n_sample = len(sample_order)
    assert n_filled == n_matched, (n_filled, n_matched)
    X = mat.T  # (samples, snps) = (15,899, 32,032)

    # --- save ---
    np.save(os.path.join(OUT_M, "d3_matrix.npy"), X)
    with open(os.path.join(OUT_M, "d3_snp_order.txt"), "w") as f:
        f.write("\n".join(snp_order) + "\n")
    with open(os.path.join(OUT_M, "d3_sample_order.txt"), "w") as f:
        f.write("\n".join(sample_order) + "\n")

    # --- historical preprocessing parity (one_hot + 206x206 wrap) ---
    from soydngpnext.reader_cpu import one_hot_CPU
    probe = int(X.min()), int(X.max())
    oh = one_hot_CPU(X[:8].astype(np.int32))
    parity = {
        "one_hot_shape": list(oh.shape),
        "one_hot_dtype": str(oh.dtype),
        "expected_shape": [8, 3, 206, 206],
        "grid_cells": 206 * 206,
        "n_snp": n_slot,
        "wrap": n_slot < 206 * 206,
        "far_corner_sample0": [float(oh[0, c, 205, 205]) for c in range(3)],
        "code_range": probe,
        "code_counts": {int(k): int(v) for k, v in
                        zip(*np.unique(X, return_counts=True))},
    }
    parity["pass"] = (oh.shape == tuple(parity["expected_shape"])
                      and parity["wrap"] and probe == (1, 3))

    manifest = {
        "stage": "D3",
        "description": "Paper input matrix (accepted cohort x 32,032 ordered SNPs)",
        "inputs": {
            "reduced_vcf": {"path": os.path.relpath(REDUCED, WORKSPACE),
                            "sha256": sha256(REDUCED),
                            "samples": n_sample, "variants": n_matched},
            "d1_contract": {"path": os.path.relpath(D1_CONTRACT, WORKSPACE),
                            "sha256": sha256(D1_CONTRACT)},
        },
        "counts": {
            "samples": n_sample,
            "snps_total": n_slot,
            "snps_matched": n_matched,
            "snps_missing_filled_code3": n_missing,
            "filled_from_vcf": n_filled,
        },
        "encoding": "legacy frozen: 1/1,1|1 -> 1; 0/1,0|1 -> 2; else (incl './.') -> 3",
        "raw_gt_breakdown": {
            "note": "3-char GT prefix counts over the 31,673 matched SNPs x 15,899 "
                    "samples in the reduced VCF. Only 0/0, 1/1, 0/1 and './.' occur "
                    "(no 1/0, no | forms), so legacy code2 == raw 0/1 count. The "
                    "legacy rule maps 1/0 -> 3 (alt-first het not recognized as 2).",
            **raw_gt,
        },
        "missing_fill": "359 absent SNPs -> code 3 (author default, no Beagle)",
        "grid": "32,032 < 42,436 -> one-hot row wrapped (tiled) to fill 206x206",
        "outputs": {
            "matrix": os.path.relpath(os.path.join(OUT_M, "d3_matrix.npy"), WORKSPACE),
            "snp_order": os.path.relpath(os.path.join(OUT_M, "d3_snp_order.txt"), WORKSPACE),
            "sample_order": os.path.relpath(os.path.join(OUT_M, "d3_sample_order.txt"), WORKSPACE),
        },
        "preprocessing_parity": parity,
    }
    with open(os.path.join(OUT_R, "d3_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(json.dumps(manifest["counts"], indent=2))
    print("parity:", json.dumps(parity, indent=2))
    print("matrix ->", os.path.join(OUT_M, "d3_matrix.npy"))


if __name__ == "__main__":
    main()
