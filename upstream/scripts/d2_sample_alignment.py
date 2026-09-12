"""D2: sample x phenotype alignment (VCF 20,087 vs phenotype 20,506).

Builds a unique mapping table with buckets:
    matched_exact    in both, identical string
    ambiguous        differ only by numeric-field zero-padding (candidate, NOT
                     accepted) -- e.g. VCF "FC1547" <-> pheno "FC001547",
                     VCF "PI19986" <-> pheno "PI019986"
    vcf_only         genotyped, no phenotype even after normalization
    phenotype_only   phenotyped, no genotype even after normalization
    duplicate        (0 on both sources)

Principle: exact string match is the ground truth. The zero-padding rule is a
documented candidate rule, surfaced in `ambiguous` for a human decision -- it
is never silently promoted to matched.

Also reports the per-trait effective cohort (non-missing value counts) on the
exact-matched cohort, so D3 can size each trait's usable cohort instead of a
blanket dropna.

Outputs (under results/data_contract/d2/):
    d2_mapping.tsv           one row per unique sample in the union
    d2_ambiguous_pairs.tsv   the zero-pad candidate pairs
    d2_vcf_only.tsv          residual vcf_only
    d2_phenotype_only.tsv    residual phenotype_only
    d2_per_trait_cohort.tsv  non-missing counts per trait (exact cohort)
    d2_summary.json          bucket counts + rule + input SHA256 + decision note
"""
import csv
import hashlib
import json
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # upstream/
WORKSPACE = os.path.dirname(ROOT)  # SoyDNGPNext/

VCF_SAMPLES = os.path.join(WORKSPACE, "data", "raw", "soybase_snp50k_gnm2", "sample_ids.txt")
PHENO_SAMPLES = os.path.join(ROOT, "data_manifest", "samples.tsv")
PHENO_TABLE = os.path.join(ROOT, "data_manifest", "phenotype_table.csv")
OUT_DIR = os.path.join(ROOT, "results", "data_contract", "d2")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_lines(path, skip=0):
    with open(path) as f:
        return [ln.rstrip("\n") for i, ln in enumerate(f) if i >= skip]


def norm_key(x):
    # (alpha prefix upper, int of digit field or None, trailing suffix)
    m = re.match(r"^([A-Za-z]+)(\d*)(.*)$", x)
    return (m.group(1).upper(), int(m.group(2)) if m.group(2) else None, m.group(3))


def read_phenotype_table(path):
    for enc in ("utf-8-sig", "gbk"):
        try:
            with open(path, encoding=enc, newline="") as f:
                rows = list(csv.reader(f))
            return rows, enc
        except UnicodeDecodeError:
            continue
    raise RuntimeError("could not decode phenotype table")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    vcf = load_lines(VCF_SAMPLES)
    pheno = load_lines(PHENO_SAMPLES, skip=1)  # skip 'acid' header
    sv, sp = set(vcf), set(pheno)
    vcf_dup = len(vcf) - len(sv)
    pheno_dup = len(pheno) - len(sp)

    vcf_by = defaultdict(list)
    for x in sv:
        vcf_by[norm_key(x)].append(x)
    pheno_by = defaultdict(list)
    for x in sp:
        pheno_by[norm_key(x)].append(x)

    mapping = []
    amb_pairs = set()
    vcf_only = []
    pheno_only = []
    n_exact = 0
    for sid in sorted(sv | sp):
        in_v, in_p = sid in sv, sid in sp
        if in_v and in_p:
            mapping.append([sid, "Y", "Y", sid, "exact", "matched_exact"])
            n_exact += 1
        elif in_v:  # vcf only
            partners = pheno_by.get(norm_key(sid), [])
            if partners:
                pid = partners[0] if len(partners) == 1 else ";".join(sorted(partners))
                mapping.append([sid, "Y", "N", pid, "zero_pad_candidate", "ambiguous"])
                amb_pairs.add((sid, pid, "zero_pad"))
            else:
                mapping.append([sid, "Y", "N", "", "none", "vcf_only"])
                vcf_only.append(sid)
        else:  # phenotype only
            partners = vcf_by.get(norm_key(sid), [])
            if partners:
                pid = partners[0] if len(partners) == 1 else ";".join(sorted(partners))
                mapping.append([sid, "N", "Y", pid, "zero_pad_candidate", "ambiguous"])
                amb_pairs.add((pid, sid, "zero_pad"))
            else:
                mapping.append([sid, "N", "Y", "", "none", "phenotype_only"])
                pheno_only.append(sid)
    amb_sorted = sorted(amb_pairs)

    # --- per-trait effective cohort on the EXACT-matched samples ---
    rows, enc = read_phenotype_table(PHENO_TABLE)
    header = rows[0]
    trait_cols = [c for c in header if c not in ("acid", "CommonName")]
    pheno_row_by_acid = {}
    for r in rows[1:]:
        if r:
            pheno_row_by_acid[r[0]] = r
    col_idx = {c: i for i, c in enumerate(header)}
    exact_matched = sorted(sv & sp)
    per_trait = []
    for c in trait_cols:
        i = col_idx[c]
        nonmiss = 0
        for sid in exact_matched:
            r = pheno_row_by_acid.get(sid)
            if r is not None and i < len(r) and r[i].strip() != "":
                nonmiss += 1
        per_trait.append([c, len(exact_matched), nonmiss, len(exact_matched) - nonmiss])

    # row-level vs per-trait missingness: count samples with SOME but not all
    # traits filled (partial). 0 => missingness is row-level (all-or-none).
    partial_missing = 0
    for sid in exact_matched:
        r = pheno_row_by_acid.get(sid)
        filled = 0
        for c in trait_cols:
            i = col_idx[c]
            if r is not None and i < len(r) and r[i].strip() != "":
                filled += 1
        if 0 < filled < len(trait_cols):
            partial_missing += 1
    shared_cohort = per_trait[0][2] if per_trait else 0

    # --- write outputs ---
    with open(os.path.join(OUT_DIR, "d2_mapping.tsv"), "w") as f:
        f.write("sample_id\tin_vcf\tin_phenotype\tmatched_id\tmatch_rule\tbucket\n")
        for row in mapping:
            f.write("\t".join(row) + "\n")
    with open(os.path.join(OUT_DIR, "d2_ambiguous_pairs.tsv"), "w") as f:
        f.write("vcf_id\tpheno_id\trule\n")
        for a, b, r in amb_sorted:
            f.write("%s\t%s\t%s\n" % (a, b, r))
    with open(os.path.join(OUT_DIR, "d2_vcf_only.tsv"), "w") as f:
        f.write("sample_id\n")
        for sid in sorted(vcf_only):
            f.write(sid + "\n")
    with open(os.path.join(OUT_DIR, "d2_phenotype_only.tsv"), "w") as f:
        f.write("sample_id\n")
        for sid in sorted(pheno_only):
            f.write(sid + "\n")
    with open(os.path.join(OUT_DIR, "d2_per_trait_cohort.tsv"), "w") as f:
        f.write("trait\tcohort_size\tnon_missing\tmissing\n")
        for c, tot, nm, ms in per_trait:
            f.write("%s\t%d\t%d\t%d\n" % (c, tot, nm, ms))

    n_amb_pairs = len(amb_sorted)
    summary = {
        "stage": "D2",
        "description": "Sample x phenotype alignment: VCF 20,087 vs phenotype 20,506",
        "inputs": {
            "vcf_samples": {"path": os.path.relpath(VCF_SAMPLES, WORKSPACE),
                            "sha256": sha256(VCF_SAMPLES), "count": len(vcf)},
            "pheno_samples": {"path": os.path.relpath(PHENO_SAMPLES, WORKSPACE),
                              "sha256": sha256(PHENO_SAMPLES), "count": len(pheno)},
            "phenotype_table": {"path": os.path.relpath(PHENO_TABLE, WORKSPACE),
                                "sha256": sha256(PHENO_TABLE), "encoding": enc},
        },
        "buckets": {
            "vcf_total": len(vcf),
            "pheno_total": len(pheno),
            "duplicate_vcf": vcf_dup,
            "duplicate_pheno": pheno_dup,
            "matched_exact": n_exact,
            "ambiguous_zero_pad_pairs": n_amb_pairs,
            "vcf_only_residual": len(vcf_only),
            "phenotype_only_residual": len(pheno_only),
        },
        "rule": {
            "exact": "identical string in both sources (ground truth)",
            "zero_pad_candidate": "same alpha prefix + same integer digit value + same "
                                  "trailing suffix; differ only by leading zeros in the "
                                  "digit field. SURFACED, NOT accepted.",
        },
        "decision_note": (
            "If the %d zero-pad candidate pairs are accepted, the matched cohort "
            "grows from %d to %d. Rejecting them keeps the strict exact-match "
            "cohort of %d. Per-trait cohorts below are computed on the exact "
            "cohort only." % (n_amb_pairs, n_exact, n_exact + n_amb_pairs, n_exact)
        ),
        "per_trait_missingness": {
            "row_level": partial_missing == 0,
            "partial_missing_samples": partial_missing,
            "shared_non_missing_cohort": shared_cohort,
            "note": (
                "Missingness is row-level: a sample has all %d traits or none "
                "(%d partial-missing samples). All traits therefore share the "
                "same non-missing cohort of %d on the exact-matched set, so D3 "
                "needs one cohort, not a per-trait dropna."
                % (len(trait_cols), partial_missing, shared_cohort)
            ),
        },
        "per_trait_cohort_on_exact": per_trait,
    }
    with open(os.path.join(OUT_DIR, "d2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(json.dumps(summary["buckets"], indent=2))
    print("ambiguous zero-pad pairs:", n_amb_pairs)
    print("decision:", summary["decision_note"])
    print("outputs ->", OUT_DIR)


if __name__ == "__main__":
    main()
