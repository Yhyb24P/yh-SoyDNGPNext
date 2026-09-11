"""D1: reconstruct the paper's ordered 32,032 effective-SNP contract.

Joins the paper SNP list (data_manifest/snp_order.txt, 32,033 source lines)
against the raw SoyBase SoySNP50K gnm2 variant order
(data/raw/soybase_snp50k_gnm2/variant_order.tsv, 41,726 variants) on
(chromosome number, gnm2 position), preserving the paper's order.

Faithful to the author pipeline:
    pos_list = pd.read_csv("./predict/snp.txt")   # default header=0
    self.pos_list = pos_list.iloc[:, 0].to_list()
pd.read_csv treats line 1 of snp.txt as the column header, so the effective
list is lines[1:] -> 32,032 SNPs (the paper's claimed figure). The first file
line is the consumed header, not an effective SNP.

Join key: (chromosome number, gnm2 position).
    paper  "Chr02_13739700"             -> chrom 2, pos 13739700
    raw    "glyma.Wm82.gnm2.Gm02 ..."   -> chrom 2 (Gm02), same gnm2 position
The raw ss-ID (assay ID) is recorded for every matched row as a cross-check.

Outputs (under results/data_contract/d1/):
    d1_contract.tsv        ordered 32,032-row contract (paper order preserved)
    d1_missing.tsv         paper SNPs absent from the raw VCF (imputed './.')
    d1_raw_unselected.tsv  raw variants NOT selected by the paper (info)
    d1_summary.json        counts + gate + input SHA256
"""
import hashlib
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # upstream/
WORKSPACE = os.path.dirname(ROOT)  # SoyDNGPNext/

SNP_ORDER = os.path.join(ROOT, "data_manifest", "snp_order.txt")
RAW_VARIANT_ORDER = os.path.join(
    WORKSPACE, "data", "raw", "soybase_snp50k_gnm2", "variant_order.tsv")
OUT_DIR = os.path.join(ROOT, "results", "data_contract", "d1")

CHR_RE = re.compile(r"^Chr(\d+)$")
GM_RE = re.compile(r"Gm(\d+)$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_paper(line):
    # "Chr02_13739700" -> (2, 13739700)
    chrom, _, pos = line.rpartition("_")
    m = CHR_RE.match(chrom)
    if not m:
        raise ValueError("unexpected paper chrom label: %r" % line)
    return int(m.group(1)), int(pos)


def parse_raw(label, pos):
    # -> (key, chrom_number_or_None)
    m = GM_RE.search(label)
    cnum = int(m.group(1)) if m else None
    key = (cnum, int(pos)) if cnum is not None else ("scaffold", int(pos))
    return key, cnum


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- paper list ---
    with open(SNP_ORDER) as f:
        paper_lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    source_lines = len(paper_lines)
    effective_lines = paper_lines[1:]  # pd.read_csv header=0 drops line 1
    effective_count = len(effective_lines)

    paper_parsed = []
    for ln in effective_lines:
        chrom, pos = parse_paper(ln)
        paper_parsed.append((chrom, pos, ln))
    n_dup_paper = len(paper_parsed) - len({(c, p) for c, p, _ in paper_parsed})

    # --- raw variants ---
    raw_records = []   # (key, label, pos, ssid, ref, alt, cnum)
    raw_by_key = {}
    with open(RAW_VARIANT_ORDER) as f:
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            label, pos, ssid, ref, alt = p[0], p[1], p[2], p[3], p[4]
            key, cnum = parse_raw(label, pos)
            rec = (key, label, int(pos), ssid, ref, alt, cnum)
            raw_records.append(rec)
            raw_by_key[key] = rec
    raw_count = len(raw_records)
    n_dup_raw = raw_count - len(raw_by_key)

    # --- ordered join (paper order preserved) ---
    contract = []
    missing = []
    matched_keys = set()
    for idx, (chrom, pos, pline) in enumerate(paper_parsed):
        rec = raw_by_key.get((chrom, pos))
        if rec is not None:
            _, label, _, ssid, ref, alt, _ = rec
            contract.append([idx, pline, chrom, pos, "MATCHED", label, ssid, ref, alt])
            matched_keys.add((chrom, pos))
        else:
            contract.append([idx, pline, chrom, pos, "MISSING", "", "", "", ""])
            missing.append((idx, pline, chrom, pos))

    n_matched = sum(1 for row in contract if row[4] == "MATCHED")
    n_missing = len(missing)
    raw_unselected = [r for r in raw_records if r[0] not in matched_keys]

    # --- write outputs ---
    with open(os.path.join(OUT_DIR, "d1_contract.tsv"), "w") as f:
        f.write("idx\tpaper_id\tchrom\tpos\tstatus\traw_chrom\traw_ss_id\traw_ref\traw_alt\n")
        for row in contract:
            f.write("\t".join(str(x) for x in row) + "\n")

    with open(os.path.join(OUT_DIR, "d1_missing.tsv"), "w") as f:
        f.write("idx\tpaper_id\tchrom\tpos\n")
        for idx, pline, chrom, pos in missing:
            f.write("%d\t%s\t%d\t%d\n" % (idx, pline, chrom, pos))

    with open(os.path.join(OUT_DIR, "d1_raw_unselected.tsv"), "w") as f:
        f.write("raw_chrom\tpos\tss_id\tref\talt\n")
        for _, label, pos, ssid, ref, alt, _ in raw_unselected:
            f.write("%s\t%d\t%s\t%s\t%s\n" % (label, pos, ssid, ref, alt))

    gate = {
        "paper_snp_source_lines": source_lines,
        "paper_snp_effective": effective_count,
        "raw_vcf_variants": raw_count,
        "contract_rows": len(contract),
        "order_preserved": [row[0] for row in contract] == list(range(len(contract))),
        "duplicated_in_paper": n_dup_paper,
        "duplicated_in_raw_chrom_pos": n_dup_raw,
        "matched": n_matched,
        "missing": n_missing,
        "match_rate": round(n_matched / effective_count, 6) if effective_count else 0.0,
        "raw_unselected": len(raw_unselected),
    }
    gate["pass"] = (
        source_lines == 32033
        and effective_count == 32032
        and raw_count == 41726
        and len(contract) == 32032
        and gate["order_preserved"]
        and n_dup_paper == 0
    )

    summary = {
        "stage": "D1",
        "description": "Ordered 32,032 paper-SNP contract vs raw 41,726 SoySNP50K gnm2 variants",
        "inputs": {
            "snp_order": {
                "path": os.path.relpath(SNP_ORDER, WORKSPACE),
                "sha256": sha256(SNP_ORDER),
                "lines": source_lines,
            },
            "raw_variant_order": {
                "path": os.path.relpath(RAW_VARIANT_ORDER, WORKSPACE),
                "sha256": sha256(RAW_VARIANT_ORDER),
                "variants": raw_count,
            },
        },
        "effective_semantics": (
            "pd.read_csv('snp.txt') with default header=0 consumes line 1 as the "
            "column header; pos_list.iloc[:,0].to_list() therefore yields lines[1:] "
            "= 32,032 SNPs (the paper's claimed figure). Consumed header line: %s "
            "(not an effective SNP)." % paper_lines[0]
        ),
        "first_effective": effective_lines[0],
        "last_effective": effective_lines[-1],
        "join_key": "(chromosome number, gnm2 position); raw ss-ID recorded as cross-reference",
        "gate": gate,
    }
    with open(os.path.join(OUT_DIR, "d1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(json.dumps(gate, indent=2))
    print("outputs ->", OUT_DIR)


if __name__ == "__main__":
    main()
