"""Data-contract check for any VCF (and optional trait CSV) against the
upstream SoyDNGPNext reader pipeline.

Usage:
    python scripts/check_data_contract.py VCF [TRAIT_CSV]

Checks (all read-only):
    1. VCF structure: header line, SNP rows, sample columns
    2. raw GT value distribution (first 3 chars), 1|0 count, annotated GTs
    3. if CSV given: sample-ID mapping both ways, per-trait missing counts
    4. Reader_CPU.readVCF: shape, dtype, legacy code distribution
    5. N_SNP vs 42436 grid: wrap (x-repeat) / exact fit / truncate
    6. one_hot on the first 8 samples: shape, dtype, far-corner cell

Run with the soydngp312 env python.
"""
import collections
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from soydngpnext.reader_cpu import Reader_CPU, one_hot_CPU

CELLS = 206 * 206  # 42436


def raw_scan(vcf_path):
    """Cheap structural scan without pandas: header index, SNP rows, samples,
    raw GT value counts (first 3 chars)."""
    raw = collections.Counter()
    annotated = 0
    n_snp = 0
    samples = None
    header_idx = None
    with open(vcf_path) as f:
        for i, line in enumerate(f):
            if line.startswith('##'):
                continue
            if line.startswith('#CHROM'):
                samples = line.rstrip('\n').split('\t')[9:]
                header_idx = i
                continue
            n_snp += 1
            for v in line.rstrip('\n').split('\t')[9:]:
                raw[v[:3]] += 1
                if len(v) > 3:
                    annotated += 1
    return header_idx, n_snp, samples, raw, annotated


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    vcf = sys.argv[1]
    csv = sys.argv[2] if len(sys.argv) > 2 else None

    print('== 1. VCF structure ==')
    header_idx, n_snp, samples, raw, annotated = raw_scan(vcf)
    print('header line index:', header_idx)
    print('SNP rows:', n_snp, '| sample columns:', len(samples))
    print('raw GT values:', dict(raw))
    print('annotated GT cells (len>3, sliced to 3 chars):', annotated)
    print('1|0 cells (legacy: map to code 3):', raw.get('1|0', 0))

    if csv:
        print('== 2. trait CSV mapping ==')
        dfc = pd.read_csv(csv, encoding='gbk')
        ids = list(dfc.iloc[:, 0])
        s_csv, s_vcf = set(ids), set(samples)
        print('csv rows:', len(ids), '| unique:', len(s_csv), '| columns:', list(dfc.columns))
        print('csv IDs in VCF: %d/%d' % (len(s_csv & s_vcf), len(s_csv)))
        print('VCF samples in csv: %d/%d' % (len(s_csv & s_vcf), len(s_vcf)))
        for col in dfc.columns[1:]:
            miss = int(dfc[col].isna().sum()) + int((dfc[col].astype(str).str.strip() == '').sum())
            print('trait %-12s missing: %d' % (col, miss))

    print('== 3. Reader_CPU.readVCF ==')
    r = Reader_CPU()
    df = r.readVCF(vcf)
    vals = df.to_numpy().ravel()
    vc = {int(k): int(v) for k, v in zip(*np.unique(vals, return_counts=True))}
    print('shape (samples, snps):', df.shape, '| dtype:', df.dtypes.unique().tolist())
    print('legacy code counts {1=1|1/1/1, 2=0|1/0/1, 3=rest}:', vc)

    print('== 4. grid sizing (42436 cells) ==')
    m = n_snp
    if m < CELLS:
        print('N_SNP=%d < 42436 -> WRAP: one-hot row repeated x%d to fill grid'
              % (m, int(np.ceil(CELLS * 3 / (m * 3)))))
    elif m == CELLS:
        print('N_SNP=%d == 42436 -> exact fit' % m)
    else:
        print('N_SNP=%d > 42436 -> TRUNCATE: only first 42436 cells used, %d dropped'
              % (m, m - CELLS))

    print('== 5. one_hot on first 8 samples ==')
    x = one_hot_CPU(df.iloc[:8].to_numpy())
    print('shape:', x.shape, '| dtype:', x.dtype)
    print('far-corner cells (sample0): ch0=%.0f ch1=%.0f ch2=%.0f'
          % (x[0, 0, 205, 205], x[0, 1, 205, 205], x[0, 2, 205, 205]))
    print('sample data usable for training only if N_SNP == 42436 or the '
          'wrap/truncate behavior is accepted as-is')


if __name__ == '__main__':
    main()
