"""G1 data-contract smoke test on the frozen upstream baseline (tag upstream-legacy).

Run from the repo root with the soydngp312 env python:
    /home/yhshy/miniconda3/envs/soydngp312/bin/python scripts/smoke_data_contract.py

Read-only with respect to upstream files; yaml outputs go to results/package_baseline/g1_smoke/.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from soydngpnext.reader_cpu import Reader_CPU, one_hot_CPU
from soydngpnext.data_process import DataProcess

DATA = os.path.join(ROOT, 'soydngpnext', 'data')
VCf = os.path.join(DATA, 'train_example.vcf')
CSV = os.path.join(DATA, 'train_example.csv')
OUT = os.path.join(ROOT, 'results', 'package_baseline', 'g1_smoke')
os.makedirs(OUT, exist_ok=True)

print('== 1. Reader_CPU.readVCF ==')
r = Reader_CPU()
df = r.readVCF(VCf)
print('shape (samples, snps):', df.shape)
print('dtypes:', df.dtypes.unique().tolist())
vals = df.to_numpy().ravel()
vc = {int(k): int(v) for k, v in zip(*np.unique(vals, return_counts=True))}
print('code counts {1=1|1/1/1, 2=0|1/0/1, 3=0|0/other/missing}:', vc)

# trace the 1|0 quirk: count raw phased-reversed genotypes in the file
raw = {}
with open(VCf) as f:
    for line in f:
        if line.startswith('#'):
            continue
        for v in line.split('\t')[9:]:
            k = v[:3]
            raw[k] = raw.get(k, 0) + 1
print('raw genotype values:', raw)
print('=> 1|0 cells (%d) are NOT in the replace rules, so they map to code 3 (same as missing)' % raw.get('1|0', 0))

print('== 2. one_hot_CPU shape / dtype / tiling ==')
subset = df.iloc[:4]
x = one_hot_CPU(subset.values)
print('output shape:', x.shape, 'dtype:', x.dtype)
# tiling check: with 2 SNPs the 6-value pattern [s0c0,s0c1,s0c2,s1c0,s1c1,s1c2]
# is repeated across the 42436-cell grid (np.resize wraps, does not zero-pad)
grid = x[0, 0, :, :].reshape(-1)
period = 206
print('grid row-major first 12 cells (ch0):', grid[:12].astype(int).tolist())
print('period-206 consistency:', bool(np.all(grid[:period] == grid[period:2 * period])))
print('=> grid is a tiled repetition of the 2-SNP one-hot row, NOT zero-padded')

print('== 3. DataProcess.convert_trait + to_dataset (protein) ==')
dp = DataProcess(VCf, CSV)
p_dict, n_dict = dp.convert_trait(OUT)
print('n_trait_dict:', n_dict)
print('p_trait_dict:', p_dict)
train_x, train_y, test_x, test_y = dp.to_dataset('protein', 0.7, is_quality=False)
print('train:', train_x.shape, train_x.dtype, '| labels:', train_y.shape, train_y.dtype,
      'range [%.3f, %.3f]' % (train_y.min(), train_y.max()))
print('test:', test_x.shape, test_x.dtype, '| labels:', test_y.shape, test_y.dtype)
print('label NaN after split:', int(np.isnan(train_y).sum()), int(np.isnan(test_y).sum()))

print('== 4. DataProcess.to_dataset (SCN3, stratified) ==')
train_q, train_lq, test_q, test_lq = dp.to_dataset('SCN3', 0.7, is_quality=True)
print('train:', train_q.shape, '| label counts:', pd.Series(train_lq).value_counts().to_dict())
print('test:', test_q.shape, '| label counts:', pd.Series(test_lq).value_counts().to_dict())

print('== summary ==')
print('samples:', df.shape[0], '| snps in VCF:', df.shape[1],
      '| grid cells needed: 42436 -> tiled x%d' % (42436 // (df.shape[1] * 3) if df.shape[1] * 3 else 0))
print('sample data is a smoke-test fixture only; real training needs a full-SNP VCF')
