"""G2.5: final CPU/GPU parity on the official full-SNP inference fixture.

One-shot check that the G2 parity contract (verified on the 13x3 adversarial
fixture) generalizes to the author's released inference VCF:
    ../data/10_test_examples.vcf  (10 samples x 42195 SNPs)

Contract (same as tests/test_gpu_cpu_parity.py, extended to the full VCF):
    genotype matrix   exact equality
    sample IDs        exact ordered equality
    SNP IDs           exact ordered equality
    one_hot           np.array_equal (CPU vs asnumpy(GPU))
    shape             (10, 3, 206, 206)
    dtype            float32

Run (needs GPU, soydngp312 env):
    /home/yhshy/miniconda3/envs/soydngp312/bin/python scripts/g2_full_parity.py [vcf]
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_VCF = os.path.normpath(os.path.join(ROOT, '..', 'data', '10_test_examples.vcf'))

import numpy as np
import cupy as cp

from soydngpnext.reader_cpu import Reader_CPU, one_hot_CPU
from soydngpnext.reader import Reader, one_hot


def main():
    vcf = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VCF
    print('VCF:', vcf)

    rc = Reader_CPU()
    dfc = rc.readVCF(vcf)
    rg = Reader()
    dfg = rg.readVCF(vcf)

    gc = dfc.to_numpy()
    gg = cp.asnumpy(dfg.values)
    print('genotype shape:', gc.shape, '| dtype:', gc.dtype, gg.dtype)
    assert gc.shape == gg.shape
    np.testing.assert_array_equal(gc, gg)
    print('PASS genotype matrix exact')

    sc = [str(v) for v in rc.indexes]
    sg = [str(v) for v in rg.indexes]
    assert sc == sg, 'sample ID order differs between CPU and GPU readers'
    print('PASS sample IDs exact ordered (%d)' % len(sc))

    cc = [str(v) for v in rc.columns]
    cg = [str(v) for v in rg.columns]
    assert cc == cg, 'SNP ID order differs between CPU and GPU readers'
    print('PASS SNP IDs exact ordered (%d)' % len(cc))

    oc = one_hot_CPU(dfc.values)
    og = one_hot(dfg.values)
    assert oc.dtype == np.float32, oc.dtype
    assert og.dtype == cp.float32, og.dtype
    assert oc.shape == og.shape == (gc.shape[0], 3, 206, 206), (oc.shape, og.shape)
    np.testing.assert_array_equal(oc, cp.asnumpy(og))
    print('PASS one_hot np.array_equal, shape', oc.shape, oc.dtype)

    print('G2.5 full-VCF parity PASSED -> G2 CLOSED')


if __name__ == '__main__':
    main()
