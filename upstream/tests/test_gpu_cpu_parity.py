"""G2 parity contract: CPU reader/one_hot vs GPU (cudf/cupy) reader/one_hot.

Frozen requirements:
    CPU genotype   == GPU genotype              exact equality
    CPU SNP IDs    == GPU SNP IDs              exact ordered equality
    CPU sample IDs == GPU sample IDs           exact ordered equality
    CPU one_hot    == asnumpy(GPU one_hot)     np.array_equal
    dtype          == float32
    shape          == (N, 3, 206, 206)

Run:  python tests/test_gpu_cpu_parity.py   (or pytest; needs the GPU)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import cupy as cp

from soydngpnext.reader_cpu import Reader_CPU, one_hot_CPU
from soydngpnext.reader import Reader, one_hot

FIXTURE = os.path.join(os.path.dirname(__file__), 'fixtures', 'adversarial.vcf')


def _host_ids(x):
    return [str(v) for v in x]


def test_gpu_cpu_parity_on_adversarial_fixture():
    rc = Reader_CPU()
    dfc = rc.readVCF(FIXTURE)
    rg = Reader()
    dfg = rg.readVCF(FIXTURE)

    # genotypes: exact equality
    gc = dfc.to_numpy()
    gg = cp.asnumpy(dfg.values)
    assert gc.shape == gg.shape == (13, 3)
    np.testing.assert_array_equal(gc, gg)

    # SNP IDs: exact ordered equality
    assert _host_ids(rc.columns) == _host_ids(rg.columns), \
        'SNP ID order must match between CPU and GPU readers'
    # sample IDs: exact ordered equality
    assert _host_ids(rc.indexes) == _host_ids(rg.indexes), \
        'sample ID order must match between CPU and GPU readers'

    # one_hot parity: CPU vs asnumpy(GPU), dtype and shape pinned
    oc = one_hot_CPU(dfc.values)
    og = one_hot(dfg.values)
    assert oc.dtype == np.float32
    assert og.dtype == cp.float32
    assert oc.shape == og.shape == (13, 3, 206, 206)
    np.testing.assert_array_equal(oc, cp.asnumpy(og))


if __name__ == '__main__':
    test_gpu_cpu_parity_on_adversarial_fixture()
    print('PASS test_gpu_cpu_parity_on_adversarial_fixture')
    print('G2 parity contract passed')
