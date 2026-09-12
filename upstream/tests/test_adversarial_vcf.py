"""Adversarial VCF contract test for the upstream legacy reader (G1 baseline).

The fixture (tests/fixtures/adversarial.vcf) exercises every GT spelling the
reader can meet:
    0/0  0|0  0/1  0|1  1/0  1|0  1/1  1|1  ./.  .|.  and GT:DP-suffixed fields

Expected codes follow the UPSTREAM LEGACY CONTRACT (reader_cpu.py / reader.py):
    '1/1' or '1|1' -> 1
    '0/1' or '0|1' -> 2
    everything else (0/0, 0|0, 1/0, 1|0, ./. , .|., ...) -> 3

In particular `1|0` maps to 3 (same bucket as missing). That is the frozen
upstream behavior: this suite locks it in as a regression test. Do NOT fix it
here. If 1|0 is ever reclassified as heterozygous, that must be a separate
labelled experiment, not a silent change to the paper-reproduction baseline.

Run:  python tests/test_adversarial_vcf.py   (or pytest)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from soydngpnext.reader_cpu import Reader_CPU

FIXTURE = os.path.join(os.path.dirname(__file__), 'fixtures', 'adversarial.vcf')

SAMPLES = ['S_00', 'S_00p', 'S_01', 'S_01p', 'S_10', 'S_10p', 'S_11',
           'S_11p', 'S_MISS', 'S_MISSP', 'S_DP11', 'S_DP01', 'S_DPMISS']
SNPS = ['Chr01_100', 'Chr02_200', 'Chr03_300']
# per-sample legacy code, identical across the 3 SNP rows of the fixture
EXPECTED = {
    'S_00': 3, 'S_00p': 3, 'S_01': 2, 'S_01p': 2,
    'S_10': 3, 'S_10p': 3, 'S_11': 1, 'S_11p': 1,
    'S_MISS': 3, 'S_MISSP': 3, 'S_DP11': 1, 'S_DP01': 2, 'S_DPMISS': 3,
}
EXPECTED_MATRIX = np.array([[EXPECTED[s]] * 3 for s in SAMPLES], dtype=np.int32)


def _read():
    r = Reader_CPU()
    df = r.readVCF(FIXTURE)
    return r, df


def test_sample_and_snp_ids_exact_order():
    r, df = _read()
    assert list(df.index) == SAMPLES, 'sample IDs must keep VCF column order'
    assert list(df.columns) == SNPS, 'SNP IDs must keep VCF row order (CHROM_POS)'
    assert list(r.indexes) == SAMPLES
    assert list(r.columns) == SNPS


def test_genotype_matrix_exact_values():
    _, df = _read()
    assert df.shape == (13, 3)
    assert df.dtypes.unique().tolist() == [np.dtype('int32')]
    np.testing.assert_array_equal(df.to_numpy(), EXPECTED_MATRIX)


def test_one_zero_phased_maps_to_three_legacy_contract():
    """REGRESSION: 1|0 -> code 3 (upstream legacy contract, frozen).

    1|0 is not in the reader's replace rules ({1/1, 1|1, 0/1, 0|1}), so it
    falls through to code 3, the same bucket as 0/0 and missing. This test
    locks that behavior. Changing it is a separate experiment, not a fix.
    """
    _, df = _read()
    np.testing.assert_array_equal(
        df.loc['S_10p'].to_numpy(), np.full(3, 3, dtype=np.int32))


def test_gt_with_field_suffixes_slices_to_first_three_chars():
    _, df = _read()
    np.testing.assert_array_equal(df.loc['S_DP11'].to_numpy(), np.full(3, 1, np.int32))
    np.testing.assert_array_equal(df.loc['S_DP01'].to_numpy(), np.full(3, 2, np.int32))
    np.testing.assert_array_equal(df.loc['S_DPMISS'].to_numpy(), np.full(3, 3, np.int32))


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
        print('PASS', fn.__name__)
    print('all adversarial contract tests passed')
