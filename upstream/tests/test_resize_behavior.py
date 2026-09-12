"""Resize contract for one_hot_CPU (G1 findings, frozen as tests).

The upstream reader fills the 206x206x3 model input with np.resize, which:
    - N_SNP < 42436  -> repeats (wraps) the one-hot row to fill the grid
    - N_SNP > 42436  -> truncates the one-hot row to the first 42436 cells
    - N_SNP == 42436 -> exact fit, neither

The grid is NEVER zero-padded. All three behaviors are upstream legacy and
must stay reproducible.

Run:  python tests/test_resize_behavior.py   (or pytest)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from soydngpnext.reader_cpu import one_hot_CPU

CELLS = 206 * 206  # 42436
CHANNELS = 3


def _expected_from_codes(codes):
    """Analytic np.resize semantics matching reader_cpu.one_hot_CPU:
    one-hot the (M,) codes, C-flatten to (M*3,), wrap-or-truncate to
    CELLS*3, reshape (206,206,3), transpose to (3,206,206)."""
    m = len(codes)
    flat = np.empty(m * CHANNELS, dtype=np.float32)
    for c in range(CHANNELS):
        if c == 0:
            vals = np.isin(codes, [1, 2])
        elif c == 1:
            vals = codes != 2
        else:
            vals = codes != 1
        flat[c::CHANNELS] = vals.astype(np.float32)
    target = CELLS * CHANNELS
    if m * CHANNELS >= target:
        resized = flat[:target]
    else:
        resized = np.tile(flat, int(np.ceil(target / (m * CHANNELS))))[:target]
    return resized.reshape(206, 206, CHANNELS).transpose(2, 0, 1)


def test_resize_wraps_when_few_snps():
    """N_SNP < 42436: the one-hot row is repeated, not zero-padded."""
    codes = np.array([1, 2], dtype=np.int32)  # like the shipped 2-SNP train_example.vcf
    out = one_hot_CPU(codes.reshape(1, 2))
    assert out.shape == (1, 3, 206, 206)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out[0], _expected_from_codes(codes))
    # wrap proof: the pattern reaches the far corner (zero-padding would give 0)
    assert out[0, 2, 205, 205] == 1.0, 'last cell must be wrapped, not zero-padded'


def test_resize_truncates_when_many_snps():
    """N_SNP > 42436: only the first 42436 cells are used, the tail is dropped."""
    m = CELLS + 1  # 42437 SNPs
    codes = np.ones(m, dtype=np.int32)
    codes[-1] = 2  # the last SNP is the only code-2 -> its ch2 value (1) must vanish
    out = one_hot_CPU(codes.reshape(1, m))
    assert out.shape == (1, 3, 206, 206)
    np.testing.assert_array_equal(out[0], _expected_from_codes(codes))
    # truncation proof: code-2 contributes ch2=1; if the tail leaked in, ch2
    # would contain 1s. All source cells that survive are code-1 (ch2=0).
    assert out[0, 2].max() == 0.0, 'tail SNP must be truncated, not wrapped in'


def test_resize_exact_fit_is_identity():
    """N_SNP == 42436: the normal path for real data, no wrap, no truncate."""
    codes = np.full(CELLS, 1, dtype=np.int32)
    out = one_hot_CPU(codes.reshape(1, CELLS))
    assert out.shape == (1, 3, 206, 206)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out[0], _expected_from_codes(codes))
    np.testing.assert_array_equal(out[0], np.stack([
        np.ones((206, 206), np.float32),
        np.ones((206, 206), np.float32),
        np.zeros((206, 206), np.float32),
    ]))


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
        print('PASS', fn.__name__)
    print('all resize contract tests passed')
