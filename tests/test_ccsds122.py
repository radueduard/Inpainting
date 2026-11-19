import numpy as np
import pytest

from hmi_infill.ccsds122 import BLOCK_LOST, BLOCK_OK, CodecParams, decode, dwt, encode
from hmi_infill.ccsds122.blocks import from_blocks, to_blocks


@pytest.fixture
def image():
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[:128, :160]
    return (20000 * np.exp(-((xx - 80) ** 2 + (yy - 64) ** 2) / 3000)
            + rng.normal(0, 25, yy.shape)).astype(np.int64)


@pytest.mark.parametrize("shape", [(16, 16), (24, 40), (64, 96)])
def test_dwt_perfect_reconstruction(shape):
    x = np.random.default_rng(0).integers(-2**20, 2**20, shape)
    assert np.array_equal(dwt.inverse(dwt.forward(x)), x)


def test_dwt_support_matches_perturbation():
    rng = np.random.default_rng(2)
    c = dwt.forward(rng.integers(-1000, 1000, (64, 64)))
    flags = np.zeros(c.shape, bool)
    flags[3, 5] = flags[40, 9] = flags[20, 50] = True
    base = dwt.inverse(c)
    changed = np.zeros(c.shape, bool)
    for _ in range(5):
        c2 = c.copy()
        c2[flags] += rng.integers(1000, 5000, flags.sum())
        changed |= dwt.inverse(c2) != base
    sup = dwt.support(flags)
    assert not np.any(changed & ~sup)       # support covers every change
    assert changed.sum() > 0.8 * sup.sum()  # and is tight


def test_blocks_roundtrip():
    c = np.arange(64 * 48).reshape(64, 48)
    assert np.array_equal(from_blocks(to_blocks(c), c.shape), c)


def test_lossless_roundtrip(image):
    enc = encode(image, CodecParams(segment_blocks=32))
    res = decode(enc.segments, enc.geometry)
    assert np.array_equal(res.image, image)
    assert np.all(res.block_status == BLOCK_OK)
    assert not res.mask_lost.any() and not res.mask_uncertain.any()


def test_non_multiple_of_8_shape():
    img = np.random.default_rng(3).integers(0, 4000, (37, 51))
    enc = encode(img, CodecParams(segment_blocks=16))
    assert np.array_equal(decode(enc.segments, enc.geometry).image, img)


@pytest.mark.parametrize("frac", [0.02, 0.1, 0.3, 0.6, 0.95])
def test_truncation_bounds_contain_truth(image, frac):
    enc = encode(image, CodecParams(segment_blocks=32))
    res = decode([s[: int(len(s) * frac)] for s in enc.segments], enc.geometry)
    padded = np.pad(image, ((0, (-image.shape[0]) % 8), (0, (-image.shape[1]) % 8)), mode="edge")
    c = dwt.forward(padded)
    assert np.all((c >= res.coeff_lo) & (c <= res.coeff_hi))
    assert np.all((res.coeff_est >= res.coeff_lo) & (res.coeff_est <= res.coeff_hi))


def test_truncation_error_decreases(image):
    enc = encode(image, CodecParams(segment_blocks=32))
    errs = []
    for frac in (0.1, 0.3, 0.6, 1.0):
        r = decode([s[: int(np.ceil(len(s) * frac))] for s in enc.segments], enc.geometry)
        errs.append(np.sqrt(np.mean((r.image - image) ** 2)))
    assert all(a >= b for a, b in zip(errs, errs[1:])) and errs[-1] == 0


def test_lost_segment_is_localised(image):
    enc = encode(image, CodecParams(segment_blocks=32))
    segs = list(enc.segments)
    segs[4] = None
    res = decode(segs, enc.geometry)
    assert (res.block_status == BLOCK_LOST).sum() == 32
    assert np.array_equal(res.image[~res.mask_lost], image[~res.mask_lost])
