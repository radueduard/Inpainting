import numpy as np

from hmi_infill.infill import DMVE, FSE, InfillInput, run


def _texture(n=128, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:n, :n]
    img = np.zeros((n, n))
    for _ in range(6):
        k1, k2 = rng.integers(1, 6, 2)
        img += rng.uniform(20, 100) * np.cos(2 * np.pi * (k1 * x + k2 * y) / 32 + rng.uniform(0, 6))
    return img


def test_fse_recovers_band_limited_texture():
    img = _texture()
    m = np.zeros(img.shape, bool)
    m[56:72, 10:118] = True                                   # 16-row strip
    out = run(FSE(), InfillInput(np.where(m, 0, img), m))
    assert np.sqrt(np.mean((out - img)[m] ** 2)) < 0.05 * img.std()


def test_fse_does_not_touch_known_pixels():
    img = _texture()
    m = np.zeros(img.shape, bool)
    m[20:28, :] = True
    out = run(FSE(), InfillInput(np.where(m, 0, img), m))
    assert np.array_equal(out[~m], img[~m])


def test_dmve_recovers_integer_motion():
    rng = np.random.default_rng(1)
    big = rng.normal(size=(160, 160)).cumsum(0).cumsum(1)     # rough texture
    cur = big[16:144, 16:144]
    past = big[16:144, 13:141]                                # scene moved by 3 px
    m = np.zeros(cur.shape, bool)
    m[40:56, 8:120] = True
    x = InfillInput(np.where(m, 0, cur), m, past=[past], past_masks=[np.zeros_like(m)])
    out = run(DMVE(search=5), x)
    assert np.allclose(out[m], cur[m])


def test_dmve_bidirectional_uses_both_sides():
    rng = np.random.default_rng(2)
    big = rng.normal(size=(160, 160)).cumsum(0).cumsum(1)
    cur, past, fut = big[16:144, 16:144], big[16:144, 14:142], big[16:144, 18:146]
    m = np.zeros(cur.shape, bool)
    m[60:68, 20:100] = True
    pm = np.zeros_like(m)
    x = InfillInput(np.where(m, 0, cur), m, past=[past], future=[fut], past_masks=[pm], future_masks=[pm])
    assert np.allclose(run(DMVE(search=4), x)[m], cur[m])
