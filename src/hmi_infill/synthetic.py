"""Synthetic HMI-like images, for tests and for running the notebooks without data."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _random_field(shape, sigma, rng):
    f = ndimage.gaussian_filter(rng.standard_normal(shape), sigma)
    return f / f.std()


def sequence(n_frames: int = 6, size: int = 512, kind: str = "continuum",
             shift_px: float = 1.5, seed: int = 0) -> list[np.ndarray]:
    """Frames of a synthetic disk whose features drift by `shift_px` per frame.

    kind: "continuum" (limb darkening, granulation, spots, ~DN) or
          "magnetogram" (bipolar regions plus 10 G noise, Gauss).
    Off-disk pixels are NaN, like real HMI data.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((size, size)).astype(float)
    c, r = size / 2, 0.45 * size
    rho2 = ((xx - c) ** 2 + (yy - c) ** 2) / r ** 2
    disk = rho2 <= 1
    mu = np.sqrt(np.clip(1 - rho2, 0, 1))

    pad = int(np.ceil(shift_px * n_frames)) + 8
    big = (size, size + pad)
    gran = _random_field(big, 1.5, rng)
    meso = _random_field(big, 12, rng)
    spots = [(rng.uniform(0.3, 0.7) * size, rng.uniform(0.3, 0.6) * size, rng.uniform(6, 14) * size / 512)
             for _ in range(3)]

    frames = []
    for t in range(n_frames):
        dx = t * shift_px
        coords = [yy, xx + pad - dx - 4]
        g = ndimage.map_coordinates(gran, coords, order=3, mode="nearest")
        m = ndimage.map_coordinates(meso, coords, order=3, mode="nearest")
        if kind == "continuum":
            img = 50000 * (1 - 0.6 * (1 - mu)) * (1 + 0.03 * g + 0.01 * m)
            for (sy, sx, sr) in spots:
                d2 = (yy - sy) ** 2 + (xx - sx - dx) ** 2
                img *= 1 - 0.7 * np.exp(-d2 / (2 * sr ** 2)) - 0.2 * np.exp(-d2 / (2 * (2.2 * sr) ** 2))
            img += rng.normal(0, 30, img.shape)
        elif kind == "magnetogram":
            img = 15 * m * mu
            for k, (sy, sx, sr) in enumerate(spots):
                for sign, off in ((1, -2.5), (-1, 2.5)):
                    d2 = (yy - sy) ** 2 + (xx - sx - dx - off * sr) ** 2
                    img += sign * (-1) ** k * 2500 * mu * np.exp(-d2 / (2 * sr ** 2))
            img += rng.normal(0, 10, img.shape)
        else:
            raise ValueError(kind)
        frames.append(np.where(disk, img, np.nan))
    return frames
