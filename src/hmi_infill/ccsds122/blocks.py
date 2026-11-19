"""Grouping of DWT coefficients into CCSDS 122 blocks (Section 4.1).

After a 3-level DWT, each block corresponds to an 8x8 pixel area and holds 64
coefficients. Families: 0 = HL, 1 = LH, 2 = HH. The index layout used
throughout this package is:

    0                      DC (LL3)
    1 + i                  parent of family i              (level 3)
    4 + 4 i + j            child j of family i             (level 2, 2x2 raster)
    16 + 16 i + 4 j + k    grandchild k of child j, fam. i (level 1, 2x2 raster)

Blocks are numbered in raster order over the (H/8, W/8) block grid.
"""

from __future__ import annotations

import numpy as np

DC = 0
PARENTS = np.arange(1, 4)
CHILDREN = np.arange(4, 16)
GRANDCHILDREN = np.arange(16, 64)


def _subbands(c: np.ndarray):
    """Return views of the 10 subbands of a 3-level Mallat layout."""
    h, w = c.shape
    bands = {}
    for lev in (1, 2, 3):
        hh, ww = h >> lev, w >> lev
        bands[("HL", lev)] = c[:hh, ww:2 * ww]
        bands[("LH", lev)] = c[hh:2 * hh, :ww]
        bands[("HH", lev)] = c[hh:2 * hh, ww:2 * ww]
    bands[("LL", 3)] = c[: h >> 3, : w >> 3]
    return bands


FAMILIES = ("HL", "LH", "HH")

# Subband weights for the integer DWT (CCSDS 122.0-B, Table 3-4). The BPE
# codes weighted coefficients so that bit-planes are comparable across bands.
_BAND_WEIGHT = {
    ("LL", 3): 8, ("HL", 3): 8, ("LH", 3): 8, ("HH", 3): 4,
    ("HL", 2): 4, ("LH", 2): 4, ("HH", 2): 2,
    ("HL", 1): 2, ("LH", 1): 2, ("HH", 1): 1,
}


def _weight_vector() -> np.ndarray:
    wv = np.empty(64, dtype=np.int64)
    wv[DC] = _BAND_WEIGHT[("LL", 3)]
    for i, f in enumerate(FAMILIES):
        wv[1 + i] = _BAND_WEIGHT[(f, 3)]
        wv[4 + 4 * i: 8 + 4 * i] = _BAND_WEIGHT[(f, 2)]
        wv[16 + 16 * i: 32 + 16 * i] = _BAND_WEIGHT[(f, 1)]
    return wv


WEIGHTS = _weight_vector()


def to_blocks(coeffs: np.ndarray) -> np.ndarray:
    """(H, W) Mallat-layout coefficients -> (n_blocks, 64)."""
    h, w = coeffs.shape
    R, C = h >> 3, w >> 3
    b = _subbands(coeffs)
    out = np.empty((R * C, 64), dtype=coeffs.dtype)
    out[:, DC] = b[("LL", 3)].reshape(-1)
    for i, f in enumerate(FAMILIES):
        out[:, 1 + i] = b[(f, 3)].reshape(-1)
        out[:, 4 + 4 * i: 8 + 4 * i] = (
            b[(f, 2)].reshape(R, 2, C, 2).transpose(0, 2, 1, 3).reshape(R * C, 4))
        out[:, 16 + 16 * i: 32 + 16 * i] = (
            b[(f, 1)].reshape(R, 2, 2, C, 2, 2).transpose(0, 3, 1, 4, 2, 5).reshape(R * C, 16))
    return out


def from_blocks(blocks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Inverse of :func:`to_blocks`."""
    h, w = shape
    R, C = h >> 3, w >> 3
    out = np.zeros(shape, dtype=blocks.dtype)
    b = _subbands(out)
    b[("LL", 3)][...] = blocks[:, DC].reshape(R, C)
    for i, f in enumerate(FAMILIES):
        b[(f, 3)][...] = blocks[:, 1 + i].reshape(R, C)
        b[(f, 2)][...] = (blocks[:, 4 + 4 * i: 8 + 4 * i]
                          .reshape(R, C, 2, 2).transpose(0, 2, 1, 3).reshape(h >> 2, w >> 2))
        b[(f, 1)][...] = (blocks[:, 16 + 16 * i: 32 + 16 * i]
                          .reshape(R, C, 2, 2, 2, 2).transpose(0, 2, 4, 1, 3, 5).reshape(h >> 1, w >> 1))
    return out


def block_grid(shape: tuple[int, int]) -> tuple[int, int]:
    return shape[0] >> 3, shape[1] >> 3
