"""Integer 9/7-M discrete wavelet transform of CCSDS 122.0-B (Section 3).

The 1-D forward transform of a signal x of even length N produces highpass
coefficients C_j and lowpass coefficients D_j (j = 0 .. N/2-1):

    C_j = x_{2j+1} - floor( 9/16 (x_{2j} + x_{2j+2}) - 1/16 (x_{2j-2} + x_{2j+4}) + 1/2 )
    D_j = x_{2j}   - floor( -1/4 (C_{j-1} + C_j) + 1/2 )

with whole-sample symmetric extension of x and half-sample symmetric extension
of C (C_{-1} = C_0). This reproduces the explicit boundary formulas of the
standard. Both steps are lifting steps, so the inverse is exact.

A 2-D level transforms rows, then columns. Three levels are applied to the LL
band. Layout of the result at each level (Mallat layout):

    +----+----+
    | LL | HL |     HL = highpass along rows (horizontal detail)
    +----+----+     LH = highpass along columns
    | LH | HH |
    +----+----+
"""

from __future__ import annotations

import numpy as np

LEVELS = 3


# --------------------------------------------------------------------------
# Neighbour helpers with the symmetric extensions used by the standard.
# All operate along the last axis.
# --------------------------------------------------------------------------

def _e_m1(e):  # e_{j-1}, whole-sample reflection: e_{-1} = e_1
    return np.concatenate([e[..., 1:2], e[..., :-1]], axis=-1)


def _e_p1(e):  # e_{j+1}, at the right end e_M = e_{M-1}
    return np.concatenate([e[..., 1:], e[..., -1:]], axis=-1)


def _e_p2(e):  # e_{j+2}, e_M = e_{M-1}, e_{M+1} = e_{M-2}
    return np.concatenate([e[..., 2:], e[..., -1:], e[..., -2:-1]], axis=-1)


def _c_m1(c):  # C_{j-1}, C_{-1} = C_0
    return np.concatenate([c[..., :1], c[..., :-1]], axis=-1)


def _predict(e):
    return (9 * (e + _e_p1(e)) - (_e_m1(e) + _e_p2(e)) + 8) >> 4


def _update(c):
    return (-(_c_m1(c) + c) + 2) >> 2


# --------------------------------------------------------------------------
# 1-D transforms along the last axis
# --------------------------------------------------------------------------

def fwd1d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward 1-D transform along the last axis. Returns (low, high)."""
    if x.shape[-1] % 2 or x.shape[-1] < 4:
        raise ValueError("length must be even and >= 4")
    e = x[..., 0::2]
    o = x[..., 1::2]
    c = o - _predict(e)
    d = e - _update(c)
    return d, c


def inv1d(d: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Inverse of :func:`fwd1d`."""
    e = d + _update(c)
    o = c + _predict(e)
    out = np.empty(d.shape[:-1] + (2 * d.shape[-1],), dtype=np.int64)
    out[..., 0::2] = e
    out[..., 1::2] = o
    return out


def _inv1d_support(d: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Boolean version of :func:`inv1d`: which outputs depend on flagged inputs."""
    e = d | _c_m1(c) | c
    o = c | e | _e_p1(e) | _e_m1(e) | _e_p2(e)
    out = np.empty(d.shape[:-1] + (2 * d.shape[-1],), dtype=bool)
    out[..., 0::2] = e
    out[..., 1::2] = o
    return out


# --------------------------------------------------------------------------
# 2-D multilevel
# --------------------------------------------------------------------------

def _fwd2d_level(a: np.ndarray) -> np.ndarray:
    lo, hi = fwd1d(a)                      # rows
    a = np.concatenate([lo, hi], axis=1)
    lo, hi = fwd1d(a.T)                    # columns
    return np.concatenate([lo, hi], axis=1).T


def _inv2d_level(a: np.ndarray, inv) -> np.ndarray:
    h, w = a.shape
    t = a.T
    a = inv(t[:, : h // 2], t[:, h // 2:]).T     # columns
    return inv(a[:, : w // 2], a[:, w // 2:])    # rows


def forward(image: np.ndarray, levels: int = LEVELS) -> np.ndarray:
    """3-level 2-D integer DWT. Image sides must be multiples of 2**levels."""
    out = np.asarray(image, dtype=np.int64).copy()
    h, w = out.shape
    if h % (1 << levels) or w % (1 << levels):
        raise ValueError(f"image shape {out.shape} must be a multiple of {1 << levels}")
    for lev in range(levels):
        hh, ww = h >> lev, w >> lev
        out[:hh, :ww] = _fwd2d_level(out[:hh, :ww])
    return out


def inverse(coeffs: np.ndarray, levels: int = LEVELS) -> np.ndarray:
    out = np.asarray(coeffs, dtype=np.int64).copy()
    h, w = out.shape
    for lev in reversed(range(levels)):
        hh, ww = h >> lev, w >> lev
        out[:hh, :ww] = _inv2d_level(out[:hh, :ww], inv1d)
    return out


def support(flags: np.ndarray, levels: int = LEVELS) -> np.ndarray:
    """Propagate a boolean coefficient mask through the inverse transform.

    Returns the pixels whose reconstruction depends on at least one flagged
    coefficient (exact synthesis-filter support).
    """
    out = np.asarray(flags, dtype=bool).copy()
    h, w = out.shape
    for lev in reversed(range(levels)):
        hh, ww = h >> lev, w >> lev
        out[:hh, :ww] = _inv2d_level(out[:hh, :ww], _inv1d_support)
    return out
