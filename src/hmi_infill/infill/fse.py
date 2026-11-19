"""Frequency Selective Extrapolation (FSE) - spatial infill.

References
----------
* K. Meisinger, A. Kaup, "Spatial error concealment of corrupted image data
  using frequency selective extrapolation", ICASSP 2004.
* J. Seiler, A. Kaup, "Complex-valued frequency selective extrapolation for
  fast image and video signal extrapolation", IEEE SPL 17(11), 2010.

Algorithm
---------
The lost region is tiled into small blocks, processed from the outside in. For
each block an extrapolation area L (block + border) is modelled as a sparse
weighted sum of 2-D DFT basis functions:

    g(m, n) = sum_k c_k exp(2 pi i (k1 m / M + k2 n / N))

fitted to the known pixels of L by successive approximation, minimising the
weighted residual energy  E = sum w (f - g)^2. The weight

    w(m, n) = rho ** dist((m, n), block centre)       for known pixels
              delta * rho ** dist(...)                 for already concealed pixels
              0                                        for unknown / invalid pixels

favours pixels close to the block. Each iteration picks the basis function
(and its complex-conjugate partner, so g stays real) that most reduces E,
adds gamma times its projection coefficient (gamma < 1 is the "orthogonality
deficiency compensation"), and updates the weighted residual directly in the
DFT domain:

    R_w(l) <- R_w(l) - c W(l - k) - conj(c) W(l + k)

with R_w = DFT(w * residual) and W = DFT(w). This is the fast frequency-domain
implementation of Seiler & Kaup.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy import ndimage

from .base import InfillInput


@njit(cache=True)
def _fse_iterate(R, W, n_iter, gamma):
    M, N = R.shape
    G = np.zeros((M, N), np.complex128)
    W0 = W[0, 0].real
    if W0 <= 0:
        return G
    for _ in range(n_iter):
        best, k1, k2 = -1.0, 0, 0
        for a in range(M):
            for b in range(N):
                v = R[a, b].real ** 2 + R[a, b].imag ** 2
                if v > best:
                    best, k1, k2 = v, a, b
        p1, p2 = (-k1) % M, (-k2) % N
        self_conj = (p1 == k1) and (p2 == k2)
        if self_conj:
            c = gamma * R[k1, k2].real / W0 + 0j
        else:
            c = gamma * R[k1, k2] / W0
        cc = np.conj(c)
        G[k1, k2] += c
        if not self_conj:
            G[p1, p2] += cc
        for a in range(M):
            for b in range(N):
                upd = c * W[(a - k1) % M, (b - k2) % N]
                if not self_conj:
                    upd += cc * W[(a + k1) % M, (b + k2) % N]
                R[a, b] -= upd
    return G


def extrapolate_area(f, w, n_iter=200, gamma=0.5):
    """FSE on one area: f values (anything where w == 0), w weights. Returns model g."""
    M, N = f.shape
    R = np.fft.fft2(np.where(w > 0, f, 0.0) * w)
    W = np.fft.fft2(w)
    G = _fse_iterate(R.astype(np.complex128), W.astype(np.complex128), n_iter, gamma)
    return np.real(np.fft.ifft2(G)) * M * N


class FSE:
    """Spatial infill by Frequency Selective Extrapolation.

    Parameters (defaults close to Seiler & Kaup): block 8x8 processed with a
    12-pixel border (32x32 area / FFT), rho = 0.8, gamma = 0.5, 200
    iterations, concealed pixels re-used with weight delta = 0.2.
    """
    name = "fse"

    def __init__(self, block=8, border=12, rho=0.8, gamma=0.5, n_iter=200, delta=0.2,
                 normalise=True):
        self.block, self.border, self.rho = block, border, rho
        self.gamma, self.n_iter, self.delta = gamma, n_iter, delta
        self.normalise = normalise

    def __call__(self, x: InfillInput) -> np.ndarray:
        img = x.damaged.astype(np.float64)
        valid = np.ones(img.shape, bool) if x.valid is None else x.valid
        todo = x.mask & valid
        if not todo.any():
            return img
        B, S = self.block, self.border
        A = B + 2 * S

        # Remove the mean/scale of the known data so FSE sees a zero-mean signal.
        known = ~x.mask & valid
        mu, sd = (img[known].mean(), img[known].std() or 1.0) if self.normalise else (0.0, 1.0)
        work = np.pad((img - mu) / sd, S)
        wmap = np.pad(np.where(known, 1.0, 0.0), S)          # 1 known, delta concealed, 0 unknown
        todo_p = np.pad(todo, S)

        yy, xx = np.mgrid[:A, :A]
        c = S + (B - 1) / 2
        decay = self.rho ** np.hypot(yy - c, xx - c)

        # Blocks containing pixels to fill, ordered from the hole boundary inwards.
        dist = ndimage.distance_transform_edt(todo)
        H, Wd = img.shape
        blocks = []
        for by in range(0, H, B):
            for bx in range(0, Wd, B):
                t = todo[by:by + B, bx:bx + B]
                if t.any():
                    blocks.append((dist[by:by + B, bx:bx + B][t].min(), by, bx))
        blocks.sort()

        for _, by, bx in blocks:
            sl = np.s_[by:by + A, bx:bx + A]               # area in padded coordinates
            f = work[sl]
            w = wmap[sl] * decay
            if f.shape != (A, A) or not (w > 0).any():
                continue
            g = extrapolate_area(f, w, self.n_iter, self.gamma)
            inner = np.s_[by + S:by + S + B, bx + S:bx + S + B]
            t = todo_p[inner]
            blk = work[inner]
            blk[t] = g[S:S + B, S:S + B][t]
            wm = wmap[inner]
            wm[t] = self.delta
        out = work[S:-S, S:-S] * sd + mu
        return np.where(todo, out, img)
