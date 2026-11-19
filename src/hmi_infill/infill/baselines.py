"""Reference infill methods (spatial and naive temporal)."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from skimage.restoration import inpaint_biharmonic

from .base import InfillInput


class ZeroFill:
    """Leave the decoder output as is (lost coefficients = 0)."""
    name = "zero"

    def __call__(self, x: InfillInput) -> np.ndarray:
        return x.damaged.astype(np.float64)


class Laplace:
    """Harmonic interpolation: solve Laplace's equation inside the mask with the
    surrounding pixels as Dirichlet boundary (smoothest membrane fill)."""
    name = "laplace"

    def __call__(self, x: InfillInput) -> np.ndarray:
        img = x.damaged.astype(np.float64)
        m = x.mask
        h, w = m.shape
        idx = -np.ones(m.shape, np.int64)
        ys, xs = np.nonzero(m)
        n = ys.size
        if n == 0:
            return img
        idx[ys, xs] = np.arange(n)
        rows, cols, vals = [], [], []
        diag = np.zeros(n)
        rhs = np.zeros(n)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = ys + dy, xs + dx
            inside = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
            diag[inside] += 1                      # Neumann at the image border
            ny_, nx_ = ny[inside], nx[inside]
            k = np.flatnonzero(inside)
            nb = idx[ny_, nx_]
            unknown = nb >= 0
            rows.append(k[unknown]); cols.append(nb[unknown]); vals.append(-np.ones(unknown.sum()))
            rhs[k[~unknown]] += img[ny_[~unknown], nx_[~unknown]]
        A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(n, n)) + sparse.diags(diag)
        out = img.copy()
        out[ys, xs] = spsolve(A.tocsc(), rhs)
        return out


class Biharmonic:
    """scikit-image biharmonic inpainting (smooth, C1-continuous fill)."""
    name = "biharmonic"

    def __call__(self, x: InfillInput) -> np.ndarray:
        img = x.damaged.astype(np.float64)
        lo, hi = img[~x.mask].min(), img[~x.mask].max()
        norm = (img - lo) / max(hi - lo, 1e-12)
        out = inpaint_biharmonic(norm, x.mask)
        return out * (hi - lo) + lo


class PreviousFrame:
    """Copy the co-located pixels of the nearest past frame (no derotation)."""
    name = "previous_frame"

    def __call__(self, x: InfillInput) -> np.ndarray:
        if not x.past:
            raise ValueError("needs at least one past frame")
        return x.past[0].astype(np.float64)
