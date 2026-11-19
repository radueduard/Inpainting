"""Evaluation metrics, computed inside the infilled region."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def _region(mask, valid):
    m = np.asarray(mask, bool)
    if valid is not None:
        m = m & valid
    return m


def evaluate(estimate: np.ndarray, truth: np.ndarray, mask: np.ndarray,
             valid: np.ndarray | None = None, scale: float = 1.0,
             data_range: float | None = None, period: float | None = None) -> dict:
    """Error statistics of `estimate` against `truth` inside `mask` (and `valid`).

    `scale` converts integer units to physical units (PrepInfo.scale), so that
    flux and errors are reported in G, m/s or DN. `period` (physical units)
    wraps errors for angles, e.g. 180 for the ambiguous azimuth.
    """
    m = _region(mask, valid)
    n = int(m.sum())
    if n == 0:
        return {"n_pixels": 0}
    e = (estimate[m].astype(np.float64) - truth[m]) * scale
    if period:
        e = (e + period / 2) % period - period / 2
    t = truth[m].astype(np.float64) * scale
    if data_range is None:
        ref = truth[valid] if valid is not None else truth
        data_range = float(np.ptp(ref)) * scale
    mse = float(np.mean(e ** 2))
    out = {
        "n_pixels": n,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(e))),
        "max_abs": float(np.max(np.abs(e))),
        "bias": float(np.mean(e)),
        "psnr": float(10 * np.log10(data_range ** 2 / mse)) if mse > 0 else float("inf"),
        # Signed and unsigned "flux" (sum over region; for magnetograms in G*px)
        "flux_rel_err": float(np.sum(e) / max(np.abs(np.sum(t)), 1e-12)),
        "unsigned_flux_rel_err": float((np.sum(np.abs(t + e)) - np.sum(np.abs(t))) / max(np.sum(np.abs(t)), 1e-12)),
        # Ratio of local variance: < 1 means the fill is too smooth
        "std_ratio": float(np.std(t + e) / np.std(t)) if np.std(t) > 0 else float("nan"),
    }
    out["ssim_bbox"] = ssim_bbox(estimate, truth, m, data_range / scale)
    return out


def ssim_bbox(estimate, truth, mask, data_range) -> float:
    """SSIM over the bounding box of the mask (SSIM needs a 2-D window)."""
    ys, xs = np.nonzero(mask)
    y0, y1 = max(ys.min() - 4, 0), min(ys.max() + 5, mask.shape[0])
    x0, x1 = max(xs.min() - 4, 0), min(xs.max() + 5, mask.shape[1])
    a = estimate[y0:y1, x0:x1].astype(np.float64)
    b = truth[y0:y1, x0:x1].astype(np.float64)
    if min(a.shape) < 7:
        return float("nan")
    _, smap = structural_similarity(a, b, data_range=data_range, full=True)
    return float(smap[mask[y0:y1, x0:x1]].mean())
