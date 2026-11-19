"""Plotting helpers for the notebooks (astropy normalisation + matplotlib)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import (AsinhStretch, ImageNormalize, LinearStretch, PercentileInterval)
from matplotlib.colors import ListedColormap

CMAPS = {"MAGNETOGRAM": "gray", "DOPPLERGRAM": "RdBu_r", "CONTINUUM INTENSITY": "afmhot"}


def norm_for(img, percentile=99.5, stretch="linear", symmetric=False):
    finite = img[np.isfinite(img)]
    if symmetric:
        v = np.percentile(np.abs(finite), percentile)
        return ImageNormalize(vmin=-v, vmax=v, stretch=LinearStretch())
    st = AsinhStretch() if stretch == "asinh" else LinearStretch()
    return ImageNormalize(finite, interval=PercentileInterval(percentile), stretch=st)


def show(img, ax=None, title=None, cmap="gray", norm=None, colorbar=True, **kw):
    ax = ax or plt.gca()
    im = ax.imshow(img, origin="lower", cmap=cmap, norm=norm or norm_for(img, **kw),
                   interpolation="nearest")
    if title:
        ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    return im


def overlay_mask(ax, mask, color="cyan", alpha=0.35):
    cm = ListedColormap([(0, 0, 0, 0), color])
    ax.imshow(mask.astype(np.uint8), origin="lower", cmap=cm, alpha=alpha,
              interpolation="nearest", vmin=0, vmax=1)


def show_block_status(status, ax=None, title="block status"):
    ax = ax or plt.gca()
    cm = ListedColormap(["#2b8a3e", "#f59f00", "#c92a2a"])
    im = ax.imshow(status, origin="lower", cmap=cm, vmin=0, vmax=2, interpolation="nearest")
    cb = plt.colorbar(im, ax=ax, ticks=[0.33, 1, 1.67], fraction=0.046, pad=0.02)
    cb.ax.set_yticklabels(["ok", "truncated", "lost"])
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])


def compare(truth, damaged, filled, mask, zoom=None, title="", cmap="gray", **kw):
    """Truth / damaged / filled / error, optionally zoomed to (y0, y1, x0, x1)."""
    sl = np.s_[:, :] if zoom is None else np.s_[zoom[0]:zoom[1], zoom[2]:zoom[3]]
    nrm = norm_for(truth[sl].astype(float), **kw)
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.5))
    show(truth[sl], axs[0], "truth", cmap, nrm)
    show(damaged[sl], axs[1], "damaged", cmap, nrm)
    overlay_mask(axs[1], mask[sl])
    show(filled[sl], axs[2], "filled", cmap, nrm)
    err = np.where(mask, filled - truth, 0)[sl].astype(float)
    show(err, axs[3], "error (in mask)", "RdBu_r", norm_for(err, symmetric=True))
    fig.suptitle(title)
    fig.tight_layout()
    return fig
