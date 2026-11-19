"""Decoder Motion Vector Estimation (DMVE) - temporal infill.

Reference
---------
* J. Zhang, J. F. Arnold, M. R. Frater, "A cell-loss concealment technique for
  MPEG-2 coded video", IEEE Trans. CSVT 10(4), 2000.

Algorithm
---------
The lost region is tiled into blocks, processed from the outside in. For each
block, a template - a ring of `template` pixels around the block, using known
pixels (weight 1) and already concealed ones (weight delta) - is matched
against every reference frame over a search window of +-`search` pixels per
frame of time distance. The cost is the weighted mean absolute difference
over template pixels that are valid in both frames. The block is then copied
from the best-matching position.

With references on both sides (past and future), the best past and best future
candidates are averaged with weights inversely proportional to their template
cost (bidirectional mode). Otherwise the single best candidate is used.

Integer-pixel motion only, no intensity compensation: this is the classic
method; derotation priors and photometric correction are left for the
proposed improvements.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy import ndimage

from .base import InfillInput


@njit(cache=True)
def _search(cur, cw, todo, ref, rvalid, by, bx, B, T, R, min_block_valid):
    """Best integer displacement of block (by, bx) in `ref`. Returns (cost, dy, dx)."""
    H, W = cur.shape
    best, bdy, bdx = np.inf, 0, 0
    y0, y1 = max(by - T, 0), min(by + B + T, H)
    x0, x1 = max(bx - T, 0), min(bx + B + T, W)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            # the pixels to fill must be (mostly) usable in the displaced reference block
            ok = 0
            tot = 0
            for y in range(by, min(by + B, H)):
                for x in range(bx, min(bx + B, W)):
                    if not todo[y, x]:
                        continue
                    yy, xx = y + dy, x + dx
                    tot += 1
                    if 0 <= yy < H and 0 <= xx < W and rvalid[yy, xx]:
                        ok += 1
            if tot == 0 or ok < min_block_valid * tot:
                continue
            num = 0.0
            den = 0.0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    if by <= y < by + B and bx <= x < bx + B:
                        continue
                    wgt = cw[y, x]
                    if wgt <= 0:
                        continue
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < H and 0 <= xx < W and rvalid[yy, xx]:
                        num += wgt * abs(cur[y, x] - ref[yy, xx])
                        den += wgt
            if den < 0.25 * (2 * T * (B + T)):     # too little template overlap
                continue
            c = num / den
            if c < best:
                best, bdy, bdx = c, dy, dx
    return best, bdy, bdx


class DMVE:
    """Temporal infill by decoder-side motion vector estimation (block matching on
    the boundary template), uni- or bidirectional.

    block: block size; template: ring width; search: search radius in pixels
    per frame of time distance (36 min at 1"/px: solar rotation moves features
    by up to ~6 px, so 8 is a safe default).
    """
    name = "dmve"

    def __init__(self, block=8, template=4, search=8, delta=0.5, bidirectional=True,
                 min_block_valid=0.9):
        self.block, self.template, self.search = block, template, search
        self.delta, self.bidirectional = delta, bidirectional
        self.min_block_valid = min_block_valid

    def __call__(self, x: InfillInput) -> np.ndarray:
        img = x.damaged.astype(np.float64)
        valid = np.ones(img.shape, bool) if x.valid is None else x.valid
        todo = x.mask & valid
        refs = [(r, m, i + 1, "past") for i, (r, m) in enumerate(zip(x.past, x.past_masks))]
        refs += [(r, m, i + 1, "future") for i, (r, m) in enumerate(zip(x.future, x.future_masks))]
        if not todo.any() or not refs:
            return img
        refs = [(np.ascontiguousarray(r, np.float64), np.ascontiguousarray(valid & ~m), d, s)
                for r, m, d, s in refs]

        cur = img.copy()
        cw = np.where(~x.mask & valid, 1.0, 0.0)
        B, T = self.block, self.template
        dist = ndimage.distance_transform_edt(todo)
        H, W = img.shape
        blocks = sorted((dist[by:by + B, bx:bx + B][todo[by:by + B, bx:bx + B]].min(), by, bx)
                        for by in range(0, H, B) for bx in range(0, W, B)
                        if todo[by:by + B, bx:bx + B].any())

        for _, by, bx in blocks:
            best = {"past": (np.inf, None), "future": (np.inf, None)}
            for r, rv, d, side in refs:
                cost, dy, dx = _search(cur, cw, todo, r, rv, by, bx, B, T, self.search * d,
                                       self.min_block_valid)
                if cost < best[side][0]:
                    best[side] = (cost, (r, rv, dy, dx))
            cands = [v for v in best.values() if v[1] is not None]
            if not cands:
                continue
            if not self.bidirectional:
                cands = [min(cands, key=lambda v: v[0])]
            t = todo[by:by + B, bx:bx + B]
            hh, ww = t.shape
            acc = np.zeros((hh, ww))
            wsum = np.zeros((hh, ww))
            for cost, (r, rv, dy, dx) in cands:
                ys = np.clip(np.arange(by, by + hh) + dy, 0, H - 1)
                xs = np.clip(np.arange(bx, bx + ww) + dx, 0, W - 1)
                patch = r[np.ix_(ys, xs)]
                ok = rv[np.ix_(ys, xs)]
                wgt = 1.0 / (cost + 1e-9)
                acc += np.where(ok, patch * wgt, 0)
                wsum += np.where(ok, wgt, 0)
            fill = t & (wsum > 0)
            blk = cur[by:by + hh, bx:bx + ww]
            blk[fill] = (acc / np.where(wsum > 0, wsum, 1))[fill]
            cwb = cw[by:by + hh, bx:bx + ww]
            cwb[fill] = self.delta
        left = todo & (cw == 0)
        for r, rv, d, side in sorted(refs, key=lambda q: q[2]):
            use = left & rv
            cur[use] = r[use]
            left &= ~use
        return np.where(todo, cur, img)
