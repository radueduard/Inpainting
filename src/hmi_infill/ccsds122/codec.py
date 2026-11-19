"""Image-level CCSDS 122 encoder and loss-tolerant decoder."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from . import dwt
from .blocks import WEIGHTS, to_blocks, from_blocks, block_grid
from .bpe import encode_segment, decode_segment, P_UNKNOWN

BLOCK_OK, BLOCK_TRUNCATED, BLOCK_LOST = 0, 1, 2
INT32_MIN, INT32_MAX = np.iinfo(np.int32).min, np.iinfo(np.int32).max


@dataclass
class CodecParams:
    segment_blocks: int = 256          # S: blocks per segment (16 .. 2**20 in the standard)
    byte_limit: int | None = None      # truncate every segment to this many bytes (lossy mode)


@dataclass
class Geometry:
    """Side information the receiver is assumed to know (fixed per mission)."""
    shape: tuple[int, int]             # original image shape
    padded_shape: tuple[int, int]      # shape after padding to multiples of 8
    segment_blocks: int

    @property
    def n_blocks(self) -> int:
        r, c = block_grid(self.padded_shape)
        return r * c

    @property
    def n_segments(self) -> int:
        return -(-self.n_blocks // self.segment_blocks)

    def segment_range(self, s: int) -> tuple[int, int]:
        a = s * self.segment_blocks
        return a, min(a + self.segment_blocks, self.n_blocks)

    def to_dict(self):
        return asdict(self)


@dataclass
class EncodedImage:
    segments: list[bytes]
    geometry: Geometry

    @property
    def n_bytes(self) -> int:
        return sum(len(s) for s in self.segments)


@dataclass
class DecodeResult:
    image: np.ndarray            # (H, W) int64; lost coefficients set to 0
    coeff_est: np.ndarray        # (Hp, Wp) int64, Mallat layout, unweighted
    coeff_lo: np.ndarray         # (Hp, Wp) int32 lower bound (INT32_MIN if unknown)
    coeff_hi: np.ndarray         # (Hp, Wp) int32 upper bound (INT32_MAX if unknown)
    bitplane: np.ndarray         # (n_blocks, 64) int8: lowest received bit-plane p (99 = nothing)
    block_status: np.ndarray     # (R, C) uint8: BLOCK_OK / BLOCK_TRUNCATED / BLOCK_LOST
    mask_lost: np.ndarray        # (H, W) bool: pixels depending on any lost coefficient
    mask_core: np.ndarray        # (H, W) bool: the 8x8 footprints of lost blocks only
    mask_uncertain: np.ndarray   # (H, W) bool: pixels depending on any inexact coefficient
    segment_status: np.ndarray   # (n_segments,) uint8: 0 ok / 1 truncated / 2 lost


def _pad(image: np.ndarray) -> np.ndarray:
    h, w = image.shape
    return np.pad(image, ((0, (-h) % 8), (0, (-w) % 8)), mode="edge")


def encode(image: np.ndarray, params: CodecParams = CodecParams()) -> EncodedImage:
    image = np.asarray(image)
    if not np.issubdtype(image.dtype, np.integer):
        raise TypeError("CCSDS 122 codes integer images; convert with hmi_infill.prep.to_int")
    padded = _pad(image.astype(np.int64))
    geom = Geometry(tuple(image.shape), tuple(padded.shape), params.segment_blocks)
    blk = to_blocks(dwt.forward(padded)) * WEIGHTS
    segments = []
    for s in range(geom.n_segments):
        a, b = geom.segment_range(s)
        buf, n = encode_segment(np.ascontiguousarray(blk[a:b]), s,
                                int(s == 0), int(s == geom.n_segments - 1))
        data = bytes(buf[:n])
        if params.byte_limit is not None:
            data = data[: params.byte_limit]
        segments.append(data)
    return EncodedImage(segments, geom)


def _intervals(mag, sgn, p):
    """Bounds of the weighted coefficients from the decoder state (block layout)."""
    known = p != P_UNKNOWN
    width = np.where(known, (np.int64(1) << np.minimum(p, 62)) - 1, 0)
    lo = np.empty_like(mag)
    hi = np.empty_like(mag)
    # DC: two's complement value with unknown low bits
    lo[:, 0] = mag[:, 0]
    hi[:, 0] = mag[:, 0] + width[:, 0]
    # AC: sign-magnitude
    m, s, wd = mag[:, 1:], sgn[:, 1:], width[:, 1:]
    top = m + wd
    lo[:, 1:] = np.where(s > 0, m, -top)
    hi[:, 1:] = np.where(s < 0, -m, top)
    return lo, hi, known


def decode(segments: list[bytes | None], geom: Geometry) -> DecodeResult:
    """Decode segments; None means the segment was lost, short bytes mean truncated."""
    nb = geom.n_blocks
    mag = np.zeros((nb, 64), np.int64)
    sgn = np.zeros((nb, 64), np.int8)
    p = np.full((nb, 64), P_UNKNOWN, np.int64)
    seg_status = np.full(geom.n_segments, 2, np.uint8)

    for s in range(geom.n_segments):
        data = segments[s] if s < len(segments) else None
        if not data:
            continue
        a, b = geom.segment_range(s)
        ok, _, m_, s_, p_ = decode_segment(np.frombuffer(data, np.uint8).copy(), b - a)
        if not ok:
            continue
        mag[a:b], sgn[a:b], p[a:b] = m_, s_, p_
        seg_status[s] = 0 if np.all(p_ == 0) else 1

    # Blocks whose DC was not received carry no usable information.
    dc_lost = p[:, 0] == P_UNKNOWN
    p[dc_lost] = P_UNKNOWN

    lo_w, hi_w, known = _intervals(mag, sgn, p)
    w = WEIGHTS[None, :]
    lo = -((-lo_w) // w)                     # ceil(lo / w): true values are multiples of w
    hi = hi_w // w
    est = np.where(np.arange(64)[None, :] == 0, (lo + hi) // 2, np.trunc((lo + hi) / 2).astype(np.int64))
    est[~known] = 0
    lo = np.where(known, np.clip(lo, INT32_MIN + 1, INT32_MAX - 1), INT32_MIN).astype(np.int32)
    hi = np.where(known, np.clip(hi, INT32_MIN + 1, INT32_MAX - 1), INT32_MAX).astype(np.int32)

    shp = geom.padded_shape
    coeff_est = from_blocks(est, shp)
    h, w_ = geom.shape
    image = dwt.inverse(coeff_est)[:h, :w_]

    exact = (p == 0)
    status = np.where(dc_lost, BLOCK_LOST, np.where(exact.all(axis=1), BLOCK_OK, BLOCK_TRUNCATED))
    lost_c = from_blocks(np.broadcast_to(dc_lost[:, None], (nb, 64)).copy(), shp)
    unc_c = from_blocks(~exact, shp)

    return DecodeResult(
        image=image,
        coeff_est=coeff_est,
        coeff_lo=from_blocks(lo, shp),
        coeff_hi=from_blocks(hi, shp),
        bitplane=np.minimum(p, P_UNKNOWN).astype(np.int8),
        block_status=status.reshape(block_grid(shp)).astype(np.uint8),
        mask_lost=dwt.support(lost_c)[:h, :w_],
        mask_core=np.kron(dc_lost.reshape(block_grid(shp)), np.ones((8, 8), bool))[:h, :w_],
        mask_uncertain=dwt.support(unc_c)[:h, :w_],
        segment_status=seg_status,
    )
