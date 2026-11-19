"""End-to-end loss simulation: encode -> packetize -> channel -> depacketize -> decode.

A Trial holds everything an infill method or an evaluation needs for one frame:
ground truth, the damaged decode, masks, coefficient bounds and metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import ccsds122, ccsds133
from .channel import NoLoss, transmit


@dataclass
class Trial:
    truth: np.ndarray           # decode without losses (== input for lossless coding)
    damaged: np.ndarray         # decode of what arrived; lost coefficients = 0
    mask_lost: np.ndarray       # pixels affected by lost blocks (full inverse-DWT support)
    mask_core: np.ndarray       # 8x8 pixel footprints of the lost blocks
    mask_uncertain: np.ndarray  # pixels affected by lost or truncated coefficients
    block_status: np.ndarray    # (H/8, W/8): 0 ok, 1 truncated, 2 lost
    bitplane: np.ndarray        # (n_blocks, 64) lowest received bit-plane per coefficient
    coeff_est: np.ndarray       # Mallat-layout DWT coefficients as decoded
    coeff_lo: np.ndarray        # lower bounds (int32, INT32_MIN = unknown)
    coeff_hi: np.ndarray        # upper bounds (int32, INT32_MAX = unknown)
    original: np.ndarray        # integer input image
    valid: np.ndarray | None = None   # on-disk / finite pixels of the input
    meta: dict = field(default_factory=dict)

    ARRAYS = ("truth", "damaged", "mask_lost", "mask_core", "mask_uncertain", "block_status", "bitplane",
              "coeff_est", "coeff_lo", "coeff_hi", "original", "valid")

    @property
    def geometry(self) -> ccsds122.Geometry:
        g = self.meta["geometry"]
        return ccsds122.Geometry(tuple(g["shape"]), tuple(g["padded_shape"]), g["segment_blocks"])

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for k in self.ARRAYS:
            v = getattr(self, k)
            if v is None:
                continue
            if v.dtype == np.int64:
                v = v.astype(np.int32)
            arrays[k] = v
        np.savez_compressed(d / "trial.npz", **arrays)
        (d / "meta.json").write_text(json.dumps(self.meta, indent=2, default=_json_default))
        return d

    @classmethod
    def load(cls, directory: str | Path) -> "Trial":
        d = Path(directory)
        with np.load(d / "trial.npz") as z:
            arrays = {k: z[k] for k in z.files}
        meta = json.loads((d / "meta.json").read_text())
        arrays.setdefault("valid", None)
        return cls(meta=meta, **arrays)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _make_trial(image, valid, enc, truth_res, segments, seg_status_rx, lost_idx, n_packets,
                channel, extra_meta) -> Trial:
    res = ccsds122.decode(segments, enc.geometry)
    meta = {
        "geometry": enc.geometry.to_dict(),
        "compressed_bytes": enc.n_bytes,
        "bits_per_pixel": enc.n_bytes * 8 / image.size,
        "n_packets": int(n_packets),
        "lost_packets": [int(i) for i in lost_idx],
        "packet_loss_rate": len(lost_idx) / max(n_packets, 1),
        "segments_lost": int(np.sum(seg_status_rx == ccsds133.SEG_LOST)),
        "segments_truncated": int(np.sum(seg_status_rx == ccsds133.SEG_TRUNCATED)),
        "blocks_lost": int(np.sum(res.block_status == ccsds122.BLOCK_LOST)),
        "blocks_truncated": int(np.sum(res.block_status == ccsds122.BLOCK_TRUNCATED)),
        "pixels_lost_fraction": float(res.mask_lost.mean()),
        "pixels_core_fraction": float(res.mask_core.mean()),
        "channel": channel.describe(),
        **extra_meta,
    }
    meta = json.loads(json.dumps(meta, default=_json_default))   # same form as after save/load
    return Trial(
        truth=truth_res.image, damaged=res.image, mask_lost=res.mask_lost, mask_core=res.mask_core,
        mask_uncertain=res.mask_uncertain, block_status=res.block_status, bitplane=res.bitplane,
        coeff_est=res.coeff_est, coeff_lo=res.coeff_lo, coeff_hi=res.coeff_hi,
        original=np.asarray(image), valid=valid, meta=meta,
    )


def simulate_sequence(images: list[np.ndarray], codec: ccsds122.CodecParams = ccsds122.CodecParams(),
                      channel=NoLoss(), seed: int = 0, max_packet_len: int = 1024, apid: int = 0x100,
                      valid: list[np.ndarray] | None = None, extra_meta: dict | None = None) -> list[Trial]:
    """Simulate the transmission of a sequence of integer images.

    All frames are packetized into one continuous packet stream (shared
    sequence counter) and passed through the channel together, so bursts can
    span frame boundaries, as on a real downlink.
    """
    rng = np.random.default_rng(seed)
    counter = ccsds133.SequenceCounter()
    encoded, truths, streams = [], [], []
    for i, img in enumerate(images):
        enc = ccsds122.encode(img, codec)
        encoded.append(enc)
        truths.append(ccsds122.decode(enc.segments, enc.geometry))
        streams.append(ccsds133.packetize(enc.segments, image_id=i, apid=apid,
                                          max_packet_len=max_packet_len, counter=counter))

    all_packets = [p for s in streams for p in s]
    received, lost = transmit(all_packets, channel, rng)

    trials, start = [], 0
    for i, (img, enc) in enumerate(zip(images, encoded)):
        n = len(streams[i])
        lost_i = lost[(lost >= start) & (lost < start + n)] - start
        start += n
        segs, status = ccsds133.depacketize(received, image_id=i, n_segments=enc.geometry.n_segments)
        meta = {"frame": i, "seed": seed, "max_packet_len": max_packet_len,
                "codec": {"segment_blocks": codec.segment_blocks, "byte_limit": codec.byte_limit},
                **(extra_meta or {})}
        trials.append(_make_trial(img, None if valid is None else valid[i], enc, truths[i],
                                  segs, status, lost_i, n, channel, meta))
    return trials


def simulate(image: np.ndarray, **kwargs) -> Trial:
    """Single-frame convenience wrapper around :func:`simulate_sequence`."""
    valid = kwargs.pop("valid", None)
    return simulate_sequence([image], valid=None if valid is None else [valid], **kwargs)[0]
