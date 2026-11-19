"""CCSDS 133.0-B Space Packet Protocol: packetization of CCSDS 122 segments.

Primary header (6 bytes, Section 4.1.3):
    version (3) | type (1) | sec. header flag (1) | APID (11)
    sequence flags (2) | packet sequence count (14)
    packet data length (16) = octets in the data field - 1

Secondary header (mission-defined, 6 bytes here):
    image id (16) | segment index (32)

Each CCSDS 122 segment starts a new packet and is split over as many packets as
needed. The sequence flags mark first / continuation / last packets, so the
receiver can tell a complete segment from a truncated one.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

SEQ_CONT, SEQ_FIRST, SEQ_LAST, SEQ_UNSEG = 0b00, 0b01, 0b10, 0b11
PRIMARY_LEN = 6
SECONDARY_FMT = ">HI"
SECONDARY_LEN = struct.calcsize(SECONDARY_FMT)
COUNT_MOD = 1 << 14

SEG_OK, SEG_TRUNCATED, SEG_LOST = 0, 1, 2


@dataclass(frozen=True)
class SpacePacket:
    apid: int
    seq_flags: int
    seq_count: int
    image_id: int
    segment_index: int
    payload: bytes
    version: int = 0
    ptype: int = 0               # 0 = telemetry

    def to_bytes(self) -> bytes:
        data = struct.pack(SECONDARY_FMT, self.image_id, self.segment_index) + self.payload
        w0 = (self.version << 13) | (self.ptype << 12) | (1 << 11) | (self.apid & 0x7FF)
        w1 = (self.seq_flags << 14) | (self.seq_count % COUNT_MOD)
        return struct.pack(">HHH", w0, w1, len(data) - 1) + data

    @classmethod
    def from_bytes(cls, raw: bytes) -> "SpacePacket":
        w0, w1, length = struct.unpack(">HHH", raw[:PRIMARY_LEN])
        data = raw[PRIMARY_LEN: PRIMARY_LEN + length + 1]
        if len(data) != length + 1:
            raise ValueError("packet shorter than its length field")
        if not (w0 >> 11) & 1:
            raise ValueError("secondary header expected")
        image_id, seg = struct.unpack(SECONDARY_FMT, data[:SECONDARY_LEN])
        return cls(apid=w0 & 0x7FF, seq_flags=w1 >> 14, seq_count=w1 & 0x3FFF,
                   image_id=image_id, segment_index=seg, payload=bytes(data[SECONDARY_LEN:]),
                   version=w0 >> 13, ptype=(w0 >> 12) & 1)


class SequenceCounter:
    """Per-APID 14-bit packet sequence counter, continuous across images."""

    def __init__(self, start: int = 0):
        self.value = start % COUNT_MOD

    def next(self) -> int:
        v = self.value
        self.value = (v + 1) % COUNT_MOD
        return v


def packetize(segments: list[bytes], image_id: int, apid: int = 0x100,
              max_packet_len: int = 1024, counter: SequenceCounter | None = None) -> list[bytes]:
    """Split each segment into Space Packets of at most max_packet_len octets."""
    chunk = max_packet_len - PRIMARY_LEN - SECONDARY_LEN
    if chunk <= 0:
        raise ValueError("max_packet_len too small")
    counter = counter or SequenceCounter()
    out = []
    for s, data in enumerate(segments):
        pieces = [data[i:i + chunk] for i in range(0, max(len(data), 1), chunk)]
        for i, piece in enumerate(pieces):
            if len(pieces) == 1:
                flags = SEQ_UNSEG
            elif i == 0:
                flags = SEQ_FIRST
            elif i == len(pieces) - 1:
                flags = SEQ_LAST
            else:
                flags = SEQ_CONT
            out.append(SpacePacket(apid, flags, counter.next(), image_id, s, piece).to_bytes())
    return out


def depacketize(packets: list[bytes], image_id: int, n_segments: int):
    """Rebuild segments of one image from the packets that arrived.

    A segment whose first packet is missing is lost. Otherwise its data is the
    longest run of consecutive packets (by sequence count) starting at the
    first one; if that run does not reach the last packet it is truncated.

    Returns (segments, status) with segments[i] = bytes or None and status
    an array of SEG_OK / SEG_TRUNCATED / SEG_LOST.
    """
    groups: dict[int, list[SpacePacket]] = {}
    for raw in packets:
        pk = SpacePacket.from_bytes(raw)
        if pk.image_id == image_id:
            groups.setdefault(pk.segment_index, []).append(pk)

    segments: list[bytes | None] = [None] * n_segments
    status = np.full(n_segments, SEG_LOST, np.uint8)
    for s, pks in groups.items():
        if s >= n_segments or pks[0].seq_flags not in (SEQ_FIRST, SEQ_UNSEG):
            continue
        data = bytearray(pks[0].payload)
        complete = pks[0].seq_flags == SEQ_UNSEG
        prev = pks[0].seq_count
        for pk in pks[1:]:
            if complete or pk.seq_count != (prev + 1) % COUNT_MOD or pk.seq_flags not in (SEQ_CONT, SEQ_LAST):
                break
            data += pk.payload
            prev = pk.seq_count
            complete = pk.seq_flags == SEQ_LAST
        segments[s] = bytes(data)
        status[s] = SEG_OK if complete else SEG_TRUNCATED
    return segments, status
