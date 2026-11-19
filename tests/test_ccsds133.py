import numpy as np

from hmi_infill.ccsds133 import (SEG_LOST, SEG_OK, SEG_TRUNCATED, SequenceCounter, SpacePacket,
                                 depacketize, packetize)
from hmi_infill.ccsds133.packet import SEQ_FIRST


def test_header_roundtrip():
    pk = SpacePacket(apid=0x7AB, seq_flags=SEQ_FIRST, seq_count=16383, image_id=5,
                     segment_index=70000, payload=b"hello")
    raw = pk.to_bytes()
    assert len(raw) == 6 + 6 + 5
    assert SpacePacket.from_bytes(raw) == pk


def _segments():
    rng = np.random.default_rng(0)
    return [rng.bytes(n) for n in (10, 1000, 2500, 3000, 1)]


def test_packetize_roundtrip():
    segs = _segments()
    pk = packetize(segs, image_id=3, max_packet_len=512)
    out, status = depacketize(pk, image_id=3, n_segments=len(segs))
    assert out == segs and np.all(status == SEG_OK)


def test_loss_classification():
    segs = _segments()
    pk = packetize(segs, image_id=0, max_packet_len=512)
    # packets per segment with 500-byte payload: 1, 2, 5, 6, 1
    lost = {1,          # first packet of segment 1 -> lost
            5}          # third packet of segment 2 -> truncated after 2 packets
    rx = [p for i, p in enumerate(pk) if i not in lost]
    out, status = depacketize(rx, image_id=0, n_segments=len(segs))
    assert status.tolist() == [SEG_OK, SEG_LOST, SEG_TRUNCATED, SEG_OK, SEG_OK]
    assert out[2] == segs[2][:1000]


def test_counter_wraps_across_images():
    c = SequenceCounter(start=16383)
    a = packetize([b"x" * 10], image_id=0, counter=c)
    b = packetize([b"y" * 10], image_id=1, counter=c)
    assert SpacePacket.from_bytes(a[0]).seq_count == 16383
    assert SpacePacket.from_bytes(b[0]).seq_count == 0
