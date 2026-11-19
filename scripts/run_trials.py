"""Generate loss-simulation trials from a directory of FITS frames (or synthetic data).

Every frame goes through: integer conversion -> CCSDS 122 -> CCSDS 133 ->
channel -> depacketize -> decode, and is saved as trials/<name>/frame_XX/.

Examples:
    python scripts/run_trials.py --fits data/raw/hmi.M_45s --loss 0.01 --burst 5 --name M_ge1pct
    python scripts/run_trials.py --synthetic magnetogram --size 1024 --loss 0.02 --name synth
"""

import argparse
import json
from pathlib import Path

from hmi_infill import prep, synthetic
from hmi_infill.ccsds122 import CodecParams
from hmi_infill.channel import Bernoulli, GilbertElliott, NoLoss
from hmi_infill.pipeline import simulate_sequence


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fits", help="directory with FITS frames (sorted by name = time)")
    src.add_argument("--synthetic", choices=["continuum", "magnetogram"])
    ap.add_argument("--size", type=int, default=1024, help="synthetic image size")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--crop", type=int, help="centred crop of the FITS frames (pixels)")
    ap.add_argument("--scale", type=float, help="physical units per integer step")
    ap.add_argument("--loss", type=float, default=0.01, help="mean packet loss rate")
    ap.add_argument("--burst", type=float, default=1.0, help="mean burst length (1 = independent losses)")
    ap.add_argument("--segment-blocks", type=int, default=256)
    ap.add_argument("--byte-limit", type=int, help="truncate segments (lossy compression)")
    ap.add_argument("--packet-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="trials")
    args = ap.parse_args()

    if args.fits:
        paths = sorted(Path(args.fits).glob("*.fits"))[: args.frames]
        loaded = [prep.load_hmi_int(p, scale=args.scale, crop=args.crop) for p in paths]
        images = [l[0] for l in loaded]
        valid = [l[1] for l in loaded]
        info = loaded[0][2]
        sources = [str(p) for p in paths]
    else:
        content = "MAGNETOGRAM" if args.synthetic == "magnetogram" else "CONTINUUM INTENSITY"
        conv = [prep.to_int(f, scale=args.scale, content=content)
                for f in synthetic.sequence(args.frames, args.size, args.synthetic, seed=args.seed)]
        images, valid, info = [c[0] for c in conv], [c[1] for c in conv], conv[0][2]
        sources = [f"synthetic:{args.synthetic}:{args.size}"] * len(images)

    if args.loss <= 0:
        channel = NoLoss()
    elif args.burst <= 1:
        channel = Bernoulli(args.loss)
    else:
        channel = GilbertElliott.from_mean(args.loss, args.burst)

    trials = simulate_sequence(
        images, codec=CodecParams(args.segment_blocks, args.byte_limit), channel=channel,
        seed=args.seed, max_packet_len=args.packet_len, valid=valid,
        extra_meta={"prep": info.to_dict()},
    )
    out = Path(args.out) / args.name
    for i, (t, s) in enumerate(zip(trials, sources)):
        t.meta["source"] = s
        t.save(out / f"frame_{i:02d}")
        print(f"frame {i}: {t.meta['packet_loss_rate']:.3%} packets lost, "
              f"{t.meta['blocks_lost']} blocks lost, {t.meta['blocks_truncated']} truncated, "
              f"{t.meta['pixels_lost_fraction']:.2%} pixels affected")
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))


if __name__ == "__main__":
    main()
