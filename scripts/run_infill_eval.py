"""Evaluate infill methods on the PMI-like sequences.

For every product: simulate the transmission of all frames (one continuous
packet stream through the channel), save the trials, run each method on every
frame that lost data, and write one CSV row per (product, frame, method).

Example:
    python scripts/run_infill_eval.py --data data/raw/pmi_like_1as --loss 0.01 --burst 5 --name pmi_1as_ge1
"""

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PERIOD = {"bazi": 180.0}          # wrap errors of the ambiguous azimuth
FIELDS = ["product", "frame", "method", "n_pixels", "pixels_frac", "rmse", "mae", "max_abs", "bias",
          "psnr", "ssim_bbox", "std_ratio", "seconds"]


def methods(n_past, n_future):
    from hmi_infill.infill import DMVE, FSE, Biharmonic, PreviousFrame
    # (name, method, needs): needs = None (spatial), "past", or "any" reference frame
    return [("biharmonic", Biharmonic(), None), ("fse", FSE(), None),
            ("previous_frame", PreviousFrame(), "past"), ("dmve", DMVE(), "any")]


def run_product(job):
    from hmi_infill import prep
    from hmi_infill.ccsds122 import CodecParams
    from hmi_infill.channel import Bernoulli, GilbertElliott
    from hmi_infill.infill import InfillInput, run
    from hmi_infill.metrics import evaluate
    from hmi_infill.pipeline import simulate_sequence

    product, a = job
    files = sorted(Path(a["data"], product).glob("*.fits"))[: a["frames"]]
    loaded = [prep.load_hmi_int(f) for f in files]
    images, valid, info = [l[0] for l in loaded], [l[1] for l in loaded], loaded[0][2]
    channel = Bernoulli(a["loss"]) if a["burst"] <= 1 else GilbertElliott.from_mean(a["loss"], a["burst"])
    # different seed per product so that losses are independent between products
    seed = a["seed"] + sum(map(ord, product))
    trials = simulate_sequence(images, codec=CodecParams(a["segment_blocks"]), channel=channel, seed=seed,
                               max_packet_len=a["packet_len"], valid=valid, extra_meta={"prep": info.to_dict()})
    out = Path(a["out"], product)
    for i, (t, f) in enumerate(zip(trials, files)):
        t.meta["source"] = str(f)
        t.save(out / f"frame_{i:02d}")

    rows = []
    for i, t in enumerate(trials):
        if t.meta["blocks_lost"] == 0:
            continue
        x = InfillInput.from_trials(trials, i, n_past=a["n_past"], n_future=a["n_future"], mask="mask_lost")
        for name, m, needs in methods(a["n_past"], a["n_future"]):
            if (needs == "past" and not x.past) or (needs == "any" and not (x.past or x.future)):
                continue
            t0 = time.time()
            est = run(m, x)
            dt = time.time() - t0
            r = evaluate(est, t.truth, x.mask, valid=t.valid, scale=info.scale, period=PERIOD.get(product))
            rows.append({"product": product, "frame": i, "method": name,
                         "pixels_frac": float((x.mask & t.valid).sum() / t.valid.sum()), "seconds": dt,
                         **{k: r.get(k) for k in FIELDS if k in r}})
        print(f"{product} frame {i}: done ({t.meta['blocks_lost']} blocks lost)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/raw/pmi_like_1as")
    ap.add_argument("--products", nargs="+", default=["icnt", "blos", "vlos", "bmag", "binc", "bazi"])
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--loss", type=float, default=0.01)
    ap.add_argument("--burst", type=float, default=5.0)
    ap.add_argument("--segment-blocks", type=int, default=256)
    ap.add_argument("--packet-len", type=int, default=1024)
    ap.add_argument("--n-past", type=int, default=3)
    ap.add_argument("--n-future", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    a = vars(args) | {"out": f"trials/{args.name}"}
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(run_product, [(p, a) for p in args.products]))
    rows = [r for rs in results for r in rs]

    res_dir = Path("results")
    res_dir.mkdir(exist_ok=True)
    csv_path = res_dir / f"{args.name}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {csv_path}: {len(rows)} rows\n")
    meths = list(dict.fromkeys(r["method"] for r in rows))
    print(f"{'product':8s} {'frames':>6s} " + " ".join(f"{m:>15s}" for m in meths) + "   (median RMSE in physical units)")
    for p in args.products:
        pr = [r for r in rows if r["product"] == p]
        frames = sorted({r["frame"] for r in pr})
        med = {m: np.median([r["rmse"] for r in pr if r["method"] == m] or [np.nan]) for m in meths}
        print(f"{p:8s} {len(frames):6d} " + " ".join(f"{med[m]:15.3f}" for m in meths))


if __name__ == "__main__":
    main()
