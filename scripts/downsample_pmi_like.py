"""Downsample the PMI-like HMI products 2x to PMI resolution (~1"/px, 2048x2048).

Reads  data/raw/pmi_like/<product>/<product>_<time>.fits
Writes data/raw/pmi_like_1as/<product>/<product>_<time>.fits   (float32)

Method "fourier" (default): spectral truncation, off-disk filled with the
nearest on-disk value before the FFT and masked again after. The vector field
(bmag, binc, bazi) is resampled jointly through B_L and the Stokes-like
B_T^2 (cos 2phi, sin 2phi), which avoids the azimuth ambiguity.
Method "bin": 2x2 block average (azimuth averaged modulo 180 deg).

Example:
    python scripts/downsample_pmi_like.py
    python scripts/downsample_pmi_like.py --method bin --dst data/raw/pmi_like_1as_bin
"""

import argparse
from pathlib import Path

from hmi_infill.resample import VECTOR_PRODUCTS, downsample_fits, downsample_vector_fits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/raw/pmi_like")
    ap.add_argument("--dst", default="data/raw/pmi_like_1as")
    ap.add_argument("--method", choices=["fourier", "bin"], default="fourier")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    products = sorted(p.name for p in src.iterdir() if p.is_dir() and not p.name.startswith("_"))
    for p in products:
        (dst / p).mkdir(parents=True, exist_ok=True)

    def todo(target):
        return args.overwrite or not target.exists()

    for p in (p for p in products if p not in VECTOR_PRODUCTS):
        files = sorted((src / p).glob("*.fits"))
        n = 0
        for f in files:
            if todo(dst / p / f.name):
                downsample_fits(f, dst / p / f.name, product=p, method=args.method)
                n += 1
        print(f"{p}: {n} written, {len(files)} total")

    if all(p in products for p in VECTOR_PRODUCTS):
        times = sorted(f.name.split("_", 1)[1] for f in (src / "bmag").glob("*.fits"))
        n = 0
        for t in times:
            srcs = {p: src / p / f"{p}_{t}" for p in VECTOR_PRODUCTS}
            dsts = {p: dst / p / f"{p}_{t}" for p in VECTOR_PRODUCTS}
            if not all(s.exists() for s in srcs.values()):
                print(f"skip {t}: incomplete vector triplet")
                continue
            if any(todo(d) for d in dsts.values()):
                downsample_vector_fits(srcs, dsts, method=args.method)
                n += 1
        print(f"bmag/binc/bazi: {n} time steps written, {len(times)} total")


if __name__ == "__main__":
    main()
