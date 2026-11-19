"""Download a short HMI sequence from JSOC as FITS files.

JSOC exports require an e-mail address registered at
http://jsoc.stanford.edu/ajax/register_email.html

Example (6 magnetograms, 45 s apart):
    python scripts/download_hmi.py --series hmi.M_45s --start 2024.05.10_12:00:00 \
        --n 6 --email you@example.com
"""

import argparse
from pathlib import Path

import drms

SEGMENTS = {"hmi.M_": "magnetogram", "hmi.V_": "Dopplergram", "hmi.Ic_": "continuum"}
CADENCE = {"_45s": 45, "_720s": 720}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="hmi.M_45s")
    ap.add_argument("--start", required=True, help="TAI time, e.g. 2024.05.10_12:00:00")
    ap.add_argument("--n", type=int, default=6, help="number of consecutive frames")
    ap.add_argument("--cadence", type=int, help="seconds between frames (default: series cadence)")
    ap.add_argument("--email", required=True, help="JSOC-registered e-mail")
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()

    segment = next(v for k, v in SEGMENTS.items() if args.series.startswith(k))
    cadence = args.cadence or next(v for k, v in CADENCE.items() if args.series.endswith(k))
    query = f"{args.series}[{args.start}_TAI/{args.n * cadence}s@{cadence}s]{{{segment}}}"
    out = Path(args.out) / args.series
    out.mkdir(parents=True, exist_ok=True)

    print("query:", query)
    client = drms.Client(email=args.email)
    req = client.export(query, method="url", protocol="fits")
    req.wait()
    files = req.download(str(out))
    print(f"downloaded {len(files)} files to {out}")


if __name__ == "__main__":
    main()
