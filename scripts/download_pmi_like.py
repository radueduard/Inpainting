"""Download an HMI sequence that mimics the data products of PMI (ESA Vigil).

PMI (Photospheric Magnetic field Imager) derives its products on board and
downlinks, at a nominal 30-min cadence, continuum intensity, line-of-sight
velocity and the vector field (strength, inclination, azimuth), plus the
line-of-sight field like SO/PHI. The closest HMI products are the 720 s series,
which are the only ones that include the full-disk vector field:

    PMI product   HMI series / segment
    icnt          hmi.Ic_720s{continuum}
    blos          hmi.M_720s{magnetogram}
    vlos          hmi.V_720s{Dopplergram}
    bmag          hmi.B_720s{field}
    binc          hmi.B_720s{inclination}
    bazi          hmi.B_720s{azimuth}      (180-degree ambiguity not resolved, as on board)

The 720 s series are slotted every 12 min, so the cadence must be a multiple of
12 min; 36 min is the closest to PMI's 30 min. Files end up in
<out>/<product>/<product>_<YYYYMMDD_HHMMSS>.fits.

JSOC exports need an e-mail registered at http://jsoc.stanford.edu/ajax/register_email.html

Example:
    python scripts/download_pmi_like.py --start 2024.05.09_08:36:00 --n 30 --cadence 36 --email you@example.com
"""

import argparse
import re
import shutil
from pathlib import Path

import drms

PRODUCTS = {
    "hmi.Ic_720s": {"continuum": "icnt"},
    "hmi.M_720s": {"magnetogram": "blos"},
    "hmi.V_720s": {"Dopplergram": "vlos"},
    "hmi.B_720s": {"field": "bmag", "inclination": "binc", "azimuth": "bazi"},
}


def check_quality(client, query_time):
    bad = {}
    for series in PRODUCTS:
        k = client.query(f"{series}[{query_time}]", key="T_REC,QUALITY")
        flagged = k[k.QUALITY != 0]
        if len(flagged):
            bad[series] = list(zip(flagged.T_REC, flagged.QUALITY.map(hex)))
        print(f"{series:12s} {len(k)} records, {len(flagged)} with QUALITY != 0")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="TAI time, e.g. 2024.05.09_08:36:00")
    ap.add_argument("--n", type=int, default=30, help="number of time steps")
    ap.add_argument("--cadence", type=int, default=36, help="minutes between steps (multiple of 12)")
    ap.add_argument("--email", required=True, help="JSOC-registered e-mail")
    ap.add_argument("--out", default="data/raw/pmi_like")
    ap.add_argument("--force", action="store_true", help="download even if some frames are flagged")
    ap.add_argument("--series", nargs="+", choices=list(PRODUCTS), default=list(PRODUCTS),
                    help="subset of HMI series to download")
    args = ap.parse_args()
    if args.cadence % 12:
        ap.error("cadence must be a multiple of 12 min for the 720 s series")

    duration = args.n * args.cadence          # JSOC's range end is exclusive
    qtime = f"{args.start}_TAI/{duration}m@{args.cadence}m"
    client = drms.Client(email=args.email)

    bad = check_quality(client, qtime)
    if bad and not args.force:
        raise SystemExit(f"flagged frames, choose another window or use --force: {bad}")

    out = Path(args.out)
    for series, segs in ((s, PRODUCTS[s]) for s in args.series):
        query = f"{series}[{qtime}]{{{','.join(segs)}}}"
        print("export:", query)
        req = client.export(query, method="url", protocol="fits")
        req.wait()
        tmp = out / "_tmp" / series
        tmp.mkdir(parents=True, exist_ok=True)
        files = req.download(str(tmp))
        for f in files.download:
            f = Path(f)
            # hmi.m_720s.20240509_083600_TAI.3.magnetogram.fits (camera number)
            # hmi.b_720s.20240509_083600_TAI.field.fits
            m = re.search(r"(\d{8}_\d{6})_TAI\.(?:\d+\.)?(\w+)\.fits$", f.name)
            if not m or m.group(2) not in segs:
                print(f"  WARNING: unrecognised file name, left in place: {f}")
                continue
            prod = segs[m.group(2)]
            dest = out / prod / f"{prod}_{m.group(1)}.fits"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), dest)
        print(f"  {len(files)} files")
    leftovers = list((out / "_tmp").rglob("*.fits"))
    if leftovers:
        print(f"WARNING: {len(leftovers)} files not sorted, kept in {out / '_tmp'}")
    else:
        shutil.rmtree(out / "_tmp", ignore_errors=True)
    for prod in sorted(p.name for p in out.iterdir() if p.is_dir()):
        print(f"{prod}: {len(list((out / prod).glob('*.fits')))} files")


if __name__ == "__main__":
    main()
