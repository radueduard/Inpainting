"""Loading SDO/HMI FITS files with astropy and converting them to integers for CCSDS 122."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from astropy.io import fits

# Physical units per integer step, chosen by the CONTENT keyword of HMI headers.
DEFAULT_SCALE = {
    "MAGNETOGRAM": 0.1,            # Gauss
    "DOPPLERGRAM": 1.0,            # m/s
    "CONTINUUM INTENSITY": 1.0,    # DN/s
}

# Scales for the PMI-like products (scripts/download_pmi_like.py), keyed by the
# product prefix of the file name. Chosen well below each product's noise level.
PRODUCT_SCALE = {
    "icnt": 1.0,     # DN/s
    "blos": 0.1,     # G
    "vlos": 0.5,     # m/s
    "bmag": 0.5,     # G
    "binc": 0.01,    # degrees
    "bazi": 0.01,    # degrees
}


def product_of(path: str | Path) -> str | None:
    name = Path(path).name.split("_")[0]
    return name if name in PRODUCT_SCALE else None


@dataclass
class PrepInfo:
    scale: float        # physical = int * scale + offset
    offset: float
    fill: int           # integer written to invalid (NaN / off-disk) pixels
    content: str

    def to_dict(self):
        return asdict(self)


def load_fits(path: str | Path) -> tuple[np.ndarray, fits.Header]:
    """Return (data, header) of the first 2-D image HDU.

    JSOC delivers Rice tile-compressed files (CompImageHDU in extension 1);
    astropy decompresses them transparently.
    """
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim == 2:
                return np.asarray(hdu.data, dtype=np.float64), hdu.header.copy()
    raise ValueError(f"no 2-D image in {path}")


def disk_geometry(header: fits.Header) -> tuple[float, float, float]:
    """(x0, y0, radius) of the solar disk in 0-based pixel coordinates."""
    x0 = header["CRPIX1"] - 1 - header.get("CRVAL1", 0.0) / header["CDELT1"]
    y0 = header["CRPIX2"] - 1 - header.get("CRVAL2", 0.0) / header["CDELT2"]
    if "R_SUN" in header:
        r = float(header["R_SUN"])
    else:
        r = float(header["RSUN_OBS"]) / float(header["CDELT1"])
    return x0, y0, r


def disk_mask(header: fits.Header, shape: tuple[int, int], margin: float = 0.0) -> np.ndarray:
    """True on the solar disk (radius reduced by `margin` pixels)."""
    x0, y0, r = disk_geometry(header)
    yy, xx = np.indices(shape)
    return (xx - x0) ** 2 + (yy - y0) ** 2 <= (r - margin) ** 2


def mu_map(header: fits.Header, shape: tuple[int, int]) -> np.ndarray:
    """mu = cos(heliocentric angle) on the disk, NaN outside."""
    x0, y0, r = disk_geometry(header)
    yy, xx = np.indices(shape)
    rho2 = ((xx - x0) ** 2 + (yy - y0) ** 2) / r ** 2
    with np.errstate(invalid="ignore"):
        return np.where(rho2 <= 1, np.sqrt(1 - rho2), np.nan)


def to_int(data: np.ndarray, scale: float | None = None, offset: float = 0.0,
           fill: int = 0, content: str = "", dtype=np.int32) -> tuple[np.ndarray, np.ndarray, PrepInfo]:
    """Quantize physical data to integers.

    Returns (int_image, valid_mask, info). NaNs become `fill`.
    """
    if scale is None:
        scale = DEFAULT_SCALE.get(content.upper(), 1.0)
    valid = np.isfinite(data)
    q = np.full(data.shape, fill, dtype=dtype)
    q[valid] = np.round((data[valid] - offset) / scale).astype(dtype)
    return q, valid, PrepInfo(scale, offset, fill, content)


def from_int(q: np.ndarray, info: PrepInfo, valid: np.ndarray | None = None) -> np.ndarray:
    out = q.astype(np.float64) * info.scale + info.offset
    if valid is not None:
        out[~valid] = np.nan
    return out


def load_hmi_int(path: str | Path, scale: float | None = None, crop: int | None = None):
    """Convenience: FITS -> (int_image, valid_mask, info, header).

    `crop` takes a centred square of that size (handy for fast experiments).
    """
    data, header = load_fits(path)
    product = product_of(path)
    if scale is None and product:
        scale = PRODUCT_SCALE[product]
    if crop:
        h, w = data.shape
        y0, x0 = (h - crop) // 2, (w - crop) // 2
        data = data[y0:y0 + crop, x0:x0 + crop]
        header = header.copy()
        header["CRPIX1"] -= x0
        header["CRPIX2"] -= y0
    q, valid, info = to_int(data, scale=scale, content=product or str(header.get("CONTENT", "")))
    return q, valid, info, header
