"""Downsampling of HMI images to PMI resolution (0.5"/px -> ~1"/px, 4096^2 -> 2048^2).

Default method: Fourier resampling (spectral truncation at the new Nyquist
frequency, scipy.signal.resample). It is the ideal band-limited decimation: no
aliasing and no attenuation of the frequencies that survive, unlike 2x2
binning (a box filter that both attenuates and aliases). Binning is kept as an
alternative for comparison.

Practical details
-----------------
* The FFT needs a finite, periodic image: off-disk NaNs are replaced by the
  nearest on-disk value before resampling (avoids ringing from the limb step)
  and masked again afterwards, slightly eroded.
* The vector field (bmag, binc, bazi) is resampled jointly through quantities
  that are smooth and free of the 180-degree ambiguity:
      B_L = B cos(gamma),   Q = B_T^2 cos(2 phi),   U = B_T^2 sin(2 phi)
  (Q, U mimic linear polarization in the weak-field regime), then converted
  back. Resampling the azimuth directly would be meaningless.

Sampling grids: Fourier resampling keeps new pixel j at old pixel 2j
(0-based); binning puts it at old pixel 2j + 0.5. Headers are updated to match.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from scipy import ndimage
from scipy.signal import resample

AXIAL_PRODUCTS = {"bazi"}
VECTOR_PRODUCTS = ("bmag", "binc", "bazi")


# --------------------------------------------------------------------------
# Arrays
# --------------------------------------------------------------------------

def fill_nearest(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace non-finite pixels by the nearest finite value. Returns (filled, valid)."""
    valid = np.isfinite(data)
    if valid.all():
        return data, valid
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return data[tuple(idx)], valid


def _valid_after(valid: np.ndarray, factor: int, mode: str) -> np.ndarray:
    if mode == "fourier":
        k = np.ones((2 * factor + 1,) * 2, bool)
        return ndimage.binary_erosion(valid, k, border_value=1)[::factor, ::factor]
    h, w = valid.shape
    return valid[: h - h % factor, : w - w % factor].reshape(
        h // factor, factor, w // factor, factor).all(axis=(1, 3))


def fourier_down(data: np.ndarray, factor: int = 2) -> np.ndarray:
    """Fourier-domain decimation by an integer factor; NaN outside the (eroded) valid region."""
    filled, valid = fill_nearest(np.asarray(data, np.float64))
    h, w = filled.shape
    out = resample(resample(filled, h // factor, axis=0), w // factor, axis=1)
    out[~_valid_after(valid, factor, "fourier")] = np.nan
    return out


def bin_down(data: np.ndarray, factor: int = 2, axial: bool = False) -> np.ndarray:
    """Block average; a block with any NaN is NaN. axial: angles in degrees modulo 180."""
    h, w = data.shape
    d = data[: h - h % factor, : w - w % factor].reshape(h // factor, factor, w // factor, factor)
    if not axial:
        return d.mean(axis=(1, 3))
    z = np.exp(2j * np.deg2rad(d)).mean(axis=(1, 3))
    return np.mod(np.rad2deg(np.angle(z)) / 2, 180.0)


def bin2(data: np.ndarray, axial: bool = False) -> np.ndarray:
    return bin_down(data, 2, axial)


def downsample(data: np.ndarray, product: str | None = None, factor: int = 2,
               method: str = "fourier") -> np.ndarray:
    """Downsample a scalar product. For the vector products use :func:`downsample_vector`."""
    if method == "bin":
        return bin_down(data, factor, axial=product in AXIAL_PRODUCTS)
    if product in AXIAL_PRODUCTS:
        raise ValueError("azimuth cannot be Fourier-resampled alone; use downsample_vector")
    return fourier_down(data, factor)


def downsample_vector(bmag, binc, bazi, factor: int = 2, method: str = "fourier"):
    """Joint downsampling of field strength [G], inclination and azimuth [deg]."""
    if method == "bin":
        return (bin_down(bmag, factor), bin_down(binc, factor), bin_down(bazi, factor, axial=True))
    g, phi = np.deg2rad(binc), np.deg2rad(bazi)
    bl = bmag * np.cos(g)
    bt2 = (bmag * np.sin(g)) ** 2
    bl_, q_, u_ = (fourier_down(x, factor) for x in (bl, bt2 * np.cos(2 * phi), bt2 * np.sin(2 * phi)))
    bt_ = np.hypot(q_, u_) ** 0.5
    b_ = np.hypot(bl_, bt_)
    inc_ = np.rad2deg(np.arctan2(bt_, bl_))
    azi_ = np.mod(np.rad2deg(np.arctan2(u_, q_)) / 2, 180.0)
    return b_, inc_, azi_


# --------------------------------------------------------------------------
# Headers and files
# --------------------------------------------------------------------------

def down_header(header: fits.Header, factor: int = 2, method: str = "fourier") -> fits.Header:
    """Update WCS / disk keywords. FITS pixel centres are 1-based integers.

    fourier: new 0-based j <-> old 0-based factor*j    -> CRPIX' = (CRPIX - 1)/f + 1
    bin:     new pixel covers old f*j .. f*j+f-1       -> CRPIX' = (CRPIX - 1 - (f-1)/2)/f + 1
    """
    h = header.copy()
    shift = 0.0 if method == "fourier" else (factor - 1) / 2
    for ax in ("1", "2"):
        if f"CRPIX{ax}" in h:
            h[f"CRPIX{ax}"] = (h[f"CRPIX{ax}"] - 1 - shift) / factor + 1
        if f"CDELT{ax}" in h:
            h[f"CDELT{ax}"] = h[f"CDELT{ax}"] * factor
    for k in ("R_SUN", "X0_MP", "Y0_MP"):
        if k in h:
            h[k] = h[k] / factor
    for k in ("SIMPLE", "XTENSION", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "PCOUNT", "GCOUNT",
              "EXTEND", "BSCALE", "BZERO", "BLANK", "CHECKSUM", "DATASUM", "ZIMAGE", "ZBITPIX",
              "ZNAXIS", "ZNAXIS1", "ZNAXIS2", "ZTILE1", "ZTILE2", "ZCMPTYPE", "ZNAME1", "ZVAL1",
              "ZNAME2", "ZVAL2", "ZQUANTIZ", "ZDITHER0", "TFIELDS", "TTYPE1", "TFORM1"):
        h.remove(k, ignore_missing=True, remove_all=True)
    h["HISTORY"] = f"hmi_infill: downsampled x{factor} to PMI-like resolution ({method})"
    return h


def bin2_header(header: fits.Header) -> fits.Header:
    return down_header(header, 2, "bin")


def _write(dst, data, header):
    fits.PrimaryHDU(np.asarray(data, np.float32), header=header).writeto(dst, overwrite=True)


def downsample_fits(src, dst, product: str | None = None, factor: int = 2, method: str = "fourier"):
    """Scalar product: FITS -> downsampled float32 FITS."""
    from .prep import load_fits, product_of

    data, header = load_fits(src)
    product = product or product_of(src)
    _write(dst, downsample(data, product, factor, method), down_header(header, factor, method))


def downsample_vector_fits(srcs: dict, dsts: dict, factor: int = 2, method: str = "fourier"):
    """srcs/dsts: {"bmag": path, "binc": path, "bazi": path}."""
    from .prep import load_fits

    loaded = {p: load_fits(srcs[p]) for p in VECTOR_PRODUCTS}
    out = downsample_vector(*(loaded[p][0] for p in VECTOR_PRODUCTS), factor=factor, method=method)
    for p, d in zip(VECTOR_PRODUCTS, out):
        _write(dsts[p], d, down_header(loaded[p][1], factor, method))


def bin2_fits(src, dst, product: str | None = None):
    downsample_fits(src, dst, product, 2, "bin")
