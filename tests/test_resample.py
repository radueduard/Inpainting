import numpy as np
import pytest
from astropy.io import fits

from hmi_infill import prep
from hmi_infill.resample import (bin2, down_header, downsample, downsample_fits, downsample_vector,
                                 fourier_down)


def test_bin2_mean_and_nan():
    d = np.arange(16, dtype=float).reshape(4, 4)
    d[0, 0] = np.nan
    b = bin2(d)
    assert b.shape == (2, 2) and np.isnan(b[0, 0])
    assert b[1, 1] == np.mean([10, 11, 14, 15])


def test_bin2_axial_wraps_at_180():
    b = bin2(np.array([[179.0, 1.0], [178.0, 2.0]]), axial=True)[0, 0]
    assert min(b, 180 - b) < 1e-9


def test_fourier_keeps_band_limited_signal_exactly():
    n = 128
    y, x = np.mgrid[:n, :n]
    f = np.cos(2 * np.pi * 5 * x / n) + 0.5 * np.sin(2 * np.pi * 12 * y / n)   # below new Nyquist (32)
    d = fourier_down(f)
    assert np.allclose(d, f[::2, ::2], atol=1e-10)


def test_fourier_removes_frequencies_above_new_nyquist():
    n = 128
    x = np.arange(n)[None, :].repeat(n, 0)
    f = np.cos(2 * np.pi * 40 * x / n)          # above new Nyquist: would alias with binning
    assert np.abs(fourier_down(f)).max() < 1e-10
    assert np.abs(bin2(f)).max() > 0.1


def test_fourier_disk_nan_handling():
    n = 128
    y, x = np.mgrid[:n, :n]
    disk = (x - 64) ** 2 + (y - 64) ** 2 < 50 ** 2
    f = np.where(disk, 1000 + 0.1 * x, np.nan)
    d = fourier_down(f)
    assert np.isnan(d[0, 0]) and np.isfinite(d[32, 32])
    assert np.nanmax(np.abs(d - (1000 + 0.1 * x[::2, ::2]))) < 1.0   # no limb ringing


def test_azimuth_alone_is_refused():
    with pytest.raises(ValueError):
        downsample(np.zeros((8, 8)), "bazi")


def test_vector_uniform_field_roundtrip():
    shape = (64, 64)
    b, inc, azi = np.full(shape, 800.0), np.full(shape, 60.0), np.full(shape, 175.0)
    b2, i2, a2 = downsample_vector(b, inc, azi)
    assert np.allclose(b2, 800) and np.allclose(i2, 60)
    assert np.allclose(a2, 175)


def test_vector_axial_average():
    # azimuths 179 and 1 alternate -> mean axis 0/180, not 90
    shape = (64, 64)
    azi = np.where(np.indices(shape).sum(0) % 2 == 0, 179.0, 1.0)
    _, _, a2 = downsample_vector(np.full(shape, 500.0), np.full(shape, 90.0), azi)
    assert np.all(np.minimum(a2, 180 - a2) < 1.5)


@pytest.mark.parametrize("method", ["fourier", "bin"])
def test_header_disk_centre(method):
    h = fits.Header({"CRPIX1": 2048.5, "CRPIX2": 2040.25, "CDELT1": 0.504, "CDELT2": 0.504,
                     "CRVAL1": 0.0, "CRVAL2": 0.0, "RSUN_OBS": 950.0, "R_SUN": 1885.0})
    n = down_header(h, 2, method)
    x0, y0, r = prep.disk_geometry(h)
    x1, y1, r1 = prep.disk_geometry(n)
    shift = 0.0 if method == "fourier" else 0.5
    assert np.isclose(x1, (x0 - shift) / 2) and np.isclose(y1, (y0 - shift) / 2)
    assert np.isclose(r1, r / 2) and np.isclose(n["CDELT1"], 1.008)


def test_downsample_fits(tmp_path):
    hdr = fits.Header({"CRPIX1": 8.5, "CRPIX2": 8.5, "CDELT1": 0.5, "CDELT2": 0.5, "RSUN_OBS": 3.0})
    data = np.random.default_rng(0).normal(size=(16, 16))
    src = tmp_path / "blos_20240509_083600.fits"
    fits.PrimaryHDU(data, header=hdr).writeto(src)
    downsample_fits(src, tmp_path / "out.fits")
    out, h = prep.load_fits(tmp_path / "out.fits")
    assert out.shape == (8, 8) and np.allclose(out, fourier_down(data), atol=1e-5)
    assert h["CDELT1"] == 1.0 and h["CRPIX1"] == 4.75
