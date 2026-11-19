# hmi-infill

Signal infill for SDO/HMI images transmitted as CCSDS 133 Space Packets and compressed with CCSDS 122.

The project simulates the whole downlink so that every experiment has a ground truth:

```
FITS (astropy) ─► integers ─► CCSDS 122 encode ─► CCSDS 133 packets ─► channel (losses)
                                                                         │
 ground truth ◄── lossless decode            damaged image + masks ◄── depacketize + loss-tolerant decode
```

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -e ".[download,notebooks,dev]"
.venv/bin/python -m ipykernel install --user --name hmi-infill --display-name "Python (hmi-infill)"
.venv/bin/pytest
```

## Data

```bash
# needs an e-mail registered at http://jsoc.stanford.edu/ajax/register_email.html
.venv/bin/python scripts/download_hmi.py --series hmi.M_45s --start 2024.05.10_12:00:00 --n 6 --email you@example.com
```

Series: `hmi.M_45s` (magnetogram, G), `hmi.V_45s` (Dopplergram, m/s), `hmi.Ic_45s` (continuum), or `_720s` versions.
FITS files go to `data/raw/<series>/` (git-ignored).

### PMI-like dataset (ESA Vigil)

Six products per time step, from the HMI 720 s series (the only ones with the full-disk vector field):
`icnt`, `blos`, `vlos`, `bmag`, `binc`, `bazi`. 36-min cadence (closest multiple of 12 min to PMI's 30 min).

```bash
.venv/bin/python scripts/download_pmi_like.py --start 2024.05.09_08:36:00 --n 30 --cadence 36 --email you@example.com
.venv/bin/python scripts/downsample_pmi_like.py      # 2x2 binning -> ~1"/px, 2048x2048
```

Full resolution: `data/raw/pmi_like/<product>/`; PMI resolution: `data/raw/pmi_like_1as/<product>/`.
Binning averages 2x2 blocks (NaN if any pixel is off-disk); the azimuth is averaged modulo 180 deg. The header
WCS (`CRPIX`, `CDELT`, `R_SUN`) is updated, so `prep.disk_geometry` works on both.

## Generating trials

```bash
# real data, bursty channel (1 % mean loss, bursts of 5 packets), 1024x1024 centre crop
.venv/bin/python scripts/run_trials.py --fits data/raw/hmi.M_45s --crop 1024 --loss 0.01 --burst 5 --name M_ge1
# synthetic data (no download needed)
.venv/bin/python scripts/run_trials.py --synthetic magnetogram --size 1024 --loss 0.02 --name synth
```

Each frame is saved to `trials/<name>/frame_XX/` as `trial.npz` + `meta.json`:

| array | meaning |
|---|---|
| `truth` | decode without losses (equals `original` in lossless mode) |
| `original` | integer input image |
| `damaged` | decode of what arrived; coefficients of lost blocks set to 0 |
| `mask_core` | 8x8 pixel footprints of lost blocks |
| `mask_lost` | every pixel the inverse DWT links to a lost coefficient (much wider than `mask_core`) |
| `mask_uncertain` | pixels affected by lost **or** truncated coefficients |
| `block_status` | per block: 0 ok, 1 truncated, 2 lost |
| `bitplane` | per block and coefficient: lowest bit-plane received (0 = exact, 99 = nothing) |
| `coeff_est`, `coeff_lo`, `coeff_hi` | decoded DWT coefficients and the interval that must contain the true value |
| `valid` | finite / on-disk pixels |

`meta.json` holds the geometry, compression ratio, lost packet indices, segment/block statistics, channel and prep parameters.

## Layout

```
src/hmi_infill/
  prep.py          FITS loading (astropy), disk geometry, float <-> int conversion
  synthetic.py     synthetic HMI-like sequences for tests/demos
  ccsds122/
    dwt.py         integer 9/7-M DWT of CCSDS 122 + exact inverse-support propagation
    blocks.py      coefficient <-> block grouping, subband weights
    bpe.py         bit-plane encoder/decoder (numba); decoder survives truncation
    codec.py       image-level encode / loss-tolerant decode
  ccsds133/
    packet.py      Space Packet headers, packetize / depacketize
  channel.py       Bernoulli and Gilbert-Elliott packet-erasure channels
  pipeline.py      simulate / simulate_sequence -> Trial (save/load)
  metrics.py       RMSE, PSNR, SSIM, flux and variance errors inside a mask
  infill/
    base.py        InfillInput (frame + past/future references) and run()
    baselines.py   zero, Laplace, biharmonic, previous-frame
  viz.py           astropy-normalised plotting helpers
notebooks/         01 FITS exploration, 02 loss simulation, 03 infill baselines
scripts/           download_hmi.py, run_trials.py
tests/
```

## Adding an infill method

```python
from hmi_infill.infill import InfillInput, run

class MyMethod:
    name = "my_method"
    def __call__(self, x: InfillInput):
        # x.damaged, x.mask, x.past[0..], x.future[0..], x.trial.coeff_lo / coeff_hi ...
        return estimate

filled = run(MyMethod(), InfillInput.from_trials(trials, index=3, n_past=3, n_future=1))
```

`run` only lets the method change pixels inside `mask`.

## Simulation design notes

* **Segments → packets.** Each CCSDS 122 segment starts a new packet; long segments span several packets
  marked first/continuation/last. The secondary header carries image id and segment index, so a segment is
  identifiable even when its first packet is lost.
* **Loss semantics.** First packet of a segment lost → segment lost (no header / DC). A later packet lost →
  segment truncated at that point (the bitstream is sequential, later packets are useless). Truncated segments
  still give every coefficient at reduced precision.
* **Side information.** Image size and segment size are treated as known mission constants.
* **References are damaged too.** All frames go through the same channel, as one continuous packet stream.

### Differences from the CCSDS 122.0-B Blue Book

The DWT (integer 9/7-M), subband weights, block structure, segment independence, DC quantization, bit-plane
order and stage structure follow the standard. Not bit-exact:

* custom segment header layout;
* significance words sent uncoded (no variable-length word codes), so compression is a little worse;
* zigzag instead of the standard's mapping for DC / bit-depth differences.

These affect the compressed size (and so the number of packets per segment), not the loss behaviour.
