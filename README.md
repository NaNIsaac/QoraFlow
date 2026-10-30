# QoraFlow

[![Tests](https://github.com/NaNIsaac/QoraFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/NaNIsaac/QoraFlow/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE.md)

A pump–probe experiment asks how a sample changes after a laser pulse. The
laser excites the sample; an X-ray pulse records a diffraction image after a
chosen delay. Repeating this at many delays produces thousands of detector
frames, **not yet a plot of what changed over time**.

**QoraFlow turns those frames into a time series.** It separates shots taken
with and without the laser, corrects detector background and incoming-beam
intensity, and averages images at each delay. It then tracks how the intensity
and position of a selected diffraction peak change with time.

For PAL-XFEL data, the inputs are HDF5 files containing JUNGFRAU detector
images, laser-on/off flags, and QBPM beam-monitor readings. The outputs are
processed image arrays (`.npz`/`.mat`), plots, TIFF stacks, and a CSV of
peak measurements. More specific processing steps, including LSB quantization,
are documented in the [pipeline](docs/pipeline.md).

![Synthetic scan reduced by QoraFlow: averaged laser-off image with ROI, on minus off difference, ROI intensity and centre of mass vs delay](https://raw.githubusercontent.com/NaNIsaac/QoraFlow/main/docs/pipeline_example.png)

**What to look at:** the first panels show the reference diffraction image and
the change when the laser is on. The plots track the selected peak's intensity
and centre position across 15 delays. The input is **synthetic**: after time
zero, the simulated peak becomes 25% weaker and moves 1.5 detector pixels.
The example adds Poisson noise, detector background, and beam-intensity
fluctuations. [Figure script](examples/readme_figure.py).

The repository ships no measured data. Examples use generated synthetic data;
the paths in `config.yaml` and the notebooks are placeholders. The dark frame
some notebooks and `scripts/temp` load is written by:

```bash
uv run python examples/synthetic_dark.py   # -> notebooks/dark.npy (514x1030 float32)
```

## Where to go next

- **Run the pipeline:** [installation](#install) and [configuration](#usage) show how to process a PAL-XFEL run.
- **Understand each correction:** [reduction steps](docs/pipeline.md) explain the dark frame, beam-monitor normalization, and output arrays.
- **Check the scope:** [validation](#validation) describes the tests and limitations. This repository does not distribute measured beamtime data.

## Install

Requires Python 3.11-3.13 and [uv](https://docs.astral.sh/uv/).

**1. Clone the repository**

```bash
git clone https://github.com/NaNIsaac/QoraFlow.git
cd QoraFlow
```

**2. Install the dependencies**

<!-- test -->
```bash
uv sync --python 3.13
```

**3. Check the install**

Runs the tests; it exits non-zero if anything is broken.

<!-- test -->
```bash
uv run pytest -q
```

Run the scripts from the repository root with `src` on the path (shown below).

## Usage

Edit `config.yaml` in the repository root. The main fields:

```yaml
runs: [109]
path:
  log_dir: ./logs
  load_dir: /path/to/raw_data/h5/type=raw   # contains run=XXX/scan=YYY/p*.h5
  analysis_dir: /path/to/analysis
  raw_dark_dir: /path/to/dark/run=171/scan=001
  save_dark_dir: ${path.analysis_dir}/dark_images/dark.npy
  output_dir: ${path.analysis_dir}/output_data
  mat_dir: ${path.analysis_dir}/mat_files
  processed_dir: ${path.analysis_dir}/processed_data
param:
  merge_num: 7          # HDF5 files (repeats) merged per dataset
  chunk_size: 100
  pump_setting: 60HZ
  hutch: eh1
  detector: jungfrau2
  xray: HX
  sdd: 1.3              # sample-detector distance [m]
  dps: 7.5e-05          # detector pixel size [m]
  beam_energy: 9        # keV
  x1: 0                 # ROI (pixels) used by analyzing_main.py
  x2: 1
  y1: 2
  y2: 3
```

`${path.x}` placeholders are resolved when the config is loaded. Then, from the repository root:

```bash
export PYTHONPATH=src
uv run python scripts/make_dark.py          # stack dark frames -> save_dark_dir (averaged when subtracted)
uv run python scripts/integrating_main.py   # raw HDF5 -> processed .npz / .mat
uv run python scripts/analyzing_main.py     # ROI intensity and COM vs delay
```

`integrating_main.py` writes one file per preprocessor to
`processed_dir/run=XXX/scan=YYY/`: `run=XXXX_scan=YYYY_LND.npz` (LSB
quantization, QBPM normalization, dark subtraction) and `..._raw.npz`, each with
`pon`, `poff` and `delay` arrays (`pon_delay`/`poff_delay` if the two differ),
plus `.mat` files per array in `mat_dir`. The QBPM-normalization ROI is picked
automatically around the brightest region of the middle file.

`analyzing_main.py` opens a Qt window to select the ROI and reads scan 1 only,
from `run=XXXX_scan=0001_LND.npz` (change `suffix` in the script to `raw` for
the unprocessed images).

The raw files follow the PAL-XFEL layout:

```
p0110.h5
├── detector/eh1/jungfrau2/image
├── metadata
├── qbpm/eh1/qbpm1
├── oh, oxc, pd
```

`PalXFELLoader` reads the frames from `image/block0_values` and their
timestamps from `image/block0_items`, sums `qbpm1/waveforms.ch1..4/block0_values`
per shot, and takes `metadata` (a pandas table indexed by timestamp) for the
laser-on flag `timestamp_info.RATE_<xray>_<pump_setting>` and the delay
(`th_value`, else `delay_value`; one delay per file). Shots are matched across
the three by timestamp.

To pick an ROI interactively (Qt window; `--roi` prints the selected rectangle):

```bash
uv run python -m QoraFlow.gui processed 109 --roi --log        # from run=0109_scan=0001_LND.npz (--suffix raw)
uv run python -m QoraFlow.gui raw 109 1 110 --roi --vmax 500   # from raw run/scan/p-file
```

## Validation

All test data is **synthetic**; no measured data ships with the repository.

- `examples/synthetic_pump_probe/` runs the reduction end to end on a scan with a known laser
  response and compares the ROI curves with it. `expected.json` keeps the truth (with tolerances)
  apart from a regression snapshot; `tests/test_example_synthetic_pump_probe.py` checks both.
- Known limit found by it: the ROI centre of mass reads peak shifts about 4 % low when the ROI holds
  a background ([#28](https://github.com/NaNIsaac/QoraFlow/issues/28)); that check is a strict
  expected failure until it is fixed.

## Structure

```
src/QoraFlow/
  integrator/     HDF5 loader (PalXFELLoader), integration, npz/mat savers
  preprocessor/   dark subtraction, QBPM normalization, LSB quantization
  analyzer/       ROI intensity / center-of-mass analysis and figures
  gui/            Qt ROI selector
  config/         YAML config with pydantic validation
  datacheck/      raw-file size checks and MAD outlier detection
scripts/          entry-point scripts (run from repo root)
notebooks/        exploratory analysis notebooks
examples/         synthetic data generators (README figure, dark frame)
```

## Credits

QoraFlow is a refactor of the earlier XFEL_data codebase written by Kim Jooheun ([@paulpeterkim](https://github.com/paulpeterkim)), 2021-2023.

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE.md](LICENSE.md).
