# QoraFlow

[![Python 3.11-3.13](https://img.shields.io/badge/python-3.11--3.13-blue)](https://github.com/NaNIsaac/QoraFlow/blob/main/pyproject.toml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/NaNIsaac/QoraFlow/blob/main/LICENSE.md)

**Reduce PAL-XFEL pump-probe X-ray diffraction runs to laser-on/off images per delay, then to ROI intensity and peak shift vs delay.**

![Synthetic scan reduced by QoraFlow: averaged laser-off image with ROI, on minus off difference, ROI intensity and centre of mass vs delay](https://raw.githubusercontent.com/NaNIsaac/QoraFlow/main/docs/pipeline_example.png)

The data here are **synthetic**: 15 delays (-2 to 10 ps) × 2 repeat files × 100
shots of a 96×128 detector, written in the PAL-XFEL HDF5 layout the loader
reads. Simulated: one Gaussian Bragg peak, Poisson photon noise, a per-pixel
dark pedestal with read noise, shot-to-shot QBPM fluctuations that scale the
signal, alternating laser on/off shots, and for laser-on shots after t=0 a 25%
intensity drop (1.5 ps) and a 1.5-pixel peak shift (3 ps). Everything after
that is QoraFlow's own code, unchanged: `scripts/make_dark.py`,
`scripts/integrating_main.py` (auto ROI, LSB quantization, QBPM normalization,
dark subtraction, merge of the 2 repeats, per-delay averaging; `LND` output)
and `DataAnalyzer.analyze_by_roi`. Regenerate with
`PYTHONPATH=src uv run python examples/readme_figure.py`; all data go to a
temporary directory.

## Quick start

```bash
git clone https://github.com/NaNIsaac/QoraFlow.git
cd QoraFlow
uv sync --python 3.13
# point config.yaml at your raw data, then
export PYTHONPATH=src
uv run python scripts/make_dark.py
uv run python scripts/integrating_main.py
```

Python scripts and a small library for reducing time-resolved pump-probe X-ray diffraction data from PAL-XFEL (JUNGFRAU detector images and QBPM intensity monitor, stored as HDF5).

QoraFlow reads raw run/scan HDF5 files, splits shots into laser-on and laser-off, applies preprocessing (dark subtraction, QBPM normalization, LSB quantization), averages images per delay and writes the result as `.npz` and `.mat` files. A second script measures ROI intensity and center of mass as a function of delay and saves figures, TIFF stacks and a CSV.

It is research code used in our group; interfaces change without notice.

## Install

Requires Python 3.11-3.13 (`tables` has no wheels for 3.14 yet) and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/NaNIsaac/QoraFlow.git
cd QoraFlow
uv sync --python 3.13
```

The package is not installed into the environment, so run the scripts with `src` on the path (shown below). Tests: `uv run pytest`.

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
examples/         README figure generator (synthetic data)
```

## Credits

QoraFlow is a refactor of the earlier XFEL_data codebase written by Kim Jooheun ([@paulpeterkim](https://github.com/paulpeterkim)), 2021-2023.

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE.md](LICENSE.md).
