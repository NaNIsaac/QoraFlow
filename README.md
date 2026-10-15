# QoraFlow

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

The package is not installed into the environment, so run the scripts with `src` on the path (shown below).

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
```

`${path.x}` placeholders are resolved when the config is loaded. Then, from the repository root:

```bash
export PYTHONPATH=src
uv run python scripts/make_dark.py          # average dark frames -> save_dark_dir
uv run python scripts/integrating_main.py   # raw HDF5 -> processed .npz / .mat
uv run python scripts/analyzing_main.py     # ROI intensity and COM vs delay
```

The raw files follow the PAL-XFEL layout:

```
p0110.h5
├── detector/eh1/jungfrau2/image
├── metadata
├── qbpm/eh1/qbpm1
├── oh, oxc, pd
```

## Structure

```
src/QoraFlow/
  integrator/     HDF5 loader (PalXFELLoader), integration, npz/mat savers
  preprocessor/   dark subtraction, QBPM normalization, LSB quantization
  analyzer/       ROI intensity / center-of-mass analysis and figures
  gui/            Qt ROI selector
  config/         YAML config with pydantic validation
scripts/          entry-point scripts (run from repo root)
notebooks/        exploratory analysis notebooks
```

## Credits

QoraFlow is a refactor of the earlier XFEL_data codebase written by Kim Jooheun ([@paulpeterkim](https://github.com/paulpeterkim)), 2021-2023.

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE.md](LICENSE.md).
