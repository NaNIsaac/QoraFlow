# Reduction pipeline

What happens to the data between the raw PAL-XFEL files and the ROI table, step by step, as the
code does it. The README's [Usage](../README.md#usage) says how to run the
scripts; this page says what they compute, so that a number in the output can be traced back.

All three scripts read `config.yaml` from the current directory. `${path.x}` placeholders in it
are resolved when it is loaded, and every field is validated (`QoraFlow.config`).

## 1. Dark frames: `scripts/make_dark.py`

Every `*.h5` file under `path.raw_dark_dir` is read with the same loader as the data (step 2) and
all of its frames are stacked. The stack is saved as it is to `path.save_dark_dir` (a `.npy`
file) and its mean is shown in a matplotlib window. The mean is taken later, when the dark is
subtracted.

## 2. Reading one raw file: `PalXFELLoader`

A raw file `p<N>.h5` holds the detector frames, the QBPM waveforms and a `metadata` table. With
`param.hutch` and `param.detector` from the config, the loader reads:

| what | where in the file |
|---|---|
| frames | `detector/<hutch>/<detector>/image/block0_values`, timestamps in `.../image/block0_items` |
| QBPM | `qbpm/<hutch>/qbpm1/waveforms.ch1` to `ch4`, `block0_values`, timestamps in `waveforms.ch1/axis1` |
| laser flag | `metadata` column `timestamp_info.RATE_<xray>_<pump_setting>` |
| delay | `metadata` column `th_value`, or `delay_value` if there is none |

- The QBPM value of a shot is the sum over the four channels and over all samples of their
  waveforms.
- Frames, QBPM values and metadata rows are joined on their timestamps; a shot missing from any
  of the three is dropped.
- A shot whose laser flag is missing is dropped with a warning, rather than read as laser-on.
  With `pump_setting: 0HZ` there is no flag column and every shot is laser-off.
- The file has one delay: the value in its first row.
- Negative pixel values are set to 0 when the frames are read.

## 3. Grouping and averaging: `CoreIntegrator`

`integrating_main.py` processes every scan directory of every run in `runs`. In a scan
directory the files are sorted by their number (`p0001.h5`, `p0002.h5`, ...) and taken in groups
of `param.merge_num`, the number of repeated files per delay. Within a group:

1. Each file is read in chunks of `param.chunk_size` shots, and each chunk is split into
   laser-on and laser-off shots.
2. Each preprocessor (step 4) is applied to each chunk, laser-on and laser-off separately.
3. The processed frames are summed per preprocessor and per delay (delays equal to 6 decimal
   places count as one) and divided by their number: one averaged laser-on and one averaged
   laser-off image per delay.

Groups do not share sums. A delay that appears in two groups gives two images with the same
delay in the output, so `merge_num` should match the number of repeats per delay.

## 4. Preprocessors

`integrating_main.py` runs two preprocessors on the same frames and saves both:

- `raw`: nothing; the frames are averaged as read.
- `LND`: three steps in this order, on each chunk:
  1. **LSB quantization.** Each pixel is rounded down to a multiple of 9.8, a constant in the
     code (`lsb_quantization`); it is not derived from `param.beam_energy`.
  2. **QBPM normalization with shot filtering** (`make_qbpm_roi_normalizer`). Shots whose QBPM
     value lies outside the chunk's mean ± 2 standard deviations are dropped. For the rest, the
     ratio of ROI sum to QBPM is formed, and shots whose ratio lies more than 0.3 standard
     deviations from the median ratio are dropped too. The kept frames are divided by their QBPM
     value and multiplied by the mean QBPM of the kept shots.
  3. **Dark subtraction** (`subtract_dark_background`): the mean dark frame (step 1, negative
     values set to 0) is subtracted, and negative results are set to 0.

The ROI used for the normalization is chosen without a window (`auto_roi`): the frames of the
middle file of the scan are summed, the brightest 10 % of pixels give an intensity-weighted
centroid, and the ROI is the 40 x 40 pixel box around it. The log prints it as
`Auto selected ROI`.

## 5. Integrated files

For each scan and each preprocessor (`LND`, `raw`):

- `path.processed_dir/run=XXX/scan=YYY/run=XXXX_scan=YYYY_<preprocessor>.npz`, with `pon` and
  `poff` (averaged images, one per delay, sorted by delay) and `delay`. If the laser-on and
  laser-off images do not cover the same delays, `pon_delay` and `poff_delay` replace `delay`. A
  run without a laser has only `poff`.
- `path.mat_dir/run=XXXX_scan=YYYY_<array>_<preprocessor>.mat`, one file per image stack, the
  stack in variable `data` with the two image axes moved to the front.

## 6. ROI analysis: `DataAnalyzer` and `scripts/analyzing_main.py`

`DataAnalyzer(npz_file)` loads the `pon`, `poff` and `delay` arrays (negative values set to 0).
`analyze_by_roi(roi, background=None, edge_width=3)` returns a table indexed by delay with, for
each of `poff` and `pon`:

- `<state>_intensity`: the mean of the ROI, divided by its value at the first delay;
- `<state>_com_x`, `<state>_com_y`: the ROI centre of mass, minus its value at the first delay,
  converted from pixels to |Δq| in 1/Å by `mul_delta_q`: (2π/λ)·arctan(`dps`/`sdd`) per pixel,
  with λ from `beam_energy`. This is the magnitude of the change of the scattering vector for a
  detector normal to the scattered beam, not the change of |q|.

`background="edge"` or `"plane"` subtracts a background estimated from the outer `edge_width`
pixels of the ROI (their median, or a least-squares plane through them) from each ROI image
before the centre of mass is taken; the intensities are not affected. The
[first run](index.md#first-run) shows what it changes on synthetic data, and the docstring of
`edge_background` says when each estimate fails.

`analyzing_main.py` reads scan 1 of each run from `run=XXXX_scan=0001_LND.npz` (the `suffix` in
the script selects `raw` instead), opens a Qt window to pick the ROI, calls `analyze_by_roi`
without a background, and writes to `path.output_dir/run=XXX/scan=YYY/<time>/`, where `<time>` is
when the run's analysis started (`YYYYmmdd_HHMMSS`), so a second ROI does not overwrite the first:

- `data.csv`, the table above;
- `poff.tif`, `pon.tif` and `roi_poff.tif`, `roi_pon.tif`, the image stacks and their ROI crops;
- `log_image.png` (the summed image with the ROI drawn), `delay-intensity.png`,
  `delay-intensity_diff.png`, `delay-com.png` and `delay-com_diff.png`.
