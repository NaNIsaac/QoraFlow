# Synthetic pump-probe scan

A known laser response is written into synthetic PAL-XFEL run files and reduced by QoraFlow's own
scripts. This shows what the reduction recovers on a scan whose answer is known. No measured data
is needed.

```bash
PYTHONPATH=src uv run python examples/synthetic_pump_probe/run.py   # prints injected vs measured, writes result.json
uv run pytest -q tests/test_example_synthetic_pump_probe.py
```

Checked with Python 3.13 (QoraFlow does not install on 3.14 yet, QoraFlow#25).

## What it does

1. **Sample.** One Gaussian Bragg peak (sigma 3 px, 30 photons in the centre pixel, 9.8 keV
   photons) on a 96 x 128 detector, with a dark pedestal, Poisson noise and QBPM fluctuations
   (sigma 15 %). The generator is `examples/readme_figure.py`.
2. **Pump.** Shots alternate laser on and off. For delay t > 0 the laser-on peak loses
   `0.25 (1 - exp(-t / 1.5 ps))` of its intensity and moves `1.5 (1 - exp(-t / 3 ps))` px along x.
   Delays -1, 0, 2, 6 and 10 ps, two files per delay, 800 shots per file.
3. **Reduction.** `scripts/make_dark.py` and `scripts/integrating_main.py` run unchanged on the files
   (dark subtraction, QBPM normalization, laser on/off split, averaging per delay). Then
   `DataAnalyzer.analyze_by_roi` measures ROI mean intensity and centre of mass on the ROI that
   `auto_roi` picks, as in the README figure.

## Result at seed 0

| delay (ps) | on/off ratio, injected | measured | x shift (px), injected | measured |
|---|---|---|---|---|
| -1 | 1.000 | 1.000 | 0.000 | 0.000 |
| 0 | 1.000 | 0.987 | 0.000 | 0.014 |
| 2 | 0.816 | 0.798 | 0.730 | 0.689 |
| 6 | 0.755 | 0.739 | 1.297 | 1.241 |
| 10 | 0.750 | 0.743 | 1.446 | 1.366 |

1 px is 5.73e-4 1/Angstrom at the config geometry (sdd 1.3 m, 75 um pixels, 9.8 keV).

**The x shift reads about 4 % low.** At seeds 1, 2 and 3 the measured/injected shift ratio was
0.945 to 0.98 at every delay, so this is systematic, not noise. The ROI centre of mass includes the
photon floor and any residual background, which pull it toward the ROI centre; this is the likely
reason but it was not isolated. The on/off ratio scatters by up to 0.03 between seeds with no
consistent sign.

## What the test checks

`expected.json` holds two sets of numbers:

- **`truth`**: the injected response with a tolerance for each quantity: on/off ratio ± 0.04,
  x shift ± 0.15 px (10 % of the largest shift, which covers the 4 % bias above), y shift and
  laser-off drift ± 0.05 px, laser-off intensity ± 0.03. A test also checks that these values are
  the ones the generator injects.
- **`regression`**: today's values at seed 0 (ratio ± 0.005, shift ± 0.02 px, the ROI and the
  pixel-to-q factor), so any change in the reduction shows up.

It also checks that the response starts only after time zero.

## Limitations

- Synthetic data. One peak, a shift along x only, no change of peak width, no detector gain
  mapping or JUNGFRAU gain switching, and a QBPM that is exactly proportional to the incident flux.
- The synthetic files carry identical timestamps in every dataset, so the example does not test how
  PalXFELLoader pairs frames with QBPM and pump state (QoraFlow#19).
- One seed in the regression snapshot, five delays.
