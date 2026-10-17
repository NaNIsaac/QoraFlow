"""Regenerate docs/pipeline_example.png, the figure at the top of the README.

    PYTHONPATH=src uv run python examples/readme_figure.py

Writes a synthetic PAL-XFEL scan and dark run in the PalXFELLoader layout to a
temporary directory, runs scripts/make_dark.py and scripts/integrating_main.py on
them unchanged, then measures the ROI with DataAnalyzer and plots the result.
Nothing but the PNG is written inside the repository.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import patches

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "pipeline_example.png"

H, W = 96, 128
PEAK_Y, PEAK_X, PEAK_SIGMA = 44.0, 70.0, 3.0
PEAK_PHOTONS = 30.0  # mean photons per shot in the peak's centre pixel
PHOTON_KEV = 9.8
SHOTS = 100  # per file, alternating laser on/off
REPEATS = 2  # files per delay, merged by merge_num
DELAYS = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10]  # ps
PUMP_COLUMN = "timestamp_info.RATE_HX_30HZ"

rng = np.random.default_rng(0)
pedestal = rng.normal(40, 4, (H, W)).astype(np.float32)
yy, xx = np.indices((H, W))


def laser_response(delay: float) -> tuple[float, float]:
    """Simulated pump effect: (intensity factor, peak shift in x [px])."""
    if delay <= 0:
        return 1.0, 0.0
    return 1 - 0.25 * (1 - np.exp(-delay / 1.5)), 1.5 * (1 - np.exp(-delay / 3))


def write_h5(file: Path, images: np.ndarray, qbpm: np.ndarray, pump: np.ndarray, delay: float, t0: int) -> None:
    timestamps = np.arange(t0, t0 + len(images), dtype=np.int64)
    metadata = pd.DataFrame({PUMP_COLUMN: pump, "delay_value": delay}, index=timestamps)
    metadata.to_hdf(file, key="metadata", mode="w")
    with h5py.File(file, "a") as hf:
        image = hf.create_group("detector/eh1/jungfrau2/image")
        image["block0_items"] = timestamps
        image["block0_values"] = images.astype(np.float32)
        for ch in range(1, 5):
            wave = hf.create_group(f"qbpm/eh1/qbpm1/waveforms.ch{ch}")
            wave["axis1"] = timestamps
            wave["block0_values"] = np.repeat(qbpm[:, None] / 32, 8, axis=1).astype(np.float32)


def shots(n: int, pump: np.ndarray, delay: float) -> tuple[np.ndarray, np.ndarray]:
    qbpm = rng.normal(1.0, 0.15, n).clip(0.4)
    factor, shift = laser_response(delay)
    images = np.empty((n, H, W), np.float32)
    for i in range(n):
        f, s = (factor, shift) if pump[i] else (1.0, 0.0)
        peak = PEAK_PHOTONS * f * np.exp(-((yy - PEAK_Y) ** 2 + (xx - PEAK_X - s) ** 2) / (2 * PEAK_SIGMA**2))
        photons = rng.poisson(qbpm[i] * (peak + 0.02))
        images[i] = photons * PHOTON_KEV + pedestal + rng.normal(0, 1.5, (H, W))
    return images, qbpm


def write_data(root: Path) -> None:
    scan_dir = root / "raw" / "run=001" / "scan=001"
    dark_dir = root / "dark" / "run=999" / "scan=001"
    scan_dir.mkdir(parents=True)
    dark_dir.mkdir(parents=True)

    pump = np.arange(SHOTS) % 2 == 0
    t0 = 1_000_000
    for i, delay in enumerate(np.repeat(DELAYS, REPEATS), start=1):
        images, qbpm = shots(SHOTS, pump, delay)
        write_h5(scan_dir / f"p{i:04d}.h5", images, qbpm, pump, delay, t0)
        t0 += SHOTS

    no_pump = np.zeros(SHOTS, bool)
    for i in (1, 2):
        dark = pedestal + rng.normal(0, 1.5, (SHOTS, H, W))
        write_h5(dark_dir / f"p{i:04d}.h5", dark, np.ones(SHOTS), no_pump, np.nan, t0)
        t0 += SHOTS

    config = {
        "runs": [1],
        "path": {
            "log_dir": str(root / "logs"),
            "load_dir": str(root / "raw"),
            "analysis_dir": str(root / "analysis"),
            "raw_dark_dir": str(dark_dir),
            "save_dark_dir": "${path.analysis_dir}/dark_images/dark.npy",
            "output_dir": "${path.analysis_dir}/output_data",
            "mat_dir": "${path.analysis_dir}/mat_files",
            "processed_dir": "${path.analysis_dir}/processed_data",
        },
        "param": {
            "merge_num": REPEATS,
            "chunk_size": 100,
            "pump_setting": "30HZ",
            "hutch": "eh1",
            "detector": "jungfrau2",
            "xray": "HX",
            "sdd": 1.3,
            "dps": 7.5e-05,
            "beam_energy": PHOTON_KEV,
        },
    }
    (root / "analysis" / "dark_images").mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


def run_script(name: str, root: Path) -> None:
    env = os.environ | {"PYTHONPATH": str(REPO / "src"), "MPLBACKEND": "Agg"}
    subprocess.run([sys.executable, str(REPO / "scripts" / name)], cwd=root, env=env, check=True)


def plot(root: Path) -> None:
    from QoraFlow.analyzer.core import DataAnalyzer
    from QoraFlow.config import ConfigManager
    from QoraFlow.gui.select_roi import auto_roi

    ConfigManager.initialize(root / "config.yaml")
    config = ConfigManager.load_config()
    # integrating_main.py picks its QBPM-normalisation ROI with auto_roi; use the same one here
    roi = auto_roi(root / "raw" / "run=001" / "scan=001", config)
    npz = config.path.processed_dir / "run=001" / "scan=001" / "run=0001_scan=0001_LND.npz"
    analyzer = DataAnalyzer(npz)
    df = analyzer.analyze_by_roi(roi)
    print(f"ROI {roi}, {len(df)} delays")
    print(df.round(4).to_string())

    y1, y2, x1, x2 = roi.to_tuple()
    poff_mean = analyzer.poff_images.mean(axis=0)
    diff = roi.slice(analyzer.pon_images[-1] - analyzer.poff_images[-1])
    delay = df.index

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), width_ratios=[1.35, 1, 1.15, 1.15], layout="constrained")
    ax = axes[0]
    ax.imshow(np.log1p(poff_mean), cmap="magma")
    ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, ec="#4dd0e1", fc="none", lw=1.5))
    ax.set_title("Laser-off mean, log (auto ROI)", fontsize=10)
    ax.set_axis_off()

    ax = axes[1]
    lim = np.abs(diff).max()
    ax.imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_title(f"On − off in ROI, {delay[-1]:g} ps", fontsize=10)
    ax.set_axis_off()

    on, off = "#d1495b", "#6c757d"
    ax = axes[2]
    ax.plot(delay, df["poff_intensity"], "o-", c=off, ms=4, label="laser off")
    ax.plot(delay, df["pon_intensity"], "o-", c=on, ms=4, label="laser on")
    ax.set(xlabel="Delay [ps]", ylabel="ROI intensity / first delay", title="ROI intensity")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[3]
    ax.plot(delay, df["poff_com_x"] * 1e3, "o-", c=off, ms=4, label="laser off")
    ax.plot(delay, df["pon_com_x"] * 1e3, "o-", c=on, ms=4, label="laser on")
    ax.set(xlabel="Delay [ps]", ylabel=r"$\Delta$COM$_x$ [10$^{-3}$ Å$^{-1}$]", title="ROI centre of mass (x)")
    for ax in axes[2:]:
        ax.title.set_fontsize(10)
        ax.grid(alpha=0.3)

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=100, facecolor="white")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    # Log and npz handles stay open until exit, which Windows cannot delete
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        write_data(root)
        run_script("make_dark.py", root)
        run_script("integrating_main.py", root)
        plot(root)


if __name__ == "__main__":
    main()
