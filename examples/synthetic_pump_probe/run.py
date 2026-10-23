"""Synthetic end-to-end reduction: a known pump-probe response -> raw run files -> QoraFlow -> ROI curves.

Writes a synthetic PAL-XFEL scan in the PalXFELLoader layout with the generator of
``examples/readme_figure.py`` (a Gaussian Bragg peak, Poisson noise, dark pedestal, QBPM
fluctuations), runs ``scripts/make_dark.py`` and ``scripts/integrating_main.py`` unchanged, then
``DataAnalyzer.analyze_by_roi``, and compares the ROI curves with the response that was put in.

    PYTHONPATH=src uv run python examples/synthetic_pump_probe/run.py

Prints the injected response next to the measured one and writes result.json beside this file.
"""

import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parents[1]
SEED = 0

# Delays in ps; 800 shots per file (alternating laser on/off), 2 files per delay.
DELAYS = [-1.0, 0.0, 2.0, 6.0, 10.0]
SHOTS = 800
# (y1, y2, x1, x2) px: the region QoraFlow.gui.select_roi.auto_roi picks for this scan, fixed here so
# the example runs without a Qt binding (that module imports the Qt ROI selector).
ROI = (25, 65, 45, 85)


def load_generator(seed: int = SEED):
    """examples/readme_figure.py, freshly loaded so its module-level generator restarts at ``seed``.

    Seed 0 is the module's own generator; other seeds replace it and redraw the dark pedestal in
    the same order as at module load.
    """
    spec = importlib.util.spec_from_file_location(
        "readme_figure", REPO / "examples/readme_figure.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DELAYS = DELAYS
    module.SHOTS = SHOTS
    if seed != 0:
        module.rng = np.random.default_rng(seed)
        module.pedestal = module.rng.normal(40, 4, (module.H, module.W)).astype(np.float32)
    return module


def injected(delay: float) -> tuple[float, float]:
    """The response the generator puts into laser-on shots: (intensity factor, x shift in px)."""
    return load_generator().laser_response(delay)


def run(seed: int = SEED, background: str | None = None) -> dict:
    """Reduce one synthetic scan; ``background`` goes to ``DataAnalyzer.analyze_by_roi``."""
    from roi_rectangle import RoiRectangle

    from QoraFlow.analyzer.core import DataAnalyzer
    from QoraFlow.config import ConfigManager
    from QoraFlow.mathematics import mul_delta_q

    gen = load_generator(seed)
    # Log and npz handles stay open until exit, which Windows cannot delete
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        gen.write_data(root)
        gen.run_script("make_dark.py", root)
        gen.run_script("integrating_main.py", root)

        ConfigManager.initialize(root / "config.yaml")
        config = ConfigManager.load_config()
        roi = RoiRectangle(*ROI)
        npz = config.path.processed_dir / "run=001" / "scan=001" / "run=0001_scan=0001_LND.npz"
        df = DataAnalyzer(npz).analyze_by_roi(roi, background=background)
        q_per_px = float(mul_delta_q(np.array([0.0, 1.0]))[1])

    def px(col: str) -> list[float]:
        return [float(v) / q_per_px for v in df[col]]

    return {
        "seed": seed,
        "background": background,
        "delay_ps": [float(d) for d in df.index],
        "roi": list(roi.to_tuple()),
        "q_per_px": q_per_px,
        "poff_intensity": [float(v) for v in df["poff_intensity"]],
        "pon_intensity": [float(v) for v in df["pon_intensity"]],
        "on_off_ratio": [float(v) for v in df["pon_intensity"] / df["poff_intensity"]],
        "shift_x_px": [on - off for on, off in zip(px("pon_com_x"), px("poff_com_x"), strict=True)],
        "shift_y_px": [on - off for on, off in zip(px("pon_com_y"), px("poff_com_y"), strict=True)],
        "poff_shift_x_px": px("poff_com_x"),
    }


if __name__ == "__main__":
    result = run()
    (HERE / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"{'delay ps':>9s}{'ratio in':>10s}{'ratio out':>10s}{'dx in px':>10s}{'dx out px':>10s}")
    for i, d in enumerate(result["delay_ps"]):
        factor, shift = injected(d)
        print(
            f"{d:9.1f}{factor:10.3f}{result['on_off_ratio'][i]:10.3f}"
            f"{shift:10.3f}{result['shift_x_px'][i]:10.3f}"
        )
    print(f"1 px = {result['q_per_px']:.3e} 1/Angstrom at the config geometry")
