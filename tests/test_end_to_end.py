"""End-to-end check on a synthetic PAL-XFEL scan, from raw run files to ROI curves.

The scan comes from ``examples/readme_figure.py``: a Gaussian Bragg peak with Poisson noise, a
dark pedestal and QBPM fluctuations. Laser-on shots after t=0 lose up to 25 % of the peak
intensity (time constant 1.5 ps) and shift up to 1.5 pixels along x (time constant 3 ps). The
test runs ``make_dark.py`` and ``integrating_main.py`` unchanged, then ``DataAnalyzer``, and checks
physical invariants, not a stored result:

* laser-off shots do not depend on the delay: intensity and centre of mass stay constant;
* laser-on shots before the pump equal laser-off shots;
* after the pump the on/off intensity ratio follows ``1 - 0.25 (1 - exp(-t / 1.5 ps))``;
* the centre of mass moves along x by ``1.5 (1 - exp(-t / 3 ps))`` pixels, converted to
  Å⁻¹ with the config geometry, and not along y.

Reproduce: ``uv run pytest tests/test_end_to_end.py`` (about 10 s). 800 shots per file keep the
Poisson noise of the on/off ratio below the tolerances. Everything is synthetic; no measured data
is used.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from QoraFlow.analyzer.core import DataAnalyzer
from QoraFlow.config import ConfigManager
from QoraFlow.gui.select_roi import auto_roi
from QoraFlow.mathematics import mul_delta_q

REPO = Path(__file__).resolve().parents[1]
DELAYS = [-1.0, 0.0, 2.0, 6.0, 10.0]  # ps


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location(
        "readme_figure", REPO / "examples/readme_figure.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DELAYS = DELAYS
    module.SHOTS = 800
    return module


@pytest.fixture(scope="module")
def curves(example, tmp_path_factory):
    root = tmp_path_factory.mktemp("scan")
    example.write_data(root)
    example.run_script("make_dark.py", root)
    example.run_script("integrating_main.py", root)

    ConfigManager.initialize(root / "config.yaml")
    config = ConfigManager.load_config()
    roi = auto_roi(root / "raw" / "run=001" / "scan=001", config)
    npz = config.path.processed_dir / "run=001" / "scan=001" / "run=0001_scan=0001_LND.npz"
    return DataAnalyzer(npz).analyze_by_roi(roi)


def test_one_row_per_delay(curves):
    assert list(curves.index) == DELAYS


def test_laser_off_does_not_depend_on_delay(curves):
    assert curves["poff_intensity"].to_numpy() == pytest.approx(1.0, abs=0.03)
    assert np.abs(curves["poff_com_x"]).max() < 0.1 * np.abs(curves["pon_com_x"]).max()


def test_before_the_pump_on_equals_off(curves):
    before = curves.loc[[-1.0, 0.0]]
    assert before["pon_intensity"].to_numpy() == pytest.approx(
        before["poff_intensity"].to_numpy(), abs=0.03
    )


def test_intensity_ratio_follows_the_injected_decay(curves):
    delay = curves.index.to_numpy()
    expected = np.where(delay > 0, 1 - 0.25 * (1 - np.exp(-delay / 1.5)), 1.0)
    ratio = (curves["pon_intensity"] / curves["poff_intensity"]).to_numpy()
    assert ratio == pytest.approx(expected, abs=0.04)
    assert np.all(np.diff(ratio[delay >= 0]) <= 0.01)


def test_centre_of_mass_shifts_along_x_only(curves):
    delay = curves.index.to_numpy()
    shift_px = np.where(delay > 0, 1.5 * (1 - np.exp(-delay / 3)), 0.0)
    expected = mul_delta_q(shift_px)
    measured = (curves["pon_com_x"] - curves["poff_com_x"]).to_numpy()
    assert measured == pytest.approx(expected, abs=0.1 * np.abs(expected).max())
    assert np.abs(curves["pon_com_y"] - curves["poff_com_y"]).max() < 0.1 * np.abs(expected).max()
