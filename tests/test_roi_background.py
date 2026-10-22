"""Background removal before the ROI centre of mass (QoraFlow#28).

A Gaussian peak 5 px right of the ROI centre moves 1 px and drops to 0.75 in laser-on frames, as in
examples/synthetic_pump_probe. Noise free. Units: pixels.
"""

import numpy as np
import pytest
from roi_rectangle import RoiRectangle

from QoraFlow.analyzer import core
from QoraFlow.analyzer.core import DataAnalyzer, edge_background

H = W = 40
SIGMA = 3.0
X0, Y0 = 24.5, 19.5  # ROI pixel mean is 19.5
SHIFT = 1.0
ROI = RoiRectangle(0, H, 0, W)
y, x = np.indices((H, W))


def peak(dx: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-((x - X0 - dx) ** 2 + (y - Y0) ** 2) / (2 * SIGMA**2))


@pytest.fixture(autouse=True)
def pixels(monkeypatch):
    # keep shifts in pixels; the Δq conversion is tested in test_delta_q.py
    monkeypatch.setattr(core, "mul_delta_q", lambda a: a)


def analyzer(tmp_path, background: np.ndarray) -> DataAnalyzer:
    poff = np.stack([peak(0, 30), peak(0, 30)]) + background
    pon = np.stack([peak(0, 30), peak(SHIFT, 22.5)]) + background
    file = tmp_path / "scan.npz"
    np.savez(file, delay=np.array([-1.0, 1.0]), poff=poff, pon=pon)
    return DataAnalyzer(file)


def shift(df) -> float:
    return float(df["pon_com_x"].iloc[1] - df["poff_com_x"].iloc[1])


FLAT = np.full((H, W), 0.5)
GRADIENT = np.broadcast_to(x / (W - 1), (H, W))


def test_default_is_the_plain_centre_of_mass(tmp_path):
    a = analyzer(tmp_path, FLAT)
    images = ROI.slice(a.pon_images)
    com = (images * x).sum((1, 2)) / images.sum((1, 2))
    df = a.analyze_by_roi(ROI)
    np.testing.assert_array_equal(df["pon_com_x"].to_numpy(), com - com[0])
    assert shift(df) < 0.8  # QoraFlow#28: the flat floor dilutes the shift


@pytest.mark.parametrize("method", ["edge", "plane"])
def test_flat_background_is_removed(tmp_path, method):
    assert shift(analyzer(tmp_path, FLAT).analyze_by_roi(ROI, background=method)) == pytest.approx(
        SHIFT, abs=2e-3
    )


def test_plane_removes_a_gradient_that_the_median_leaves(tmp_path):
    a = analyzer(tmp_path, GRADIENT)
    assert shift(a.analyze_by_roi(ROI, background="plane")) == pytest.approx(SHIFT, abs=2e-3)
    assert abs(shift(a.analyze_by_roi(ROI, background="edge")) - SHIFT) > 0.1


def test_intensities_are_not_changed(tmp_path):
    a = analyzer(tmp_path, FLAT)
    plain, removed = a.analyze_by_roi(ROI), a.analyze_by_roi(ROI, background="edge")
    for key in ("poff_intensity", "pon_intensity"):
        np.testing.assert_array_equal(removed[key], plain[key])


def test_plane_is_exact_on_a_plane():
    stack = np.stack([1 + 0.2 * x - 0.1 * y, 3 + 0 * x + 0.05 * y])
    np.testing.assert_allclose(edge_background(stack, "plane"), stack, atol=1e-12)


def test_unknown_method():
    with pytest.raises(ValueError, match="unknown background method"):
        edge_background(np.zeros((1, H, W)), "mean")
