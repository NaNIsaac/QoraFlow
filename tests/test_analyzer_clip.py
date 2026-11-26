"""Clipping before the centre of mass (QoraFlow#56) and the absolute centre of mass.

A Gaussian peak left of the ROI centre on zero-mean noise; pump off is noisier than pump on, as for
a single pulse against a mean of many. Units: pixels.
"""

import numpy as np
import pytest
from roi_rectangle import RoiRectangle

from QoraFlow.analyzer.core import DataAnalyzer

H, W = 60, 128
X0, Y0 = 40.0, 30.0
y, x = np.indices((H, W))
PEAK = 50 * np.exp(-((x - X0) ** 2 + (y - Y0) ** 2) / (2 * 3.0**2))


@pytest.fixture
def scan(tmp_path):
    rng = np.random.default_rng(0)
    poff = PEAK + rng.normal(0, 0.3, (2, H, W))
    pon = PEAK + rng.normal(0, 0.05, (2, H, W))
    file = tmp_path / "scan.npz"
    np.savez(file, delay=np.array([0.0, 1.0]), poff=poff, pon=pon)
    return file


def gap(analyzer: DataAnalyzer, roi: RoiRectangle) -> float:
    return float(
        analyzer.center_of_mass(roi, "pon")[0][0] - analyzer.center_of_mass(roi, "poff")[0][0]
    )


def test_clipping_biases_the_noisier_state(scan):
    whole = RoiRectangle(0, H, 0, W)
    assert gap(DataAnalyzer(scan), whole) < -2
    assert abs(gap(DataAnalyzer(scan, clip_negative=False), whole)) < 0.5


def test_center_of_mass_is_absolute(scan):
    box = RoiRectangle(int(Y0) - 10, int(Y0) + 11, int(X0) - 10, int(X0) + 11)
    com_x, com_y = DataAnalyzer(scan, clip_negative=False).center_of_mass(box, "pon")
    np.testing.assert_allclose(com_x, X0, atol=0.05)
    np.testing.assert_allclose(com_y, Y0, atol=0.05)


def test_center_of_mass_needs_the_state(tmp_path):
    file = tmp_path / "off_only.npz"
    np.savez(file, delay=np.array([0.0]), poff=PEAK[None])
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="no 'pon'"):
        DataAnalyzer(file).center_of_mass(RoiRectangle(0, H, 0, W), "pon")
