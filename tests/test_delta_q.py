"""Pixel shift to |Δq| (QoraFlow#12), against the rotation of the scattered wavevector.

Units: lengths in m, energy in keV, q in 1/Å.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from QoraFlow import mathematics
from QoraFlow.mathematics import mul_delta_q

SDD = 1.3
DPS = 75e-6
ENERGY = 9.7
HC = 12.398419843320026  # keV Å, CODATA 2018


@pytest.fixture(autouse=True)
def config(monkeypatch):
    param = SimpleNamespace(sdd=SDD, dps=DPS, beam_energy=ENERGY)
    monkeypatch.setattr(
        mathematics.ConfigManager,
        "load_config",
        staticmethod(lambda: SimpleNamespace(param=param)),
    )


def rotated_q(shift_px: float) -> float:
    """|k_f(shift) - k_f(0)| for a pixel `shift_px` away on a flat detector normal to k_f."""
    k = 2 * np.pi * ENERGY / HC
    at = np.array([shift_px * DPS, 0.0, SDD])
    return float(np.linalg.norm(k * at / np.linalg.norm(at) - k * np.array([0.0, 0.0, 1.0])))


@pytest.mark.parametrize("shift", [1.0, 2.5, 10.0])
def test_shift_matches_the_wavevector_rotation(shift):
    got = mul_delta_q(np.array([3.0, 3.0 + shift]))
    assert got[0] == 0.0
    # differences are the chord-arc and arctan nonlinearity, below 1e-7 up to 10 px
    assert got[1] == pytest.approx(rotated_q(shift), rel=1e-6)


def test_one_pixel_is_k_times_the_pixel_angle():
    per_pixel = 2 * np.pi * ENERGY / HC * np.arctan(DPS / SDD)
    assert mul_delta_q(np.array([0.0, 1.0]))[1] == pytest.approx(per_pixel, rel=1e-12)
    assert per_pixel == pytest.approx(2.8360e-4, rel=1e-4)  # 9.7 keV, 75 µm at 1.3 m; was 5.672e-4
