"""Pixel position to |q| on a flat detector normal to the direct beam (QoraFlow#36).

Units: lengths in m, energy in keV, q in 1/Å.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from QoraFlow import mathematics
from QoraFlow.mathematics import pixel_to_del_q, pixel_to_q

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


def scattered_q(pixels: float) -> float:
    """|k_f - k_i| for a pixel `pixels` from where the direct beam meets the detector."""
    k = 2 * np.pi * ENERGY / HC
    at = np.array([pixels * DPS, 0.0, SDD])
    return float(np.linalg.norm(k * at / np.linalg.norm(at) - k * np.array([0.0, 0.0, 1.0])))


def test_q_grows_from_zero_at_the_direct_beam():
    got = pixel_to_q(np.array([0.0, 1.0, 10.0, 1000.0]))
    assert got[0] == 0.0
    assert got[1:] == pytest.approx([scattered_q(n) for n in (1.0, 10.0, 1000.0)], rel=1e-12)
    assert got[1] == pytest.approx(2.8360e-4, rel=1e-4)  # was 6.95 at 0 px and 2.84e-5 at 10 px


def test_shift_is_measured_from_the_first_pixel():
    got = pixel_to_del_q(np.array([3.0, 4.0, 13.0]))
    assert got[0] == 0.0
    assert got[1:] == pytest.approx([scattered_q(1.0), scattered_q(10.0)], rel=1e-12)
