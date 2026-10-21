"""examples/synthetic_pump_probe end to end: synthetic raw run files -> make_dark -> integrating_main
-> DataAnalyzer ROI curves.

Checks the reduction against the response that was injected (truth, with tolerances set by the
Poisson noise of 800 shots per file), today's output as a regression snapshot, and that the truth in
expected.json is the response the generator really injects. Units: delay in ps, shifts in detector
pixels, intensities relative to the first delay.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

EXAMPLE = Path(__file__).parents[1] / "examples" / "synthetic_pump_probe"
EXPECTED = json.loads((EXAMPLE / "expected.json").read_text(encoding="utf-8"))

_spec = importlib.util.spec_from_file_location("synthetic_pump_probe", EXAMPLE / "run.py")
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)


@pytest.fixture(scope="module")
def result():
    return example.run()


def _check(name: str, got, spec: dict):
    got, want = np.asarray(got, float), np.asarray(spec["value"], float)
    assert np.all(np.abs(got - want) <= spec["tol"]), f"{name}: {got} vs {want} ± {spec['tol']}"


def test_truth_is_the_injected_response():
    for i, delay in enumerate(EXPECTED["delay_ps"]):
        factor, shift = example.injected(delay)
        assert EXPECTED["truth"]["on_off_ratio"]["value"][i] == pytest.approx(factor, abs=1e-4)
        assert EXPECTED["truth"]["shift_x_px"]["value"][i] == pytest.approx(shift, abs=1e-4)


def test_one_row_per_delay_in_order(result):
    assert result["delay_ps"] == EXPECTED["delay_ps"]


@pytest.mark.parametrize("key", list(k for k in EXPECTED["truth"] if k != "about"))
def test_reduction_matches_the_truth(result, key):
    _check(key, result[key], EXPECTED["truth"][key])


def test_the_response_only_starts_after_time_zero(result):
    delay = np.asarray(result["delay_ps"])
    ratio = np.asarray(result["on_off_ratio"])
    shift = np.asarray(result["shift_x_px"])
    after = delay > 0
    assert ratio[after].max() < ratio[~after].min()
    assert shift[after].min() > shift[~after].max()


@pytest.mark.parametrize("key", list(k for k in EXPECTED["regression"] if k != "about"))
def test_output_is_unchanged(result, key):
    _check(key, result[key], EXPECTED["regression"][key])
