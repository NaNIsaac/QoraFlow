"""Scenario-matrix tests exercise the real PAL loader against known synthetic truth."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from test_palxfel_loader import fake_config  # noqa: F401

from QoraFlow.integrator.loader import PalXFELLoader

SOURCE = Path(__file__).resolve().parents[1] / "examples" / "synthetic_loader_cases.py"
SPEC = importlib.util.spec_from_file_location("synthetic_loader_cases", SOURCE)
SCENARIOS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCENARIOS)


@pytest.mark.parametrize("scenario", SCENARIOS.CASES)
def test_loader_scenarios_against_independent_truth(tmp_path, scenario):
    path, truth = SCENARIOS.write_case(tmp_path, scenario)
    assert json.loads((tmp_path / f"{scenario}.json").read_text(encoding="utf-8")) == truth
    if truth["expect_loader_error"]:
        with pytest.raises(ValueError, match="No matching data"):
            PalXFELLoader(path)
        return

    with PalXFELLoader(path) as loader:
        found = {}
        for chunk in loader.get_chunked_data(chunk_size=3):
            for state, is_on in (("poff", False), ("pon", True)):
                if state not in chunk:
                    continue
                for image, qbpm in zip(
                    chunk[state], chunk[f"{state}_qbpm"], strict=True
                ):
                    index = int(image[0, 0])
                    assert np.all(image == index)
                    assert index not in found
                    found[index] = (is_on, float(qbpm))
        full = loader.get_data()
        for state in ("poff", "pon"):
            expected_indices = [
                i for i in truth["expected_frame_indices"]
                if truth["expected_pump_on"][str(i)] is (state == "pon")
            ]
            if expected_indices:
                np.testing.assert_array_equal(
                    full[state][:, 0, 0], expected_indices
                )
            else:
                assert state not in full

    assert sorted(found) == truth["expected_frame_indices"]
    for index, (pump_on, qbpm) in found.items():
        assert pump_on is truth["expected_pump_on"][str(index)]
        assert qbpm == pytest.approx(truth["expected_qbpm"][str(index)])


def test_unknown_scenario_rejected_without_creating_data(tmp_path):
    with pytest.raises(ValueError, match="Unknown scenario"):
        SCENARIOS.write_case(tmp_path, "unexpected")
    assert not list(tmp_path.iterdir())
