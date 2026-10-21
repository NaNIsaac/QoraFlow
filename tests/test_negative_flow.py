"""Inputs that must fail loudly or degrade in a defined way, not produce quiet wrong output."""

from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
from roi_rectangle import RoiRectangle
from test_palxfel_loader import collect, fake_config, write_scan  # noqa: F401

from QoraFlow import mathematics
from QoraFlow.analyzer.converter import convert_data, save_mat
from QoraFlow.analyzer.core import DataAnalyzer
from QoraFlow.config.enums import Hertz
from QoraFlow.integrator import loader as loader_module
from QoraFlow.integrator.loader import PalXFELLoader


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PalXFELLoader(tmp_path / "absent.h5")


def test_no_timestamp_overlap_raises(tmp_path):
    path = tmp_path / "scan.h5"
    write_scan(path, [0, 1, 2, 3, 4, 5])
    with h5py.File(path, "a") as hf:
        del hf["qbpm/eh1/qbpm1/waveforms.ch1/axis1"]
        hf["qbpm/eh1/qbpm1/waveforms.ch1/axis1"] = np.arange(5000, 5006, dtype=np.int64)
    with pytest.raises(ValueError, match="No matching data"):
        PalXFELLoader(path)


def test_file_without_detector_group_raises(tmp_path):
    path = tmp_path / "scan.h5"
    pd.DataFrame({"x": [1]}, index=[1000]).to_hdf(path, key="metadata", mode="w")
    with pytest.raises(KeyError, match="detector"):
        PalXFELLoader(path)


def test_missing_delay_column_gives_nan(tmp_path):
    path = tmp_path / "scan.h5"
    write_scan(path, [0, 1, 2, 3, 4, 5])
    metadata = pd.read_hdf(path, key="metadata").drop(columns="delay_value")
    metadata.to_hdf(path, key="metadata", mode="a")
    with PalXFELLoader(path) as loader:
        assert np.isnan(loader.delay)


def test_missing_rate_column_raises(tmp_path):
    path = tmp_path / "scan.h5"
    write_scan(path, [0, 1, 2, 3, 4, 5])
    metadata = pd.read_hdf(path, key="metadata").drop(columns="timestamp_info.RATE_HX_10HZ")
    metadata.to_hdf(path, key="metadata", mode="a")
    with pytest.raises(KeyError):
        PalXFELLoader(path)


def test_pump_off_setting_marks_every_frame_pump_off(tmp_path, monkeypatch):
    path = tmp_path / "scan.h5"
    write_scan(path, [0, 1, 2, 3, 4, 5])
    config = loader_module.ConfigManager.load_config()
    monkeypatch.setattr(config.param, "pump_setting", Hertz.ZERO)
    with PalXFELLoader(path) as loader:
        assert not loader.pump_state.any()
        pairs = collect(loader)
    assert all(not pump_on for _, pump_on in pairs.values())


def test_pump_state_is_not_inverted(tmp_path):
    path = tmp_path / "scan.h5"
    write_scan(path, [0, 1, 2, 3, 4, 5])
    with PalXFELLoader(path) as loader:
        np.testing.assert_array_equal(loader.pump_state, [False, True] * 3)


def test_converter_rejects_unknown_formats(tmp_path):
    with pytest.raises(ValueError, match="input format"):
        convert_data("tiff", "npz", "x", str(tmp_path / "out"))
    npz = tmp_path / "in.npz"
    np.savez(npz, poff=np.zeros((1, 2, 2)), delay=np.zeros(1))
    with pytest.raises(ValueError, match="output format"):
        convert_data("npz", "csv", str(npz), str(tmp_path / "out"))


def test_save_mat_requires_a_poff_stack(tmp_path):
    with pytest.raises(KeyError):
        save_mat(str(tmp_path / "x.mat"), {"delay": np.zeros(1)})


def write_npz(path, **arrays):
    np.savez(path, **arrays)
    return path


def test_analyzer_requires_a_delay_axis(tmp_path):
    path = write_npz(tmp_path / "a.npz", poff=np.ones((2, 3, 3)), pon=np.ones((2, 3, 3)))
    with pytest.raises(ValueError, match="delay"):
        DataAnalyzer(path)


def test_analyzer_without_any_image_state_fails(tmp_path):
    path = write_npz(tmp_path / "a.npz", delay=np.arange(2.0))
    with pytest.raises(ValueError, match="neither"):
        DataAnalyzer(path)


def test_analyzer_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DataAnalyzer(tmp_path / "absent.npz")


@pytest.mark.parametrize("present", ["poff", "pon"])
def test_a_missing_pump_state_is_absent_not_invented(tmp_path, monkeypatch, present):
    """#30: the missing state used to be filled with ones and looked like data.

    A pump-off run (pump_setting 0HZ) legitimately has only 'poff', so loading still works, with a
    warning; the missing state is None, its ROI columns are NaN, and methods that need both states
    refuse to run.
    """
    missing = {"poff": "pon", "pon": "poff"}[present]
    geometry = SimpleNamespace(param=SimpleNamespace(sdd=1.3, dps=7.5e-5, beam_energy=9.8))
    monkeypatch.setattr(mathematics.ConfigManager, "load_config", staticmethod(lambda: geometry))
    path = write_npz(tmp_path / "a.npz", delay=np.arange(2.0), **{present: np.full((2, 3, 3), 5.0)})
    with pytest.warns(UserWarning, match=f"no '{missing}'"):
        analyzer = DataAnalyzer(path)
    assert getattr(analyzer, f"{missing}_images") is None
    assert getattr(analyzer, f"{present}_images").shape == (2, 3, 3)

    df = analyzer.analyze_by_roi(RoiRectangle(0, 3, 0, 3))
    assert df[[c for c in df if c.startswith(missing)]].isna().all().all()
    assert df[f"{present}_intensity"].tolist() == [1.0, 1.0]

    with pytest.raises(ValueError, match=missing):
        analyzer.pon_subtract_by_poff()
    summed = dict(zip(("poff", "pon"), analyzer.get_summed_image(), strict=True))
    assert summed[missing] is None
