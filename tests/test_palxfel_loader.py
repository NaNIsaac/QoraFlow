from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
from loguru import logger

from QoraFlow.config.enums import Detector, Hertz, Hutch, Xray
from QoraFlow.integrator import loader as loader_module
from QoraFlow.integrator.loader import PalXFELLoader

H, W = 2, 2


@pytest.fixture(autouse=True)
def fake_config(monkeypatch):
    config = SimpleNamespace(
        param=SimpleNamespace(
            hutch=Hutch.EH1,
            detector=Detector.JUNGFRAU1,
            pump_setting=Hertz.TEN,
            xray=Xray.HARD,
        )
    )
    monkeypatch.setattr(loader_module.ConfigManager, "load_config", staticmethod(lambda: config))
    monkeypatch.setattr(loader_module, "setup_logger", lambda: logger)


def write_scan(path, metadata_order: list[int]):
    """A PalXFEL-shaped file with 6 frames.

    Frame i has every pixel equal to i, qbpm 10 * (i + 1) and is pump-on when i is odd.
    `metadata_order` is the row order of the metadata table, which the loader joins on.
    """
    n = 6
    timestamps = np.arange(1000, 1000 + n, dtype=np.int64)
    metadata = pd.DataFrame(
        {
            "timestamp_info.RATE_HX_10HZ": [bool(i % 2) for i in metadata_order],
            "delay_value": 1.5,
        },
        index=timestamps[metadata_order],
    )
    metadata.to_hdf(path, key="metadata", mode="w")

    with h5py.File(path, "a") as hf:
        image = hf.create_group("detector/eh1/jungfrau1/image")
        image["block0_items"] = timestamps
        image["block0_values"] = np.arange(n, dtype=np.float32)[:, None, None] * np.ones((n, H, W))

        qbpm = hf.create_group("qbpm/eh1/qbpm1")
        per_channel = (np.arange(n, dtype=np.float32) + 1)[:, None] * 2.5  # sums to 10 * (i + 1)
        for ch in range(1, 5):
            group = qbpm.create_group(f"waveforms.ch{ch}")
            group["axis1"] = timestamps
            group["block0_values"] = per_channel
    return n


def collect(loader: PalXFELLoader) -> dict[int, tuple[float, bool]]:
    """Map frame number (from the pixel value) to the (qbpm, pump_on) it was paired with."""
    pairs = {}
    for chunk in loader.get_chunked_data(chunk_size=4):
        for state, pump_on in (("poff", False), ("pon", True)):
            if state not in chunk:
                continue
            for img, q in zip(chunk[state], chunk[f"{state}_qbpm"], strict=True):
                pairs[int(img[0, 0])] = (float(q), pump_on)
    return pairs


@pytest.mark.parametrize(
    "order",
    [[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0], [2, 0, 5, 1, 4, 3]],
    ids=["sorted", "reversed", "shuffled"],
)
def test_frames_keep_their_own_qbpm_and_pump_state(tmp_path, order):
    path = tmp_path / "scan.h5"
    n = write_scan(path, order)
    with PalXFELLoader(path) as loader:
        pairs = collect(loader)

    assert sorted(pairs) == list(range(n))
    for frame, (q, pump_on) in pairs.items():
        assert q == pytest.approx(10.0 * (frame + 1))
        assert pump_on is bool(frame % 2)


def test_get_data_agrees_with_chunked_data(tmp_path):
    path = tmp_path / "scan.h5"
    write_scan(path, [3, 1, 5, 0, 4, 2])
    with PalXFELLoader(path) as loader:
        data = loader.get_data()
    for state in ("pon", "poff"):
        frames = data[state][:, 0, 0]
        np.testing.assert_allclose(data[f"{state}_qbpm"], 10.0 * (frames + 1))


def test_frames_with_an_unknown_pump_flag_are_dropped_not_read_as_pump_on(tmp_path):
    path = tmp_path / "scan.h5"
    write_scan(path, [0, 1, 2, 3, 4, 5])
    metadata = pd.read_hdf(path, key="metadata")
    flags = metadata["timestamp_info.RATE_HX_10HZ"].astype(float)
    flags.iloc[2] = np.nan  # frame 2 has no pump flag
    metadata["timestamp_info.RATE_HX_10HZ"] = flags
    metadata.to_hdf(path, key="metadata", mode="a")

    with PalXFELLoader(path) as loader:
        pairs = collect(loader)

    assert sorted(pairs) == [0, 1, 3, 4, 5]
    for frame, (q, pump_on) in pairs.items():
        assert q == pytest.approx(10.0 * (frame + 1))
        assert pump_on is bool(frame % 2)


def test_a_file_where_every_pump_flag_is_unknown_is_refused(tmp_path):
    path = tmp_path / "scan.h5"
    write_scan(path, [0, 1, 2, 3, 4, 5])
    metadata = pd.read_hdf(path, key="metadata")
    metadata["timestamp_info.RATE_HX_10HZ"] = np.nan
    metadata.to_hdf(path, key="metadata", mode="a")
    with pytest.raises(ValueError, match="No matching data"):
        PalXFELLoader(path)
