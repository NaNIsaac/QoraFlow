from pathlib import Path

import numpy as np
import pytest

from QoraFlow.integrator import core
from QoraFlow.integrator.core import CoreIntegrator
from QoraFlow.integrator.loader import RawDataLoader


class FakeLoader(RawDataLoader):
    """Reads '<delay> <states>' from a text file named like an h5 file."""

    def __init__(self, file):
        text = Path(file).read_text().split()
        self.delay = float(text[0])
        self.states = text[1:]

    def get_data(self):
        raise NotImplementedError

    def get_chunked_data(self, chunk_size):
        chunk = {"delay": self.delay}
        for state in self.states:
            chunk[state] = np.full((2, 3, 3), self.delay)
        yield chunk


def write_scan(scan_dir: Path, specs: list[str]) -> None:
    scan_dir.mkdir(parents=True)
    for i, spec in enumerate(specs, start=1):
        (scan_dir / f"p{i:04d}.h5").write_text(spec)


@pytest.fixture(autouse=True)
def no_config(monkeypatch):
    monkeypatch.setattr(core.ConfigManager, "load_config", staticmethod(lambda: None))


def test_relative_scan_dir(tmp_path, monkeypatch):
    write_scan(tmp_path / "data" / "run001", ["1.0 pon poff", "2.0 pon poff"])
    monkeypatch.chdir(tmp_path)

    result = CoreIntegrator(FakeLoader).run(Path("data/run001"))["no_processing"]

    np.testing.assert_array_equal(result["delay"], [1.0, 2.0])
    assert result["pon"].shape == (2, 3, 3)


def test_all_batches_fail_raises(tmp_path):
    scan_dir = tmp_path / "scan"
    write_scan(scan_dir, ["not-a-number", "also-bad"])

    with pytest.raises(RuntimeError, match="failed to load"):
        CoreIntegrator(FakeLoader).run(scan_dir)
