from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from QoraFlow.integrator import saver


@pytest.fixture
def tif_saver(tmp_path, monkeypatch):
    config = SimpleNamespace(path=SimpleNamespace(processed_dir=tmp_path))
    monkeypatch.setattr(saver.ConfigManager, "load_config", staticmethod(lambda: config))
    return saver.get_saver_strategy("tif")


@pytest.mark.parametrize(
    ("shape", "axes"),
    [((4, 5), "YX"), ((3, 4, 5), "ZYX"), ((2, 3, 4, 5), "TZYX"), ((2, 3, 2, 4, 5), "TZCYX")],
)
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.uint16, np.int64])
def test_tif_saver_writes_imagej_hyperstacks(tif_saver, shape, axes, dtype):
    data = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    tif_saver.save(7, 1, {"poff": data})
    with tifffile.TiffFile(tif_saver.file) as tif:
        assert tif.is_imagej
        assert tif.imagej_metadata["unit"] == "um"
        assert tif.series[0].axes == axes
        stored = tif.asarray()
    assert stored.dtype in (np.float32, np.uint16)
    np.testing.assert_array_equal(stored, data.astype(stored.dtype))


def test_tif_saver_rejects_six_dimensions(tif_saver):
    with pytest.raises(ValueError, match="2 to 5 dimensions"):
        tif_saver.save(7, 1, {"poff": np.zeros((1,) * 6, np.float32)})
