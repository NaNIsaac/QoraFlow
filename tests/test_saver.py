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


@pytest.mark.parametrize("shape", [(4, 5), (3, 4, 5)])
def test_tif_saver_writes_images_and_stacks(tif_saver, shape):
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    tif_saver.save(7, 1, {"poff": data})
    np.testing.assert_array_equal(tifffile.imread(tif_saver.file), data)
