from types import SimpleNamespace

import numpy as np
import pytest

from QoraFlow.preprocessor import generic_preprocessors as gp


@pytest.fixture(autouse=True)
def clear_cache():
    gp._load_cached_dark_image.cache_clear()
    yield
    gp._load_cached_dark_image.cache_clear()


def use_dark(monkeypatch, path):
    config = SimpleNamespace(path=SimpleNamespace(save_dark_dir=path))
    monkeypatch.setattr(gp.ConfigManager, "load_config", staticmethod(lambda: config))


def test_subtracts_2d_dark(tmp_path, monkeypatch):
    dark_file = tmp_path / "dark.npy"
    np.save(dark_file, np.full((3, 3), 2.0))
    use_dark(monkeypatch, dark_file)
    images = np.full((2, 3, 3), 5.0)
    np.testing.assert_array_equal(gp.subtract_dark(images), np.full((2, 3, 3), 3.0))


def test_averages_3d_dark_stack_and_clips_at_zero(tmp_path, monkeypatch):
    dark_file = tmp_path / "dark.npy"
    np.save(dark_file, np.stack([np.full((2, 2), 1.0), np.full((2, 2), 3.0)]))
    use_dark(monkeypatch, dark_file)
    out = gp.subtract_dark(np.full((1, 2, 2), 1.0))
    np.testing.assert_array_equal(out, np.zeros((1, 2, 2)))


def test_missing_dark_file_raises(tmp_path, monkeypatch):
    use_dark(monkeypatch, tmp_path / "missing.npy")
    with pytest.raises(FileNotFoundError):
        gp.subtract_dark(np.ones((1, 2, 2)))


def test_unset_dark_path_raises(monkeypatch):
    use_dark(monkeypatch, gp.Path())
    with pytest.raises(FileNotFoundError):
        gp.subtract_dark(np.ones((1, 2, 2)))


def test_unreadable_dark_file_raises(tmp_path, monkeypatch):
    dark_file = tmp_path / "dark.npy"
    dark_file.write_bytes(b"not a numpy file")
    use_dark(monkeypatch, dark_file)
    with pytest.raises(ValueError, match="Failed to load"):
        gp.subtract_dark(np.ones((1, 2, 2)))


def test_wrong_ndim_raises(tmp_path, monkeypatch):
    dark_file = tmp_path / "dark.npy"
    np.save(dark_file, np.ones(4))
    use_dark(monkeypatch, dark_file)
    with pytest.raises(ValueError, match="2D or 3D"):
        gp.subtract_dark(np.ones((1, 2, 2)))
