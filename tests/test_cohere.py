import hashlib

import numpy as np
import pytest
import tifffile

from QoraFlow.cohere import load_cohere_project, read_cohere_config, read_cohere_tif


def write_cohere_tif(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, arr.transpose().astype(np.float32))


def make_project(root, rng):
    amplitude = rng.random((6, 8, 10)).astype(np.float32)
    prep = rng.random((5, 7, 9)).astype(np.float32)
    image = (rng.random((6, 8, 10)) + 1j * rng.random((6, 8, 10))).astype(np.complex64)
    support = image.real > 0.5

    write_cohere_tif(root / "phasing_data" / "data.tif", amplitude)
    write_cohere_tif(root / "preprocessed_data" / "prep_data.tif", prep)
    (root / "conf").mkdir()
    (root / "conf" / "config_data").write_text(
        "// data preparation\nintensity_threshold = 2.0\nadjust_dimensions = (-2, -2, 0, 0)\n"
        'alien_alg = "none"\n',
        encoding="utf-8",
    )
    (root / "conf" / "config_rec").write_text(
        '# reconstruction\nalgorithm_sequence = "3* (20*ER + 180*HIO) + 20*ER"\n'
        "shrink_wrap_trigger = [1, 1]\nnote = 'a=b'\n",
        encoding="utf-8",
    )
    results = root / "results_phasing"
    results.mkdir()
    np.save(results / "image.npy", image)
    np.save(results / "support.npy", support)
    np.savetxt(results / "errors.txt", np.array([0.5, 0.25, 0.125]))
    return amplitude, prep, image, support


def test_tif_axes_come_back_in_cohere_order(tmp_path):
    arr = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
    write_cohere_tif(tmp_path / "a.tif", arr)

    assert tifffile.imread(tmp_path / "a.tif").shape == (5, 3, 2)
    np.testing.assert_array_equal(read_cohere_tif(tmp_path / "a.tif"), arr)


def test_config_literals_comments_and_tuples(tmp_path):
    conf = tmp_path / "config_rec"
    conf.write_text(
        "// comment\n# another\n\nreconstructions = 5\nbeta = 0.9\n"
        "sequence = \"3* (20*ER + 180*HIO)\"\ncrop = (1, 2)\nlabel = 'a=b'\n",
        encoding="utf-8",
    )

    assert read_cohere_config(conf) == {
        "reconstructions": 5,
        "beta": 0.9,
        "sequence": "3* (20*ER + 180*HIO)",
        "crop": [1, 2],
        "label": "a=b",
    }


def test_config_rejects_a_bare_word(tmp_path):
    conf = tmp_path / "config"
    conf.write_text("data_dir = /some/where\n", encoding="utf-8")

    with pytest.raises(ValueError, match="config:1: data_dir"):
        read_cohere_config(conf)


def test_load_project_reads_data_results_and_provenance(tmp_path):
    amplitude, prep, image, support = make_project(tmp_path, np.random.default_rng(0))

    project = load_cohere_project(tmp_path)

    np.testing.assert_array_equal(project.amplitude, amplitude)
    np.testing.assert_allclose(project.intensity, amplitude.astype(np.float64) ** 2)
    np.testing.assert_array_equal(project.prep_data, prep)
    np.testing.assert_array_equal(project.image, image)
    np.testing.assert_array_equal(project.support, support)
    np.testing.assert_array_equal(project.errors, [0.5, 0.25, 0.125])
    assert project.configs["config_data"]["adjust_dimensions"] == [-2, -2, 0, 0]
    assert set(project.configs) == {"config_data", "config_rec"}

    data_file = tmp_path / "phasing_data" / "data.tif"
    assert project.provenance["phasing_data/data.tif"] == {
        "bytes": data_file.stat().st_size,
        "sha256": hashlib.sha256(data_file.read_bytes()).hexdigest(),
    }
    assert "results_phasing/image.npy" in project.provenance


def test_load_project_without_results_or_prep(tmp_path):
    make_project(tmp_path, np.random.default_rng(1))

    project = load_cohere_project(tmp_path, results="results_missing", prep=False)

    assert project.prep_data is None
    assert project.image is None
    assert project.support is None
    assert project.errors is None
    assert "preprocessed_data/prep_data.tif" not in project.provenance


def test_errors_npy_is_preferred(tmp_path):
    make_project(tmp_path, np.random.default_rng(2))
    np.save(tmp_path / "results_phasing" / "errors.npy", np.array([1.0, 0.1]))

    project = load_cohere_project(tmp_path)

    np.testing.assert_array_equal(project.errors, [1.0, 0.1])


def test_missing_data_tif_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Cohere project"):
        load_cohere_project(tmp_path)
