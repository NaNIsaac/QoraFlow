import h5py
import numpy as np
import pytest

from QoraFlow.damnit import DATAARRAY, load_damnit_run

PHONY = "This is a netCDF dimension but not a netCDF variable.      {}"


def write_dataarray(handle, name, values, scales, attrs=None):
    group = handle.create_group(name)
    group.attrs["_damnit_objtype"] = "dataarray"
    data = group.create_dataset(DATAARRAY, data=values)
    for k, v in (attrs or {}).items():
        data.attrs[k] = v
    for axis, (dim, coord) in enumerate(scales):
        scale = group.create_dataset(dim, data=coord)
        scale.make_scale(dim)
        data.dims[axis].attach_scale(scale)


@pytest.fixture
def damnit_file(tmp_path):
    path = tmp_path / "run.h5"
    position = np.linspace(18.9, 19.5, 4)
    image = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
    image[0, 0, 0] = np.nan
    trains = np.arange(100, 106, dtype=np.uint64)
    with h5py.File(path, "w") as handle:
        write_dataarray(
            handle,
            "avg",
            image,
            [("position", position), ("dim_1", np.zeros(3, ">f4")), ("dim_2", np.zeros(2, ">f4"))],
        )
        group = handle["avg"]
        for dim in ("dim_1", "dim_2"):
            group[dim].attrs["NAME"] = np.bytes_(PHONY.format(group[dim].shape[0]))
        write_dataarray(
            handle,
            "xgm",
            np.linspace(1, 2, 6, dtype=np.float32),
            [("trainId", trains)],
            {"max_diff": 1.0},
        )
        handle.create_group("n_trains").create_dataset("data", data=6)
        handle.create_group("scan_type").create_dataset(
            "data", data=np.array([b"dscan th"], dtype=object), dtype=h5py.string_dtype()
        )
        reduced = handle.create_group(".reduced")
        reduced["n_trains"] = 6
        reduced["xgm"] = np.float32(1.5)
        reduced.create_dataset(
            "scan_type", data=np.array([b"dscan th"], dtype=object), dtype=h5py.string_dtype()
        )
        reduced["preview"] = np.frombuffer(b"\x89PNG", dtype=np.uint8)
    return path, position, image, trains


def test_names_and_summary(damnit_file):
    path, *_ = damnit_file
    run = load_damnit_run(path)
    assert run.names == ("avg", "n_trains", "scan_type", "xgm")
    assert run.summary["n_trains"] == 6
    assert run.summary["xgm"] == pytest.approx(1.5)
    assert run.summary["scan_type"] == "dscan th"
    assert bytes(run.summary["preview"]) == b"\x89PNG"
    assert len(run.sha256) == 64


def test_dataarray_with_coordinate_and_placeholder_dimensions(damnit_file):
    path, position, image, _ = damnit_file
    avg = load_damnit_run(path, sha256=False).read("avg")
    assert avg.dims == ("position", "dim_1", "dim_2")
    assert list(avg.coords) == ["position"]
    np.testing.assert_array_equal(avg.coords["position"], position)
    np.testing.assert_array_equal(avg.values, image)
    assert np.isnan(avg.values[0, 0, 0])


def test_per_train_variable_keeps_its_attributes(damnit_file):
    path, *_, trains = damnit_file
    xgm = load_damnit_run(path, sha256=False).read("xgm")
    assert xgm.dims == ("trainId",)
    np.testing.assert_array_equal(xgm.coords["trainId"], trains)
    assert xgm.attrs == {"max_diff": 1.0}


def test_plain_values(damnit_file):
    path, *_ = damnit_file
    run = load_damnit_run(path, sha256=False)
    assert run.read("n_trains").values == 6
    assert run.read("scan_type").values == "dscan th"
    with pytest.raises(KeyError, match="variables: avg"):
        run.read("missing")
