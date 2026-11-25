"""Read a DAMNIT extracted-data file: the per-run values a DAMNIT context file computed.

DAMNIT (European XFEL) runs the variables of a ``context.py`` on every run and stores them in
one HDF5 file per run, ``extracted_data/p<proposal>_r<run>.h5``. Layout of these files:

- One group per variable. A group with the attribute ``_damnit_objtype = "dataarray"`` holds an
  xarray ``DataArray`` written as netCDF4: the values in ``__xarray_dataarray_variable__`` and
  one dimension-scale dataset per dimension, attached through ``DIMENSION_LIST``. A dimension
  without coordinates is a scale whose ``NAME`` starts with "This is a netCDF dimension but not
  a netCDF variable"; its values are placeholders. Other groups hold a single ``data`` dataset
  (a number, a string or an array).
- ``.reduced`` holds one summary value per variable, the value DAMNIT shows in its table (a
  number, a string, or the bytes of a preview image).

Floating-point values keep the netCDF ``_FillValue`` (NaN) as stored. Strings are decoded as
UTF-8. Variables are read one at a time, so a run with large images opens quickly.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

DATAARRAY = "__xarray_dataarray_variable__"
PHONY_DIMENSION = b"This is a netCDF dimension but not a netCDF variable"
SUMMARY_GROUP = ".reduced"


@dataclass(frozen=True)
class DamnitVariable:
    """One variable of a run: values, dimension names and the coordinates that exist."""

    name: str
    values: npt.NDArray
    dims: tuple[str, ...] = ()
    coords: dict[str, npt.NDArray] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind in "OS":
        decoded = [_decode(v) for v in value.ravel()]
        return (
            decoded[0] if value.size == 1 else np.array(decoded, dtype=object).reshape(value.shape)
        )
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_dataarray(name: str, group: h5py.Group) -> DamnitVariable:
    dataset = group[DATAARRAY]
    dims, coords = [], {}
    for axis, refs in enumerate(dataset.attrs.get("DIMENSION_LIST", [])):
        scale = group.file[refs[0]]
        dim = scale.name.rsplit("/", 1)[-1]
        dims.append(dim)
        if not bytes(scale.attrs.get("NAME", b"")).startswith(PHONY_DIMENSION):
            if scale.shape != (dataset.shape[axis],):
                raise ValueError(
                    f"{name}: coordinate {dim} has shape {scale.shape}, axis {axis} has {dataset.shape[axis]}"
                )
            coords[dim] = _decode(scale[()])
    if len(dims) != dataset.ndim:
        raise ValueError(f"{name}: {dataset.ndim} axes but {len(dims)} dimensions attached")
    skip = {"DIMENSION_LIST", "_Netcdf4Dimid", "_Netcdf4Coordinates", "_FillValue"}
    attrs = {k: _decode(v) for k, v in dataset.attrs.items() if k not in skip}
    return DamnitVariable(name, _decode(dataset[()]), tuple(dims), coords, attrs)


@dataclass
class DamnitRun:
    """A DAMNIT extracted-data file: the variable names, the summary values and the file hash."""

    path: Path
    names: tuple[str, ...]
    summary: dict[str, Any]
    sha256: str | None = None

    def read(self, name: str) -> DamnitVariable:
        """Read one variable from the file."""
        if name not in self.names:
            raise KeyError(
                f"{name} is not a variable of {self.path.name}; variables: {', '.join(self.names)}"
            )
        with h5py.File(self.path, "r") as handle:
            group = handle[name]
            if group.attrs.get("_damnit_objtype") == "dataarray":
                return _read_dataarray(name, group)
            if "data" not in group:
                raise ValueError(f"{name}: neither a data array nor a 'data' dataset")
            return DamnitVariable(name, _decode(group["data"][()]))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def load_damnit_run(path: Path | str, *, sha256: bool = True) -> DamnitRun:
    """Open a DAMNIT extracted-data file: list its variables and read the summary values.

    The values of each variable are read later with :meth:`DamnitRun.read`. With ``sha256``
    the hash of the whole file is computed, so a result can name the exact input.
    """
    path = Path(path)
    with h5py.File(path, "r") as handle:
        names = tuple(
            sorted(k for k in handle if k != SUMMARY_GROUP and isinstance(handle[k], h5py.Group))
        )
        summary = {}
        if SUMMARY_GROUP in handle:
            summary = {k: _decode(v[()]) for k, v in handle[SUMMARY_GROUP].items()}
    return DamnitRun(path, names, summary, _file_sha256(path) if sha256 else None)
