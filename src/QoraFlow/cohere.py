"""Read a Cohere phasing project: the prepared diffraction data, its configuration and results.

A Cohere project directory holds ``conf/config*`` files, ``preprocessed_data/prep_data.tif``
(the detector frames stacked over the rocking scan), ``phasing_data/data.tif`` (the array that
goes into phase retrieval) and one or more result directories with ``image.npy``,
``support.npy`` and ``errors.npy`` or ``errors.txt``.

Conventions taken from the ``cohere_core`` 4.0.0 source; the tif functions are the same in 3.2
and in the 4.5 series:

- ``utils.save_tif`` writes ``arr.transpose()`` and ``utils.read_tif`` transposes back, so the
  axis order inside Cohere is the reverse of the order in the tif file. The arrays returned
  here are in Cohere's order, the one ``image.npy`` and ``support.npy`` are saved in.
- ``phasing_data/data.tif`` holds amplitudes: the standard preprocessing zeroes values at or
  below ``intensity_threshold``, takes the square root, crops or pads, moves the maximum to the
  array centre and bins. The data are centre-origin.
- Phasing works on ``fftshift(data)``; the far field of an object is ``ifftn(object)``, the
  opposite sign to ``fftn``. A reconstruction made with the ``fftn`` convention from the same
  data is therefore the conjugate-flipped twin of Cohere's, which an ambiguity-aware comparison
  resolves along with translation and global phase.
- Zeros in ``data.tif`` are treated by Cohere as measured zeros; there is no mask.
"""

import ast
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import tifffile

CONFIG_NAMES = ("config", "config_prep", "config_data", "config_instr", "config_rec", "config_disp")


def read_cohere_config(path: Path | str) -> dict[str, Any]:
    """Parse a Cohere configuration file of ``name = python-literal`` lines.

    Lines starting with ``/`` or ``#`` are comments. A value written as a tuple becomes a list,
    as in Cohere. A value that is not a Python literal raises ``ValueError``, which Cohere does
    too, so any file Cohere accepted parses here.
    """
    path = Path(path)
    params: dict[str, Any] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith(("/", "#")) or "=" not in line:
            continue
        name, _, value = line.partition("=")
        value = value.strip()
        if value.startswith("("):
            value = value.replace("(", "[").replace(")", "]")
        try:
            params[name.strip()] = ast.literal_eval(value)
        except (ValueError, SyntaxError) as error:
            raise ValueError(
                f"{path}:{number}: {name.strip()} = {value} is not a literal"
            ) from error
    return params


def read_cohere_tif(path: Path | str) -> npt.NDArray:
    """Read a tif written by Cohere, in Cohere's axis order (the file's order reversed)."""
    return tifffile.imread(path).transpose()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CohereProject:
    """The contents of a Cohere project directory.

    ``amplitude`` is ``phasing_data/data.tif``. ``prep_data`` is the stacked rocking scan from
    ``preprocessed_data/prep_data.tif`` before thresholding and the square root. ``image``,
    ``support`` and ``errors`` come from the chosen result directory. Arrays are in Cohere's
    axis order; anything absent from the project is ``None``. ``provenance`` maps each file read,
    relative to ``root``, to its size and SHA-256.
    """

    root: Path
    amplitude: npt.NDArray[np.float32]
    configs: dict[str, dict[str, Any]]
    prep_data: npt.NDArray | None = None
    image: npt.NDArray[np.complexfloating] | None = None
    support: npt.NDArray | None = None
    errors: npt.NDArray | None = None
    provenance: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def intensity(self) -> npt.NDArray[np.float64]:
        """Centre-origin intensity, the square of the phasing amplitudes."""
        return np.square(self.amplitude, dtype=np.float64)


def load_cohere_project(
    root: Path | str, results: str = "results_phasing", *, prep: bool = True
) -> CohereProject:
    """Load a Cohere project from its directory.

    ``results`` names the result directory under ``root``. ``prep=False`` skips
    ``prep_data.tif``, which can be much larger than the phasing data.
    """
    root = Path(root)
    data_file = root / "phasing_data" / "data.tif"
    if not data_file.is_file():
        raise FileNotFoundError(f"{data_file} not found; is {root} a Cohere project?")

    provenance: dict[str, dict[str, Any]] = {}

    def record(path: Path) -> Path:
        provenance[path.relative_to(root).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        return path

    def optional(path: Path, reader):
        return reader(record(path)) if path.is_file() else None

    configs = {
        name: read_cohere_config(record(root / "conf" / name))
        for name in CONFIG_NAMES
        if (root / "conf" / name).is_file()
    }
    amplitude = read_cohere_tif(record(data_file))
    prep_data = (
        optional(root / "preprocessed_data" / "prep_data.tif", read_cohere_tif) if prep else None
    )

    result_dir = root / results
    errors = optional(result_dir / "errors.npy", np.load)
    if errors is None:
        errors = optional(result_dir / "errors.txt", np.loadtxt)

    return CohereProject(
        root=root,
        amplitude=amplitude,
        configs=configs,
        prep_data=prep_data,
        image=optional(result_dir / "image.npy", np.load),
        support=optional(result_dir / "support.npy", np.load),
        errors=errors,
        provenance=provenance,
    )
