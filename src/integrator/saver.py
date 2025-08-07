from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.io import savemat
import tifffile

from src.config import ExpConfig, load_config
from src.filesystem import make_run_scan_dir
from src.mathematics import axis_np2mat


def get_file_base_name(run_n: int, scan_n: int) -> str:
    """Return formated file name"""
    return f"run={run_n:04}_scan={scan_n:04}"


class SaverStrategy(ABC):
    """Save data_dict to a file."""

    @abstractmethod
    def save(
        self,
        run_n: int,
        scan_n: int,
        data_dict: dict[str, npt.NDArray],
        comment: str = "",
    ):
        pass

    @property
    @abstractmethod
    def file(self) -> str:
        """Return File Name"""

    @property
    @abstractmethod
    def file_type(self) -> str:
        """Return File Type"""


class MatSaverStrategy(SaverStrategy):

    def __init__(self):
        self._file: str = None

    def save(
        self,
        run_n: int,
        scan_n: int,
        data_dict: dict[str, npt.NDArray],
        comment: str = "",
    ):
        comment = "_" + comment if comment else ""
        config: ExpConfig = load_config()
        mat_dir: Path = config.path.mat_dir
        mat_dir.mkdir(parents=True, exist_ok=True)
        for key, val in data_dict.items():
            if val.ndim == 3:
                mat_format_images = axis_np2mat(val)
                file_base_name = get_file_base_name(run_n, scan_n)
                mat_file: Path = mat_dir / f"{file_base_name}_{key}{comment}.mat"
                savemat(mat_file, {"data": mat_format_images})
        self._file = mat_file

    @property
    def file(self) -> str:
        return self._file

    @property
    def file_type(self) -> str:
        return "mat"


class NpzSaverStrategy(SaverStrategy):

    def __init__(self):
        self._file: str = None

    def save(
        self,
        run_n: int,
        scan_n: int,
        data_dict: dict[str, npt.NDArray],
        comment: str = "",
    ):
        comment = "_" + comment if comment else ""
        config = load_config()
        processed_dir = config.path.processed_dir
        processed_dir.mkdir(parents=True, exist_ok=True)
        file_name = get_file_base_name(run_n, scan_n) + comment + ".npz"
        npz_file: Path = make_run_scan_dir(
            processed_dir, run_n, scan_n, sub_path=file_name
        )
        np.savez(npz_file, **data_dict)
        self._file = npz_file

    @property
    def file(self) -> str:
        return self._file

    @property
    def file_type(self) -> str:
        return "npz"


class TifSaverStrategy(SaverStrategy):
    def save(
        self,
        run_n: int,
        scan_n: int,
        data_dict: dict[str, npt.NDArray],
        comment: str = "",
    ):
        comment = "_" + comment if comment else ""
        config = load_config()
        processed_dir = config.path.processed_dir
        processed_dir.mkdir(parents=True, exist_ok=True)
        file_name = get_file_base_name(run_n, scan_n) + comment + ".tif"
        tif_file: Path = make_run_scan_dir(
            processed_dir, run_n, scan_n, sub_path=file_name
        )
        tifffile.imsave(
            "output_with_metadata.tif",
            data[0],  # Save a single frame
            imagej=True,  # For ImageJ compatibility
            metadata={"spacing": 0.5, "unit": "um", "axes": "YX"},
        )

    @property
    def file(self) -> str:
        """Return File Name"""
        return self._file

    @property
    def file_type(self) -> str:
        """Return File Type"""
        return "npz"


def get_saver_strategy(file_type: str) -> SaverStrategy:
    """Get SaverStrategy by file type."""
    match file_type:
        case "mat":
            return MatSaverStrategy()
        case "npz":
            return NpzSaverStrategy()
        case "tiff" | "tif":
            return TifSaverStrategy()
        case _:
            raise ValueError(f"Unsupported file type: {file_type}")
