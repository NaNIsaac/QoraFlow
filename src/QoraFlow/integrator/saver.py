from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.io import savemat
import tifffile

from ..config import ExpConfig, ConfigManager
from ..filesystem import make_run_scan_dir
from ..mathematics import axis_np2mat


def get_file_base_name(run_n: int, scan_n: int) -> str:
    """Return formatted file name: run=XXXX_scan=XXXX"""
    return f"run={run_n:04d}_scan={scan_n:04d}"


class SaverStrategy(ABC):
    """Abstract Base Class for saving data dictionaries to files."""

    def __init__(self) -> None:
        self._file: Optional[Path] = None
        self.config: ExpConfig = ConfigManager.load_config()

    @abstractmethod
    def save(
        self,
        run_n: int,
        scan_n: int,
        data_dict: dict[str, npt.NDArray],
        comment: str = "",
    ) -> None:
        pass

    @property
    def file(self) -> str:
        """Return the path of the last saved file as a string."""
        return str(self._file) if self._file else ""

    @property
    @abstractmethod
    def file_type(self) -> str:
        """Return File Type extension (e.g., 'npz')"""
        pass

    def _get_formatted_comment(self, comment: str) -> str:
        """Helper to format the comment with a leading underscore if it exists."""
        return f"_{comment}" if comment else ""


class MatSaverStrategy(SaverStrategy):
    """Saves 3D arrays from the data dictionary as individual .mat files."""

    @property
    def file_type(self) -> str:
        return "mat"

    def save(
        self,
        run_n: int,
        scan_n: int,
        data_dict: dict[str, npt.NDArray],
        comment: str = "",
    ) -> None:
        comment_str = self._get_formatted_comment(comment)
        mat_dir: Path = self.config.path.mat_dir
        mat_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = get_file_base_name(run_n, scan_n)

        for key, val in data_dict.items():
            # Mat files often require specific axis ordering (e.g., for MATLAB)
            if val.ndim == 3:
                mat_format_images = axis_np2mat(val)
                file_name = f"{base_name}_{key}{comment_str}.mat"
                save_path = mat_dir / file_name
                
                savemat(save_path, {"data": mat_format_images})
                self._file = save_path


class NpzSaverStrategy(SaverStrategy):
    """Saves the entire dictionary into a single compressed .npz file."""

    @property
    def file_type(self) -> str:
        return "npz"

    def save(
        self,
        run_n: int,
        scan_n: int,
        data_dict: dict[str, npt.NDArray],
        comment: str = "",
    ) -> None:
        comment_str = self._get_formatted_comment(comment)
        processed_dir = self.config.path.processed_dir
        
        file_name = f"{get_file_base_name(run_n, scan_n)}{comment_str}.npz"
        
        # make_run_scan_dir handles directory creation
        npz_file: Path = make_run_scan_dir(
            processed_dir, run_n, scan_n, sub_path=file_name
        )
        
        np.savez(npz_file, **data_dict)
        self._file = npz_file


class TifSaverStrategy(SaverStrategy):
    """Saves arrays from the data dictionary as individual .tif files."""

    @property
    def file_type(self) -> str:
        return "tif"

    def save(
        self,
        run_n: int,
        scan_n: int,
        data_dict: dict[str, npt.NDArray],
        comment: str = "",
    ) -> None:
        comment_str = self._get_formatted_comment(comment)
        processed_dir = self.config.path.processed_dir
        
        base_name = get_file_base_name(run_n, scan_n)

        # Iterate through dict to find saveable arrays
        for key, val in data_dict.items():
            # Skip scalar values or empty arrays if necessary, currently saving all arrays
            file_name = f"{base_name}_{key}{comment_str}.tif"
            
            tif_file: Path = make_run_scan_dir(
                processed_dir, run_n, scan_n, sub_path=file_name
            )

            # Metadata for ImageJ
            metadata = {"spacing": 0.5, "unit": "um", "axes": "YX"}
            
            # Ensure proper float32 casting if specific to your pipeline, 
            # otherwise save raw. Added check to ensure valid array.
            if isinstance(val, np.ndarray):
                 # If 3D, ImageJ expects (Z, Y, X) usually.
                tifffile.imwrite(
                    tif_file,
                    val,
                    imagej=True,
                    metadata=metadata,
                )
                self._file = tif_file


def get_saver_strategy(file_type: str) -> SaverStrategy:
    """Factory function to get SaverStrategy by file type."""
    # Normalize input to lowercase to prevent casing errors
    match file_type.lower():
        case "mat":
            return MatSaverStrategy()
        case "npz":
            return NpzSaverStrategy()
        case "tiff" | "tif":
            return TifSaverStrategy()
        case _:
            raise ValueError(f"Unsupported file type: {file_type}")