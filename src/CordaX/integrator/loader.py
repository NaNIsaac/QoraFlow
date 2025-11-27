from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional

import h5py
import hdf5plugin  # pylint: disable=unused-import
import numpy as np
import numpy.typing as npt
import pandas as pd

from ..config import ExpConfig, ConfigManager
from ..config.enums import Hertz
from ..logger import Logger, setup_logger


class RawDataLoader(ABC):
    """
    Abstract base class for loading raw data from various sources.

    This class defines the interface for loading raw data using different strategies.
    Subclasses must implement the abstract methods to provide specific implementations.
    """

    @abstractmethod
    def __init__(self, file: Path | str) -> None:
        """
        Initialize the RawDataLoader with the path to the raw data file.
        """

    @abstractmethod
    def get_data(self) -> dict[str, npt.NDArray]:
        """
        Retrieve the raw data as a dictionary of numpy arrays.

        Returns:
            dict[str, npt.NDArray]: A dictionary where keys are data identifiers
            and values are numpy arrays containing the raw data.
        """


class PalXFELLoader(RawDataLoader):
    """Load hdf5 file and remove unmatching data, supporting lazy image loading."""

    def __init__(self, file: Path | str):
        """
        Initializes the Loader. Data arrays (images) are loaded lazily in get_data().
        """
        self.file: Path = Path(file)
        if not self.file.exists():
            raise FileNotFoundError(f"No such file: {str(self.file)}")

        self.logger: Logger = setup_logger()
        self.config: ExpConfig = ConfigManager.load_config()

        self._h5_file = h5py.File(self.file, "r")
        self._image_path = f"detector/{self.config.param.hutch.value}/{self.config.param.detector.value}/image"
        self._qbpm_path = f"qbpm/{self.config.param.hutch.value}/qbpm1"
        
        metadata: pd.DataFrame = pd.read_hdf(self.file, key="metadata")
        merged_df: pd.DataFrame = self._get_metadata_and_qbpm_df(metadata)
        
        if merged_df.empty:
            self._h5_file.close()
            raise ValueError(f"No matching data found in {self.file}")

        self.qbpm: npt.NDArray[np.float32] = np.array(merged_df["qbpm"].tolist(), dtype=np.float32)
        self.pump_state: npt.NDArray[np.bool_] = self._get_pump_mask(merged_df)
        self.delay: np.float64 | float = self._get_delay(merged_df)
        self.metadata_index: npt.NDArray[np.int64] = merged_df.index.values.astype(np.int64)
        
        self.logger.debug(
            f"Initialized {len(self.metadata_index)} indices for images and {len(self.qbpm)} qbpm data."
        )

    def __del__(self):
        """Ensure h5py file handle is closed when the object is destroyed."""
        if hasattr(self, "_h5_file") and self._h5_file:
            self._h5_file.close()

    def _get_metadata_and_qbpm_df(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Merges metadata, image timestamps, and qbpm data based on timestamps.
        Critically, it loads only the QBPM *values* into memory, not images.
        """
        hf = self._h5_file # 이미 __init__에서 열린 핸들 사용

        if "detector" not in hf:
            raise KeyError(f"Key 'detector' not found in {self.file}")

        # 이미지 타임스탬프만 로드
        image_group = hf[self._image_path]
        images_ts = np.array(image_group["block0_items"], dtype=np.int64)

        qbpm_group = hf[self._qbpm_path]
        qbpm_ts = np.array(qbpm_group["waveforms.ch1/axis1"], dtype=np.int64)
        qbpm = np.sum(
            np.stack(
                [
                    qbpm_group[f"waveforms.ch{i + 1}/block0_values"]
                    for i in range(4)
                ],
                axis=0,
                dtype=np.float32,
            ),
            axis=(0, 2),
        )
        
        # 이미지 타임스탬프만 있는 DF 생성 (images 필드는 없고, index만 있음)
        image_ts_df = pd.DataFrame(index=images_ts, data={'temp_key': 0})
        qbpm_df = pd.DataFrame({"qbpm": list(qbpm)}, index=qbpm_ts)

        merged_df = image_ts_df.join(qbpm_df, how="inner").drop(columns=['temp_key'])
        
        # 메타데이터와 최종 병합
        return metadata.join(merged_df, how="inner")

    def _get_delay(self, merged_df: pd.DataFrame) -> np.float64 | float:
        """Retrieves the delay value."""
        if "th_value" in merged_df:
            return np.asarray(merged_df["th_value"], dtype=np.float64)[0]
        if "delay_value" in merged_df:
            return np.asarray(merged_df["delay_value"], dtype=np.float64)[0]
        return np.nan

    def _get_pump_mask(self, merged_df: pd.DataFrame) -> npt.NDArray[np.bool_]:
        """Generates a pump status mask."""
        if self.config.param.pump_setting is Hertz.ZERO:
            return np.zeros(merged_df.shape[0], dtype=np.bool_)
        return np.asarray(
            merged_df[
                f"timestamp_info.RATE_{self.config.param.xray.value}_{self.config.param.pump_setting.value}"
            ],
            dtype=np.bool_,
        )

    def get_data(self) -> dict[str, npt.NDArray]:
        """
        Retrieves data by loading images from HDF5 lazily.
        Returns: Images and qbpm data for both pump-on and pump-off states.
        """
        image_block_data = self._h5_file[self._image_path]["block0_values"]
        images: npt.NDArray = image_block_data[self.metadata_index]
        
        data: dict[str, npt.NDArray] = {"delay": self.delay}

        poff_images = images[~self.pump_state]
        poff_qbpm = self.qbpm[~self.pump_state]
        pon_images = images[self.pump_state]
        pon_qbpm = self.qbpm[self.pump_state]

        poff_images = np.maximum(0, poff_images)
        pon_images = np.maximum(0, pon_images)

        if poff_images.size > 0:
            data["poff"] = poff_images
            data["poff_qbpm"] = poff_qbpm
        if pon_images.size > 0:
            data["pon"] = pon_images
            data["pon_qbpm"] = pon_qbpm

        return data


def get_hdf5_images(file: str, config: ExpConfig) -> npt.NDArray:
    """get images form hdf5"""
    with h5py.File(file, "r") as hf:
        if "detector" not in hf:
            raise KeyError(f"Key 'detector' not found in {file}")
        key: str = f"detector/{config.param.hutch.value}/{config.param.detector.value}/image/block0_values"
        images: np.ndarray = hf[key][:]

        return np.maximum(images, 0)


if __name__ == "__main__":
    import time

    from CordaX.filesystem import get_run_scan_dir

    config: ExpConfig = ConfigManager.load_config()
    load_dir: Path = config.path.load_dir
    print("load_dir:", load_dir)
    file: Path = get_run_scan_dir(load_dir, 163, 1, sub_path="p0050.h5")

    start = time.time()
    loader = PalXFELLoader(file) 
    print(f"Initialization Time: {time.time() - start:.4f} sec")
    
    start_data = time.time()
    data = loader.get_data()
    print(f"Data Loading Time: {time.time() - start_data:.4f} sec")

    print(f"POFF shape: {data['poff'].shape}")
    print(f"PON shape: {data['pon'].shape}")