from abc import ABC, abstractmethod
from pathlib import Path

import h5py
import hdf5plugin  # pylint: disable=unused-import
import numpy as np
import numpy.typing as npt
import pandas as pd

from ..config import ExpConfig, ConfigManager
from ..config.enums import Hertz
from ..logger import Logger, setup_logger


class RawDataLoader(ABC):
    @abstractmethod
    def __init__(self, file: Path | str) -> None:
        """Initialize"""

    @abstractmethod
    def get_data(self) -> dict[str, npt.NDArray]:
        """Retrieve data"""
    
    def close(self):
        """Optional close method"""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


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
            self.close()
            raise ValueError(f"No matching data found in {self.file}")

        self.qbpm: npt.NDArray[np.float32] = np.array(merged_df["qbpm"].tolist(), dtype=np.float32)
        self.pump_state: npt.NDArray[np.bool_] = self._get_pump_mask(merged_df)
        self.delay: np.float64 | float = self._get_delay(merged_df)

        raw_indices = merged_df["original_image_index"].values.astype(np.int64)
        self.metadata_index: npt.NDArray[np.int64] = np.sort(raw_indices)

        self.logger.debug(
            f"Initialized {len(self.metadata_index)} indices for images and {len(self.qbpm)} qbpm data."
        )

    def close(self):
            """Explicitly close the HDF5 file handle."""
            if hasattr(self, "_h5_file") and self._h5_file:
                try:
                    self._h5_file.close()
                except Exception:
                    pass
                self._h5_file = None

    def __del__(self):
            self.close()

    def _get_metadata_and_qbpm_df(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Merges metadata, image timestamps, and qbpm data based on timestamps.
        Critically, it loads only the QBPM *values* into memory, not images.
        """
        hf = self._h5_file

        if "detector" not in hf:
            raise KeyError(f"Key 'detector' not found in {self.file}")

        image_group = hf[self._image_path]
        images_ts = np.array(image_group["block0_items"], dtype=np.int64)

        qbpm_group = hf[self._qbpm_path]
        qbpm_ts = np.array(qbpm_group["waveforms.ch1/axis1"], dtype=np.int64)
        qbpm = np.sum(
            np.stack(
                [qbpm_group[f"waveforms.ch{i + 1}/block0_values"] for i in range(4)],
                axis=0,
                dtype=np.float32,
            ),
            axis=(0, 2),
        )

        image_ts_df = pd.DataFrame(
            {
                "original_image_index": np.arange(len(images_ts), dtype=np.int64)
            }, 
            index=images_ts
        )
        qbpm_df = pd.DataFrame({"qbpm": list(qbpm)}, index=qbpm_ts)

        merged_df = image_ts_df.join(qbpm_df, how="inner")

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
        if not self._h5_file:
            self._h5_file = h5py.File(self.file, "r")

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

    start_total = time.time()
    # 'with' 블록을 나가면 자동으로 loader.close()가 호출됨
    with PalXFELLoader(file) as loader:
        print(f"Initialization Time: {time.time() - start_total:.4f} sec")
        
        start_load = time.time()
        data = loader.get_data()
        print(f"Data Loading Time: {time.time() - start_load:.4f} sec")

        # 데이터 확인
        if 'poff' in data:
            print(f"POFF Images Shape: {data['poff'].shape}")
        if 'pon' in data:
            print(f"PON Images Shape: {data['pon'].shape}")
        print(f"Delay: {data['delay']}")

    print(f"POFF shape: {data['poff'].shape}")
    print(f"PON shape: {data['pon'].shape}")