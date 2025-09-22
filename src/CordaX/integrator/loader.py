import os
from abc import ABC, abstractmethod
from pathlib import Path

import h5py
import hdf5plugin  # pylint: disable=unused-import
import numpy as np
import numpy.typing as npt
import pandas as pd

from ..config import ExpConfig, load_config
from ..config.enums import Hertz
from ..logger import Logger, setup_logger


class RawDataLoader(ABC):
    """
    Abstract base class for loading raw data from various sources.

    This class defines the interface for loading raw data using different strategies.
    Subclasses must implement the abstract methods to provide specific implementations.

    Attributes:
        file (str): The path to the raw data file.
    """

    @abstractmethod
    def __init__(self, file: str) -> None:
        """
        Initialize the RawDataLoader with the path to the raw data file.

        Args:
            file (str): The path to the raw data file.
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
    """Load hdf5 file and remove unmatching data."""

    def __init__(self, file: str):
        """
        Initializes the HDF5FileLoader by loading
        metadata, images, and qbpm data from the given file.

        Parameters:
        - file (str): Path to the HDF5 file.
        """
        if not os.path.exists(file):
            raise FileNotFoundError(f"No such file: {file}")
        self.logger: Logger = setup_logger()
        self.file: str = file
        self.config: ExpConfig = load_config()

        metadata: pd.DataFrame = pd.read_hdf(self.file, key="metadata")
        merged_df: pd.DataFrame = self._get_merged_df(metadata)
        if merged_df.empty:
            raise ValueError(f"No matching data found in {self.file}")

        # self.images: npt.NDArray[np.float32] = np.stack(merged_df["image"].values)
        # self.qbpm: npt.NDArray[np.float32] = np.stack(merged_df["qbpm"].values)
        self.images = np.array(merged_df["image"].tolist(), dtype=np.float32)
        self.qbpm = np.array(merged_df["qbpm"].tolist(), dtype=np.float32)
        self.pump_state: npt.NDArray[np.bool_] = self._get_pump_mask(merged_df)
        self.delay: npt.NDArray[np.float64] = self._get_delay(merged_df)

        self.logger.debug(
            f"Loaded {len(self.images)} images and {len(self.qbpm)} qbpm data."
        )
        self.logger.debug(
            f"Pump state distribution: {np.sum(self.pump_state)} on, {np.sum(~self.pump_state)} off."
        )

    def _get_merged_df(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Merges image and qbpm data with metadata based on timestamps.

        Parameters:
        - metadata (pd.DataFrame): Metadata DataFrame.

        Returns:
        - pd.DataFrame: Merged DataFrame containing metadata, images, and qbpm data.
        """
        with h5py.File(self.file, "r") as hf:
            if "detector" not in hf:
                raise KeyError(f"Key 'detector' not found in {self.file}")

            # DataFrame 생성
            image_group = hf[
                f"detector/{self.config.param.hutch.value}/{self.config.param.detector.value}/image"
            ]
            images_ts = np.array(image_group["block0_items"], dtype=np.int64)
            images = np.array(image_group["block0_values"], dtype=np.float64)
            qbpm_group = hf[f"qbpm/{self.config.param.hutch.value}/qbpm1"]
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
        image_df = pd.DataFrame({"image": list(images)}, index=images_ts)
        qbpm_df = pd.DataFrame({"qbpm": list(qbpm)}, index=qbpm_ts)
        merged_df = image_df.join(qbpm_df, how="inner")
        return metadata.join(merged_df, how="inner")
        # image_df = pd.DataFrame(
        #     {"timestamp": images_ts, "image": list(images)}
        # ).set_index("timestamp")
        # qbpm_df = pd.DataFrame({"timestamp": qbpm_ts, "qbpm": list(qbpm)}).set_index(
        #     "timestamp"
        # )
        # merged_df = pd.merge(
        #     image_df, qbpm_df, left_index=True, right_index=True, how="inner"
        # )
        # return pd.merge(
        #     metadata, merged_df, left_index=True, right_index=True, how="inner"
        # )

    def _get_delay(self, merged_df: pd.DataFrame) -> np.float64 | float:
        """
        Retrieves the delay value from the merged_df.

        Parameters:
        - merged_df (pd.DataFrame): merged_df DataFrame.

        Returns:
        - Union[np.float64, float]: Delay value or NaN if not found.
        """
        if "th_value" in merged_df:
            return np.asarray(merged_df["th_value"], dtype=np.float64)[0]
        if "delay_value" in merged_df:
            return np.asarray(merged_df["delay_value"], dtype=np.float64)[0]
        return np.nan

    def _get_pump_mask(self, merged_df: pd.DataFrame) -> npt.NDArray[np.bool_]:
        """
        Generates a pump status mask based on the configuration settings.

        Parameters:
        - merged_df (pd.DataFrame): Merged DataFrame.

        Returns:
        - npt.NDArray[np.bool_]: Pump status mask.
        """
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
        Retrieves data based on pump status.

        Returns:
        - dict[str, npt.NDArray]: Dictionary containing images and qbpm data for both pump-on and pump-off states.
        """
        data: dict[str, npt.NDArray] = {"delay": self.delay}

        poff_images = self.images[~self.pump_state]
        poff_qbpm = self.qbpm[~self.pump_state]
        pon_images = self.images[self.pump_state]
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

        images = hf[
            f"detector/{config.param.hutch.value}/{config.param.detector.value}/image/block0_values"
        ][:]

        return np.maximum(images, 0)


if __name__ == "__main__":
    import time

    from CordaX.filesystem import get_run_scan_dir

    config: ExpConfig = load_config()
    load_dir: str = config.path.load_dir
    print("load_dir:", load_dir)
    file: Path = get_run_scan_dir(load_dir, 163, 1, sub_path="p0050.h5")

    start = time.time()
    loader = PalXFELLoader(file)
    print(f"{time.time() - start} sec")
