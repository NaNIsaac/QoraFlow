import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Tuple, Dict

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import numpy.typing as npt
import pandas as pd

from ..config import ExpConfig, ConfigManager
from ..config.enums import Hertz
from ..logger import Logger, setup_logger

# Type alias
PreprocessorFunc = Callable[[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]


class RawDataLoader(ABC):
    """Abstract Base Class for Data Loaders."""
    
    @abstractmethod
    def __init__(self, file: Path | str) -> None: ...

    @abstractmethod
    def get_data(self) -> Dict[str, Any]: ...
    
    @abstractmethod
    def calculate_statistics(self, preprocessor: PreprocessorFunc) -> Dict[str, Any]: ...

    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()


class PalXFELLoader(RawDataLoader):
    """
    [Standard Loader for PAL-XFEL]
    - Sequential I/O Optimized: Reads full dataset to maximize disk throughput.
    - Handles Hertz.ZERO (All POFF) and Alignment logic.
    """

    def __init__(self, file: Path | str):
        self.file = Path(file)
        if not self.file.exists():
            raise FileNotFoundError(f"File not found: {self.file}")

        self.logger: Logger = setup_logger()
        self.config: ExpConfig = ConfigManager.load_config()

        self._cached_data: Dict[str, Any] | None = None
        self._image_path_str = f"detector/{self.config.param.hutch.value}/{self.config.param.detector.value}/image/block0_values"

        # 1. Load & Align Metadata
        try:
            self._merged_df = self._load_and_align_metadata()
        except Exception as e:
            raise ValueError(f"Metadata alignment failed in {self.file.name}: {e}")

        if self._merged_df.empty:
            raise ValueError(f"No valid frames found in {self.file.name}")

        # 2. Extract Info
        self.qbpm: npt.NDArray[np.float32] = self._merged_df["qbpm"].to_numpy(dtype=np.float32)
        self.pump_state: npt.NDArray[np.bool_] = self._get_pump_mask(self._merged_df)
        self.delay: float = self._get_delay(self._merged_df)

        # 3. Store Indices for filtering (Int64 for safe indexing)
        self.metadata_index: npt.NDArray[np.int64] = np.sort(
            self._merged_df["original_image_index"].values.astype(np.int64)
        )
        
        self.logger.debug(f"Initialized {self.file.name}: {len(self.metadata_index)} frames.")

    def close(self):
        self._cached_data = None

    def _load_and_align_metadata(self) -> pd.DataFrame:
        """Reads HDF5 structure and merges with pandas metadata."""
        # Read Pandas Metadata
        try:
            metadata: pd.DataFrame = pd.read_hdf(self.file, key="metadata")
        except (KeyError, FileNotFoundError):
             # 메타데이터 키가 없는 경우 빈 DF 반환 -> 상위에서 에러 처리
            return pd.DataFrame()

        # Filter required columns
        req_cols = []
        # Delay Check
        if "th_value" in metadata.columns: req_cols.append("th_value")
        elif "delay_value" in metadata.columns: req_cols.append("delay_value")
        
        # Pump Check
        if self.config.param.pump_setting is not Hertz.ZERO:
            pump_key = f"timestamp_info.RATE_{self.config.param.xray.value}_{self.config.param.pump_setting.value}"
            if pump_key in metadata.columns:
                req_cols.append(pump_key)

        subset_meta = metadata[req_cols].dropna() if req_cols else metadata

        # Read HDF5 raw timestamps & Calculate QBPM
        with h5py.File(self.file, "r") as hf:
            if "detector" not in hf:
                raise KeyError("Key 'detector' not found in HDF5")
            
            # Paths
            img_ts_path = f"detector/{self.config.param.hutch.value}/{self.config.param.detector.value}/image"
            qbpm_path = f"qbpm/{self.config.param.hutch.value}/qbpm1"

            # 1. Image Timestamps
            images_ts = hf[img_ts_path]["block0_items"][:].astype(np.int64)
            
            # 2. QBPM Calculation (Summing 4 channels)
            qbpm_grp = hf[qbpm_path]
            qbpm_ts = qbpm_grp["waveforms.ch1/axis1"][:].astype(np.int64)
            
            # Stack 4 channels and sum: (4, N, 1000) -> (N,)
            # Reading all channels at once is safer
            channels = [qbpm_grp[f"waveforms.ch{i+1}/block0_values"][:] for i in range(4)]
            qbpm_sum = np.sum(np.stack(channels, axis=0), axis=(0, 2))

        # Merge Logic
        image_ts_df = pd.DataFrame(
            {"original_image_index": np.arange(len(images_ts), dtype=np.int64)}, 
            index=images_ts
        )
        qbpm_df = pd.DataFrame({"qbpm": qbpm_sum}, index=qbpm_ts)

        # Join: Metadata <-> ImageTS <-> QBPM
        return subset_meta.join(image_ts_df.join(qbpm_df, how="inner"), how="inner")

    def _get_delay(self, df: pd.DataFrame) -> float:
        """Extracts representative delay value."""
        for key in ["th_value", "delay_value"]:
            if key in df:
                return float(df[key].iloc[0])
        return np.nan

    def _get_pump_mask(self, df: pd.DataFrame) -> npt.NDArray[np.bool_]:
        """Returns boolean mask for Pump-On images."""
        if self.config.param.pump_setting is Hertz.ZERO:
            return np.zeros(len(df), dtype=bool)
        
        key = f"timestamp_info.RATE_{self.config.param.xray.value}_{self.config.param.pump_setting.value}"
        if key in df:
            return df[key].to_numpy(dtype=bool)
        
        self.logger.warning(f"Pump key {key} missing. Treating as POFF.")
        return np.zeros(len(df), dtype=bool)

    def get_data(self) -> Dict[str, Any]:
        """
        Loads images into RAM. 
        **Optimized:** Uses Sequential Read (Full Load) instead of Fancy Indexing.
        """
        if self._cached_data is not None:
            return self._cached_data

        with h5py.File(self.file, "r") as hf:
            ds = hf[self._image_path_str]
            
            # [CRITICAL OPTIMIZATION]
            # Read the ENTIRE dataset sequentially first (Disk Speed)
            # Then filter by index in Memory (RAM Speed)
            full_volume = ds[:] 
            
            # Filter valid frames
            valid_images = full_volume[self.metadata_index]
            
            # Clean up raw volume from memory immediately
            del full_volume

        # Apply Pump Mask
        pon_imgs = valid_images[self.pump_state]
        poff_imgs = valid_images[~self.pump_state]
        
        # Apply ReLU (maximum(0, x))
        np.maximum(pon_imgs, 0, out=pon_imgs)
        np.maximum(poff_imgs, 0, out=poff_imgs)

        # Determine shapes for empty fallbacks
        h, w = (valid_images.shape[1], valid_images.shape[2]) if valid_images.ndim == 3 else (0, 0)
        dtype = valid_images.dtype

        data = {
            "delay": self.delay,
            "pon": pon_imgs if pon_imgs.size > 0 else np.zeros((0, h, w), dtype=dtype),
            "pon_qbpm": self.qbpm[self.pump_state],
            "poff": poff_imgs if poff_imgs.size > 0 else np.zeros((0, h, w), dtype=dtype),
            "poff_qbpm": self.qbpm[~self.pump_state],
        }

        self._cached_data = data
        return data

    def calculate_statistics(self, preprocessor: PreprocessorFunc) -> Dict[str, Any]:
        """Calculates Sum and Count for the loaded batch."""
        data = self.get_data()
        
        stats = {
            "delay": self.delay,
            "pon_sum": None, "pon_count": 0,
            "poff_sum": None, "poff_count": 0
        }

        # Helper for processing
        def process_state(key_img, key_qbpm, out_sum_key, out_count_key):
            if data[key_img].size > 0:
                proc_img, _ = preprocessor((data[key_img], data[key_qbpm]))
                if proc_img.size > 0:
                    stats[out_sum_key] = proc_img.sum(axis=0, dtype=np.float64)
                    stats[out_count_key] = proc_img.shape[0]

        process_state("pon", "pon_qbpm", "pon_sum", "pon_count")
        process_state("poff", "poff_qbpm", "poff_sum", "poff_count")

        return stats


# --- Compatibility Function ---
def get_hdf5_images(file: str | Path, config: ExpConfig) -> npt.NDArray:
    """Legacy function to get all images from HDF5."""
    with h5py.File(file, "r") as hf:
        if "detector" not in hf:
            raise KeyError(f"Key 'detector' not found in {file}")
        
        key = f"detector/{config.param.hutch.value}/{config.param.detector.value}/image/block0_values"
        images = hf[key][:] # Full read
        return np.maximum(images, 0)


# --- Main Test Block ---
if __name__ == "__main__":
    import time
    import gc
    import psutil
    from ..filesystem import get_run_scan_dir

    def get_current_memory_mb():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    ConfigManager.initialize("config.yaml")
    config = ConfigManager.load_config()

    # 테스트 파일 경로 설정 (예외 처리)
    try:
        # P0001.h5 등 테스트 파일 지정
        target_file = get_run_scan_dir(config.path.load_dir, 165, 1, sub_path="p0001.h5")
        if not target_file.exists():
            print(f"Test file not found: {target_file}")
            exit()
    except Exception as e:
        print(f"Path setup failed: {e}")
        exit()

    print(f"Target: {target_file}")
    print(f"Initial Memory: {get_current_memory_mb():.2f} MB")
    
    gc.collect()
    start_time = time.time()

    try:
        loader = PalXFELLoader(target_file)
        
        # Identity Preprocessor
        stats = loader.calculate_statistics(lambda x: x)

        elapsed = time.time() - start_time
        print(f"Time: {elapsed:.4f}s")
        print(f"PON Count: {stats['pon_count']}")
        print(f"POFF Count: {stats['poff_count']}")
        print(f"Final Memory: {get_current_memory_mb():.2f} MB")
        
    except Exception as e:
        print(f"Loader failed: {e}")