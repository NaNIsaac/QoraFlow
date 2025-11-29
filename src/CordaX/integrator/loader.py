from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence, Union

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import numpy.typing as npt
import pandas as pd

from ..config import ExpConfig, ConfigManager
from ..config.enums import Hertz
from ..logger import Logger, setup_logger
from ..preprocessor.image_qbpm_preprocessor import ImagesQbpmProcessor

class LazyImageSequence(Sequence):
    """Proxy object for HDF5 dataset to enable lazy loading via slicing."""
    def __init__(self, h5_file_path: Path, dataset_path: str, indices: np.ndarray):
        self.h5_file_path = h5_file_path
        self.dataset_path = dataset_path
        self.indices = indices
        self._shape = None
        self._dtype = None

    def _load_meta_info(self):
        with h5py.File(self.h5_file_path, "r") as f:
            ds = f[self.dataset_path]
            self._shape = (len(self.indices),) + ds.shape[1:]
            self._dtype = ds.dtype

    @property
    def shape(self) -> tuple:
        if self._shape is None: self._load_meta_info()
        return self._shape

    @property
    def dtype(self):
        if self._dtype is None: self._load_meta_info()
        return self._dtype

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, key: Union[int, slice, np.ndarray]) -> np.ndarray:
        if isinstance(key, slice):
            target_indices = self.indices[key]
        elif isinstance(key, int):
            if key < 0: key += len(self)
            if key >= len(self) or key < 0: raise IndexError("Index out of range")
            target_indices = [self.indices[key]]
        elif isinstance(key, (np.ndarray, list)):
            target_indices = self.indices[key]
        else:
            raise TypeError(f"Invalid index type: {type(key)}")

        if len(target_indices) == 0:
            return np.empty((0,) + self.shape[1:], dtype=self.dtype)

        with h5py.File(self.h5_file_path, "r") as f:
            ds = f[self.dataset_path]
            loaded_data = ds[target_indices] if isinstance(target_indices, np.ndarray) else ds[target_indices]
                
        if isinstance(key, int): return loaded_data[0]
        return loaded_data


class RawDataLoader(ABC):
    @abstractmethod
    def __init__(self, file: Path | str) -> None: ...

    @abstractmethod
    def get_data(self) -> dict[str, Any]: ...
    
    @abstractmethod
    def calculate_statistics(
        self, 
        preprocessor: ImagesQbpmProcessor, 
        chunk_size: int = 100
    ) -> dict[str, Any]:
        """
        Calculates Sum and Count for the file using the given preprocessor.
        """
        pass

    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()


class PalXFELLoader(RawDataLoader):
    def __init__(self, file: Path | str):
        self.file: Path = Path(file)
        if not self.file.exists():
            raise FileNotFoundError(f"No such file: {str(self.file)}")

        self.logger: Logger = setup_logger()
        self.config: ExpConfig = ConfigManager.load_config()

        # Load Metadata
        metadata: pd.DataFrame = pd.read_hdf(self.file, key="metadata")
        self._merged_df: pd.DataFrame = self._get_valid_df(metadata)

        if self._merged_df.empty:
            raise ValueError(f"No matching data found in {self.file}")

        self.qbpm: npt.NDArray[np.float32] = np.array(self._merged_df["qbpm"].tolist(), dtype=np.float32)
        self.pump_state: npt.NDArray[np.bool_] = self._get_pump_mask(self._merged_df)
        self.delay: float = self._get_delay(self._merged_df)

        raw_indices = self._merged_df["original_image_index"].values.astype(np.int64)
        self.metadata_index: npt.NDArray[np.int64] = np.sort(raw_indices)
        self._image_path_str = f"detector/{self.config.param.hutch.value}/{self.config.param.detector.value}/image/block0_values"

        self.logger.debug(f"Initialized {len(self.metadata_index)} frames.")

    def close(self): pass

    def _get_valid_df(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        with h5py.File(self.file, "r") as hf:
            if "detector" not in hf: raise KeyError(f"Key 'detector' not found")
            image_path = f"detector/{self.config.param.hutch.value}/{self.config.param.detector.value}/image"
            qbpm_path = f"qbpm/{self.config.param.hutch.value}/qbpm1"

            images_ts = np.array(hf[image_path]["block0_items"], dtype=np.int64)
            qbpm_ts = np.array(hf[qbpm_path]["waveforms.ch1/axis1"], dtype=np.int64)
            qbpm_group = hf[qbpm_path]
            qbpm = np.sum(np.stack([qbpm_group[f"waveforms.ch{i+1}/block0_values"][:] for i in range(4)], axis=0), axis=(0, 2))

        image_ts_df = pd.DataFrame({"original_image_index": np.arange(len(images_ts), dtype=np.int64)}, index=images_ts)
        qbpm_df = pd.DataFrame({"qbpm": list(qbpm)}, index=qbpm_ts)
        return metadata_df.join(image_ts_df.join(qbpm_df, how="inner"), how="inner")

    def _get_delay(self, merged_df: pd.DataFrame) -> float:
        for key in ["th_value", "delay_value"]:
            if key in merged_df:
                vals = merged_df[key].dropna()
                if not vals.empty: return float(vals.iloc[0])
        return np.nan

    def _get_pump_mask(self, merged_df: pd.DataFrame) -> npt.NDArray[np.bool_]:
        if self.config.param.pump_setting is Hertz.ZERO:
            return np.zeros(len(merged_df), dtype=np.bool_)
        key = f"timestamp_info.RATE_{self.config.param.xray.value}_{self.config.param.pump_setting.value}"
        return np.asarray(merged_df[key], dtype=np.bool_) if key in merged_df else np.zeros(len(merged_df), dtype=np.bool_)

    def get_data(self) -> dict[str, Any]:
        """Returns Lazy Objects."""
        data: dict[str, Any] = {"delay": self.delay}
        pon_mask = self.pump_state
        poff_mask = ~self.pump_state

        if np.any(poff_mask):
            data["poff"] = LazyImageSequence(self.file, self._image_path_str, self.metadata_index[poff_mask])
            data["poff_qbpm"] = self.qbpm[poff_mask]
        if np.any(pon_mask):
            data["pon"] = LazyImageSequence(self.file, self._image_path_str, self.metadata_index[pon_mask])
            data["pon_qbpm"] = self.qbpm[pon_mask]
        return data

    # [NEW FEATURE] Calculate Statistics internally
    def calculate_statistics(
        self, 
        preprocessor: ImagesQbpmProcessor, 
        chunk_size: int = 100
    ) -> dict[str, Any]:
        """
        Process the file internally in chunks and return the SUM and COUNT.
        This allows single-file analysis and reduces overhead in CoreIntegrator.
        """
        lazy_data = self.get_data()
        stats = {
            "delay": self.delay,
            "pon_sum": None, "pon_count": 0,
            "poff_sum": None, "poff_count": 0
        }

        # Helper to process a specific state (pon/poff)
        def process_state(state_key: str, sum_key: str, count_key: str):
            if state_key not in lazy_data: 
                return
            
            lazy_imgs = lazy_data[state_key]
            full_qbpm = lazy_data[f"{state_key}_qbpm"]
            total_frames = len(lazy_imgs)

            for start_idx in range(0, total_frames, chunk_size):
                end_idx = min(start_idx + chunk_size, total_frames)
                s = slice(start_idx, end_idx)

                # Load Chunk
                img_chunk = lazy_imgs[s]
                qbpm_chunk = full_qbpm[s]

                # Preprocess
                proc_imgs, _ = preprocessor((img_chunk, qbpm_chunk))

                # Accumulate
                if proc_imgs.size > 0:
                    chunk_sum = proc_imgs.sum(axis=0, dtype=np.float64)
                    chunk_cnt = proc_imgs.shape[0]

                    if stats[sum_key] is None:
                        stats[sum_key] = chunk_sum
                    else:
                        stats[sum_key] += chunk_sum
                    stats[count_key] += chunk_cnt

        # Run for both states
        process_state("pon", "pon_sum", "pon_count")
        process_state("poff", "poff_sum", "poff_count")

        return stats


def get_hdf5_images(file: Path | str, config: ExpConfig) -> npt.NDArray:

    """get images form hdf5"""

    with h5py.File(file, "r") as hf:

        if "detector" not in hf:

            raise KeyError(f"Key 'detector' not found in {file}")

        key: str = f"detector/{config.param.hutch.value}/{config.param.detector.value}/image/block0_values"

        images: np.ndarray = hf[key][:]

        return np.maximum(images, 0)


if __name__ == "__main__":
    import time
    import os
    import gc
    import psutil
    from ..filesystem import get_run_scan_dir

    # 메모리 측정 헬퍼 함수 (MB 단위)
    def get_current_memory_mb():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    # 설정 로드
    ConfigManager.initialize("config.yaml")
    config = ConfigManager.load_config()
    
    # 테스트할 파일 경로 (환경에 맞게 수정)
    try:
        file: Path = get_run_scan_dir(config.path.load_dir, 165, 1, sub_path="p0001.h5")
        if not file.exists():
            print(f"File not found: {file}")
            exit()
    except Exception as e:
        print(f"Path Error: {e}")
        exit()

    print(f"Target File: {file}")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Legacy Way: Load All -> Calculate Mean
    # ---------------------------------------------------------
    print("[1] Legacy Way (Load All -> Mean)")
    gc.collect()  # 이전 메모리 정리
    mem_before = get_current_memory_mb()
    start_time = time.time()

    try:
        # 모든 이미지를 RAM에 로드 (메모리 스파이크 발생 지점)
        images = get_hdf5_images(file, config)
        
        # 평균 계산
        if images.size > 0:
            mean_legacy = images.mean(axis=0)
        
        end_time = time.time()
        mem_after = get_current_memory_mb()
        
        print(f"  - Execution Time : {end_time - start_time:.4f} sec")
        print(f"  - Memory Usage   : +{mem_after - mem_before:.2f} MB (Total: {mem_after:.2f} MB)")
        print(f"  - Result Shape   : {mean_legacy.shape}")

        # 메모리 해제
        del images
        del mean_legacy
    except Exception as e:
        print(f"  - Failed (Likely OOM): {e}")

    # ---------------------------------------------------------
    # 2. New Way: PalXFELLoader -> calculate_statistics (Chunked)
    # ---------------------------------------------------------
    print("\n[2] New Way (Chunked Processing)")
    gc.collect()  # Legacy에서 쓴 메모리 정리
    mem_before = get_current_memory_mb()
    start_time = time.time()

    try:
        loader = PalXFELLoader(file)
        
        # Identity Preprocessor (그냥 통과)
        # 100장씩 끊어서 로드 -> 합계 누적 -> 메모리 해제 반복
        stats = loader.calculate_statistics(lambda x: x, chunk_size=100)

        # 결과 계산
        if stats['pon_count'] > 0:
            mean_new = stats['pon_sum'] / stats['pon_count']
        
        end_time = time.time()
        mem_after = get_current_memory_mb()

        print(f"  - Execution Time : {end_time - start_time:.4f} sec")
        print(f"  - Memory Usage   : +{mem_after - mem_before:.2f} MB (Total: {mem_after:.2f} MB)")
        
        if 'mean_new' in locals():
            print(f"  - Result Shape   : {mean_new.shape}")

    except Exception as e:
        print(f"  - Failed: {e}")

    print("=" * 60)