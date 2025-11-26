from collections import defaultdict
from pathlib import Path
from typing import Any, Optional
from itertools import batched

import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from tables.exceptions import HDF5ExtError

from ..config import ExpConfig, ConfigManager
from .loader import RawDataLoader
from .saver import SaverStrategy
from ..logger import Logger, setup_logger
from ..preprocessor.image_qbpm_preprocessor import ImagesQbpmProcessor


class CoreIntegrator:
    """
    Use ETL Pattern
    """

    def __init__(
        self,
        LoaderStrategy: type[RawDataLoader],
        scan_dir: str | Path,
        merge_num: int = 1,
        preprocessor: Optional[dict[str, ImagesQbpmProcessor]] = None,
        logger: Optional[Logger] = None,
    ) -> None:
        self.LoaderStrategy: type[RawDataLoader] = LoaderStrategy
        scan_dir = Path(scan_dir)
        self.merge_num: int = merge_num
        self.preprocessor: dict[str, ImagesQbpmProcessor] = preprocessor or {
            "no_processing": lambda x: x
        }
        self.logger: Logger = logger or setup_logger()
        self.result: dict[str, defaultdict[str, npt.NDArray]] = self.integrate(scan_dir)
        self.config: ExpConfig = ConfigManager.load_config()

        self.logger.info(f"Loader: {self.LoaderStrategy.__name__}")
        self.logger.info(f"Meta Data:\n{self.config}")

    def integrate(self, scan_dir: Path) -> dict[str, defaultdict[str, npt.NDArray]]:
        """
        Apply preprocessing to each multi-shots and average them.
        """
        self.logger.info(f"Starting scan: {scan_dir}")

        preprocessor_data_dict: dict[str, defaultdict[str, list]] = {
            name: defaultdict(list) for name in self.preprocessor
        }

        hdf5_files: list[Path] = sorted(
            scan_dir.glob("*.h5"), key=lambda file: int(file.stem[1:])
        )
        iterator = batched(hdf5_files, self.merge_num)
        desc = str(Path(*scan_dir.parts[-2:]))
        pbar = tqdm(iterator, total=(len((hdf5_files) + 1) // self.merge_num), desc=desc)
        for h5_batch in pbar:
            h5_batch_dirs = [scan_dir / file for file in h5_batch]
            loaders = self._get_loaders(h5_batch_dirs)
            if loaders is None:
                continue

            preprocessed_data = self._preprocess_data(loaders)
            for name, data in preprocessed_data.items():
                for data_key, data_value in data.items():
                    preprocessor_data_dict[name][data_key].append(data_value)

        self.logger.info(f"Completed processing: {scan_dir}")

        return {
            name: {key: np.stack(values) for key, values in data.items()}
            for name, data in preprocessor_data_dict.items()
        }

    def _get_loaders(self, hdf5_batch_dirs: list[Path]) -> Optional[list[RawDataLoader]]:
        """
        Get Loader for the given HDF5 file.

        Parameters:
        - hdf5_dir (str): Path to the HDF5 file.

        Returns:
        - Optional[RawDataLoader]: Loader instance or None if loading fails.
        """
        try:
            return [self.LoaderStrategy(h5_dir) for h5_dir in hdf5_batch_dirs]
        except (KeyError, FileNotFoundError, ValueError, HDF5ExtError) as e:
            self.logger.warning(f"{type(e)} occurred while loading {hdf5_batch_dirs}")
            return None
        except Exception as e:
            self.logger.critical(f"{type(e)} occurred while loading {hdf5_batch_dirs}")
            raise
        # except Exception as e:
        #     self.logger.exception(f"{type(e)} occurred while loading {hdf5_dir}")
        #     return None

    def _preprocess_data(
        self,
        loaders: list[RawDataLoader],
    ) -> dict[str, dict[str, Any]]:
        """Preprocess data using the preprocessor and average the results."""
        preprocessed_data: dict[str, dict[str, Any]] = {}

        raw_collections = {
            "pon": [], "pon_qbpm": [],
            "poff": [], "poff_qbpm": [],
        }
        common_delay = None

        # 2. 로더들을 순회하며 데이터 수집 (Merge 준비)
        for loader in loaders:
            loader_data = loader.get_data()
            
            if common_delay is None and "delay" in loader_data:
                common_delay = loader_data["delay"]

            if "pon" in loader_data:
                raw_collections["pon"].append(loader_data["pon"])
                if "pon_qbpm" in loader_data:
                    raw_collections["pon_qbpm"].append(loader_data["pon_qbpm"])
            
            if "poff" in loader_data:
                raw_collections["poff"].append(loader_data["poff"])
                if "poff_qbpm" in loader_data:
                    raw_collections["poff_qbpm"].append(loader_data["poff_qbpm"])

        for name, preprocessor in self.preprocessor.items():
            result_data: dict[str, Any] = {}
            
            if raw_collections["pon"]:
                merged_pon = np.concatenate(raw_collections["pon"], axis=0)
                merged_pon_qbpm = np.concatenate(raw_collections["pon_qbpm"], axis=0)
                
                processed_pon_tuple = preprocessor((merged_pon, merged_pon_qbpm))
                
                result_data["pon"] = processed_pon_tuple[0].mean(axis=0)

            if raw_collections["poff"]:
                merged_poff = np.concatenate(raw_collections["poff"], axis=0)
                merged_poff_qbpm = np.concatenate(raw_collections["poff_qbpm"], axis=0)
                
                processed_poff_tuple = preprocessor((merged_poff, merged_poff_qbpm))
                
                result_data["poff"] = processed_poff_tuple[0].mean(axis=0)

            if common_delay is not None:
                result_data["delay"] = common_delay

            preprocessed_data[name] = result_data

        return preprocessed_data

    def save(self, saver: SaverStrategy, run_n: int, scan_n: int):
        """
        Saves processed images using a specified saving strategy.
        """
        self.logger.info(f"Start to save as {saver.file_type.capitalize()}")

        if not self.result:
            self.logger.error("Nothing to save")
            raise ValueError("Nothing to save")

        for name, data_dict in self.result.items():
            saver.save(run_n, scan_n, data_dict)
            self.logger.info(f"Finished preprocessor: {name}")
            self.logger.info(f"Data Dict Keys: {data_dict.keys()}")
            self.logger.info(f"Saved file '{saver.file}'")
