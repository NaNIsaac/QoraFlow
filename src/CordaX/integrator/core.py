from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from tables.exceptions import HDF5ExtError

from ..config import ExpConfig, ConfigManager
from ..functional import batched
from .loader import RawDataLoader
from .saver import SaverStrategy
from ..logger import Logger, setup_logger
from ..preprocessor.image_qbpm_preprocessor import ImagesQbpmProcessor


class CoreIntegrator:
    """
    Core pipeline class for integrating, preprocessing, and saving XFEL scan data.
    Implements the ETL pattern (Extract, Transform, Load).
    """

    def __init__(
        self,
        LoaderStrategy: type[RawDataLoader],
        merge_num: int = 1,
        preprocessor: dict[str, ImagesQbpmProcessor] | None = None,
        logger: Logger | None = None,
    ) -> None:
        """Initialize CoreIntegrator instance (excluding data processing)."""
        self.LoaderStrategy: type[RawDataLoader] = LoaderStrategy
        self.merge_num: int = merge_num
        self.preprocessor: dict[str, ImagesQbpmProcessor] = preprocessor or {
            "no_processing": lambda x: x
        }
        self.logger: Logger = logger or setup_logger()
        self.config: ExpConfig = ConfigManager.load_config()

        self.logger.info(f"Loader: {self.LoaderStrategy.__name__}")
        self.logger.info(f"Meta Data:\n{self.config}")
        
        self._result: dict[str, dict[str, npt.NDArray]] | None = None

    def run_integration(self, scan_dir: str | Path) -> dict[str, dict[str, npt.NDArray]]:
        """
        Run the integration process: load data in batches, preprocess, average, and stack results.
        """
        scan_dir = Path(scan_dir)
        self.logger.info(f"Starting scan integration for: {scan_dir}")

        preprocessor_data_dict: dict[str, defaultdict[str, list[Any]]] = {
            name: defaultdict(list) for name in self.preprocessor
        }

        hdf5_files: list[Path] = sorted(
            scan_dir.glob("*.h5"), key=lambda file: int(file.stem[1:])
        )

        batches = list(batched(hdf5_files, self.merge_num))
        desc = f"Processing {scan_dir.name}"

        pbar = tqdm(batches, desc=desc, unit="batch")

        for h5_batch in pbar:
            h5_batch_dirs = [scan_dir / file for file in h5_batch]

            loaders = self._get_loaders(h5_batch_dirs)
            if loaders is None:
                continue

            preprocessed_data = self._preprocess_data(loaders)

            for name, data in preprocessed_data.items():
                for data_key, data_value in data.items():
                    preprocessor_data_dict[name][data_key].append(data_value)

            del loaders
            
        self.logger.info(f"Completed processing: {scan_dir}")

        final_result = {
            name: {key: np.stack(values) for key, values in data.items()}
            for name, data in preprocessor_data_dict.items()
        }

        self._result = final_result
        return final_result

    def _get_loaders(self, hdf5_batch_dirs: list[Path]) -> list[RawDataLoader] | None:
        """
        Get Loader instances for the given batch of HDF5 files.
        """
        try:
            return [self.LoaderStrategy(h5_dir) for h5_dir in hdf5_batch_dirs]
        except (KeyError, FileNotFoundError, ValueError, HDF5ExtError) as e:
            self.logger.warning(f"{type(e).__name__} occurred while loading batch: {e}")
            return None
        except Exception as e:
            self.logger.critical(f"Unexpected {type(e).__name__} while loading batch")
            raise

    def _preprocess_data(
        self,
        loaders: list[RawDataLoader],
    ) -> dict[str, dict[str, Any]]:
        """
        Extract (get_data), Transform (preprocessor), and calculate the average result.
        """
        preprocessed_data: dict[str, dict[str, Any]] = {}

        raw_collections = {
            "pon": [], "pon_qbpm": [],
            "poff": [], "poff_qbpm": [],
        }
        common_delay = None

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

    @property
    def result(self) -> dict[str, dict[str, npt.NDArray]]:
        """Returns the stacked, processed result data."""
        if self._result is None:
            raise AttributeError("Integration has not been run. Call run_integration() first.")
        return self._result

    def save(self, saver: SaverStrategy, run_n: int, scan_n: int):
        """
        Saves processed images using a specified saving strategy.
        Requires run_integration() to be executed first.
        """
        self.logger.info(f"Start to save as {saver.file_type.capitalize()}")

        if self._result is None:
            self.logger.error("Nothing to save: Integration result is missing.")
            raise ValueError("Nothing to save: Call run_integration() before saving.")

        for name, data_dict in self._result.items():
            saver.save(run_n, scan_n, data_dict, comment=name)

            self.logger.info(f"Finished preprocessor: {name}")
            self.logger.info(f"Data dict Keys: {list(data_dict.keys())}")
            self.logger.info(f"Saved file '{saver.file}'")