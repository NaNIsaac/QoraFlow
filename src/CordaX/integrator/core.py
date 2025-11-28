'''D:\Dev\CordaX\src\CordaX\integrator\core.py'''
from collections import defaultdict
from contextlib import ExitStack  # <--- 추가됨: 여러 context manager를 관리하는 도구
from pathlib import Path
from typing import Any
import gc

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
    Refactored to use Incremental Averaging to prevent memory overflow.
    """

    def __init__(
        self,
        LoaderStrategy: type[RawDataLoader],
        merge_num: int = 1,
        preprocessor: dict[str, ImagesQbpmProcessor] | None = None,
        logger: Logger | None = None,
    ) -> None:
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

    def run(self, scan_dir: str | Path) -> dict[str, dict[str, npt.NDArray]]:
        scan_dir = Path(scan_dir)
        self.logger.info(f"Starting scan integration for: {scan_dir}")

        running_sums: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
        running_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        final_metadata: dict[str, Any] = {}

        hdf5_files: list[Path] = sorted(
            scan_dir.glob("*.h5"), key=lambda file: int(file.stem[1:])
        )

        batches = list(batched(hdf5_files, self.merge_num))
        desc = f"Processing {scan_dir.name}"

        pbar = tqdm(batches, desc=desc, unit="batch")

        for h5_batch in pbar:
            h5_batch_dirs = [scan_dir / file for file in h5_batch]

            with ExitStack() as stack:
                try:
                    loaders = [
                        stack.enter_context(self.LoaderStrategy(h5_dir)) 
                        for h5_dir in h5_batch_dirs
                    ]
                except (KeyError, FileNotFoundError, ValueError, HDF5ExtError) as e:
                    self.logger.warning(f"{type(e).__name__} occurred while loading batch: {e}")
                    continue
                except Exception as e:
                    self.logger.critical(f"Unexpected {type(e).__name__} while loading batch")
                    raise

                batch_results = self._preprocess_batch_incremental(loaders)

                for proc_name, data_map in batch_results.items():
                    for key, (batch_sum, batch_count) in data_map.items():
                        if key == "delay":
                            final_metadata["delay"] = batch_sum
                            continue

                        if key not in running_sums[proc_name]:
                            running_sums[proc_name][key] = batch_sum
                            running_counts[proc_name][key] = batch_count
                        else:
                            running_sums[proc_name][key] += batch_sum
                            running_counts[proc_name][key] += batch_count
            del batch_results

            gc.collect()
            
        self.logger.info(f"Completed processing: {scan_dir}")

        # Final Average Calculation
        final_result = {}
        for proc_name, key_map in running_sums.items():
            final_result[proc_name] = {}
            for key, total_sum in key_map.items():
                count = running_counts[proc_name][key]
                if count > 0:
                    final_result[proc_name][key] = total_sum / count
                else:
                    final_result[proc_name][key] = np.zeros_like(total_sum)
            
            if "delay" in final_metadata:
                final_result[proc_name]["delay"] = final_metadata["delay"]

        self._result = final_result
        return final_result

    def _preprocess_batch_incremental(
        self,
        loaders: list[RawDataLoader],
    ) -> dict[str, dict[str, tuple[Any, int]]]:
        """
        Extract and Transform data.
        """
        preprocessed_data: dict[str, dict[str, tuple[Any, int]]] = {}

        raw_collections = {
            "pon": [], "pon_qbpm": [],
            "poff": [], "poff_qbpm": [],
        }
        common_delay = None

        for loader in loaders:
            # Loader가 이미 open 상태임 (ExitStack 덕분)
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

        # Apply preprocessors
        for name, preprocessor in self.preprocessor.items():
            result_data: dict[str, tuple[Any, int]] = {}
            
            if raw_collections["pon"]:
                merged_pon = np.concatenate(raw_collections["pon"], axis=0)
                merged_pon_qbpm = np.concatenate(raw_collections["pon_qbpm"], axis=0)
                
                processed_pon_tuple = preprocessor((merged_pon, merged_pon_qbpm))
                
                batch_sum = processed_pon_tuple[0].sum(axis=0)
                batch_count = processed_pon_tuple[0].shape[0]
                result_data["pon"] = (batch_sum, batch_count)

            if raw_collections["poff"]:
                merged_poff = np.concatenate(raw_collections["poff"], axis=0)
                merged_poff_qbpm = np.concatenate(raw_collections["poff_qbpm"], axis=0)

                processed_poff_tuple = preprocessor((merged_poff, merged_poff_qbpm))
                
                batch_sum = processed_poff_tuple[0].sum(axis=0)
                batch_count = processed_poff_tuple[0].shape[0]
                result_data["poff"] = (batch_sum, batch_count)

            if common_delay is not None:
                result_data["delay"] = (common_delay, 1)

            preprocessed_data[name] = result_data

        return preprocessed_data

    @property
    def result(self) -> dict[str, dict[str, npt.NDArray]]:
        if self._result is None:
            raise AttributeError("Integration has not been run. Call run_integration() first.")
        return self._result

    def save(self, saver: SaverStrategy, run_n: int, scan_n: int):
        self.logger.info(f"Start to save as {saver.file_type.capitalize()}")

        if self._result is None:
            self.logger.error("Nothing to save: Integration result is missing.")
            raise ValueError("Nothing to save: Call run_integration() before saving.")

        for name, data_dict in self._result.items():
            saver.save(run_n, scan_n, data_dict, comment=name)

            self.logger.info(f"Finished preprocessor: {name}")
            self.logger.info(f"Data dict Keys: {list(data_dict.keys())}")
            self.logger.info(f"Saved file '{saver.file}'")