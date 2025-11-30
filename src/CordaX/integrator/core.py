from collections import defaultdict
from pathlib import Path
from typing import Any
import gc

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from ..config import ExpConfig, ConfigManager
from ..functional import batched
from .loader import RawDataLoader
from .saver import SaverStrategy
from ..logger import Logger, setup_logger
from ..preprocessor.image_qbpm_preprocessor import ImagesQbpmProcessor


class CoreIntegrator:
    """
    Core pipeline class for integrating XFEL scan data.
    
    [Single-Process / Sequential Version]
    - Optimized for I/O Bound tasks (Disk speed limited).
    - Chunking logic removed (Delegated to Loader or handled entirely in memory).
    - Processes files one by one to maximize sequential read performance.
    """

    def __init__(
        self,
        LoaderStrategy: type[RawDataLoader],
        merge_num: int = 1,
        processors: dict[str, ImagesQbpmProcessor] | None = None,
        logger: Logger | None = None,
    ) -> None:
        self.LoaderStrategy = LoaderStrategy
        self.merge_num = merge_num
        
        # Default to identity function if no preprocessor is provided
        self.preprocessor = processors or {
            "no_processing": lambda x: x
        }
        self.logger = logger or setup_logger()
        self.config: ExpConfig = ConfigManager.load_config()

        self.logger.info(f"Loader Strategy: {self.LoaderStrategy.__name__}")
        self.logger.info(f"Merge Bundle Size: {self.merge_num}")
        self.logger.info("Mode: Sequential Processing (Full-Load / No-Chunking)")
        
        self._result: dict[str, dict[str, npt.NDArray]] | None = None

    def run(self, scan_dir: str | Path) -> dict[str, dict[str, npt.NDArray]]:
        scan_dir = Path(scan_dir)
        self.logger.info(f"Starting sequential integration for: {scan_dir}")

        # Final storage: { 'proc_name': { 'pon': [arr...], 'poff': [arr...], 'delay': [float...] } }
        final_results_list = defaultdict(lambda: defaultdict(list))
        
        # Sort files to ensure order (S001.h5 -> S002.h5 ...)
        hdf5_files = sorted(
            scan_dir.glob("*.h5"), key=lambda file: int(file.stem[1:])
        )

        batches = list(batched(hdf5_files, self.merge_num))
        desc = f"Processing {scan_dir.name} (bundles={self.merge_num})"

        # Progress bar
        pbar = tqdm(batches, desc=desc, unit="batch")

        for h5_batch in pbar:
            h5_batch_paths = [scan_dir / file for file in h5_batch]

            try:
                # [Core Logic] Process batch sequentially
                batch_stats = self._process_batch_sequential(h5_batch_paths)
                
                # If batch failed or was skipped (no delay), move to next
                if not batch_stats:
                    continue

                # Collect results
                for proc_name, data_map in batch_stats.items():
                    for delay_key, result_content in data_map.items():
                        if "pon" in result_content:
                            final_results_list[proc_name]["pon"].append(result_content["pon"])
                        if "poff" in result_content:
                            final_results_list[proc_name]["poff"].append(result_content["poff"])
                        
                        final_results_list[proc_name]["delay"].append(delay_key)

            except Exception as e:
                self.logger.error(f"Critical error in batch starting with {h5_batch[0]}: {e}")
                continue

            # Explicit GC is still useful for large array operations
            # gc.collect()
            
        self.logger.info(f"Completed processing: {scan_dir}")

        # Convert lists to 3D Numpy Arrays
        self._result = self._stack_results(final_results_list)
        return self._result

    def _process_batch_sequential(
        self,
        file_paths: list[Path],
    ) -> dict[str, dict[float, dict[str, Any]]]:
        """
        Process a group of files sequentially without chunking.
        """
        
        # 1. Initialize Accumulators
        accumulators = defaultdict(lambda: {
            "pon_sum": None, "pon_count": 0,
            "poff_sum": None, "poff_count": 0
        })

        representative_delay: float | None = None

        # 2. Iterate through files in the batch
        for i, file_path in enumerate(file_paths):
            
            with self.LoaderStrategy(file_path) as loader:
                
                for name, processor_func in self.preprocessor.items():
                    
                    # [Removed Chunking] Just call statistics calculation directly
                    file_stats = loader.calculate_statistics(processor_func)
                    
                    # [Delay Logic] Only trust the first file of the batch
                    if i == 0 and representative_delay is None:
                        if "delay" in file_stats and file_stats["delay"] is not None:
                            representative_delay = float(file_stats["delay"])
                        elif hasattr(loader, "delay") and loader.delay is not None:
                            representative_delay = float(loader.delay)

                    # Accumulate Sums and Counts
                    acc = accumulators[name]
                    
                    if file_stats.get("pon_sum") is not None:
                        if acc["pon_sum"] is None: acc["pon_sum"] = file_stats["pon_sum"]
                        else: acc["pon_sum"] += file_stats["pon_sum"]
                        acc["pon_count"] += file_stats["pon_count"]

                    if file_stats.get("poff_sum") is not None:
                        if acc["poff_sum"] is None: acc["poff_sum"] = file_stats["poff_sum"]
                        else: acc["poff_sum"] += file_stats["poff_sum"]
                        acc["poff_count"] += file_stats["poff_count"]

        # 3. Validation: If no delay found in the first file, discard batch
        if representative_delay is None:
            self.logger.warning(f"Skipping batch: No delay found in first file {file_paths[0].name}")
            return {}

        # 4. Finalize: Calculate Mean
        final_delay_key = round(representative_delay, 6)
        batch_output = defaultdict(dict)

        for name, acc in accumulators.items():
            res = {}
            if acc["pon_count"] > 0:
                res["pon"] = acc["pon_sum"] / acc["pon_count"]
            
            if acc["poff_count"] > 0:
                res["poff"] = acc["poff_sum"] / acc["poff_count"]

            if res:
                batch_output[name][final_delay_key] = res

        return batch_output

    def _stack_results(self, results_list: dict) -> dict:
        """Convert collected lists into sorted numpy arrays based on delay."""
        stacked = {}
        for name, data in results_list.items():
            if not data["delay"]:
                continue

            stacked[name] = {}
            delays = np.array(data["delay"])
            sort_idx = np.argsort(delays)
            
            stacked[name]["delay"] = delays[sort_idx]
            
            if "pon" in data and data["pon"]: 
                stacked[name]["pon"] = np.stack(data["pon"], axis=0)[sort_idx]
            if "poff" in data and data["poff"]: 
                stacked[name]["poff"] = np.stack(data["poff"], axis=0)[sort_idx]
        return stacked

    @property
    def result(self):
        if self._result is None:
            raise AttributeError("Integration has not been run. Call 'run()' first.")
        return self._result

    def save(self, saver: SaverStrategy, run_n: int, scan_n: int):
        self.logger.info(f"Start to save as {saver.file_type.capitalize()}")
        if not self._result:
            raise ValueError("Nothing to save.")
        
        for name, data in self._result.items():
            saver.save(run_n, scan_n, data, comment=name)
            self.logger.info(f"Saved file '{saver.file}' ({name})")