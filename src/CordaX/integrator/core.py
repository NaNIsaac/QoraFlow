from collections import defaultdict
from contextlib import ExitStack
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
    Delegates heavy processing to the Loader via `calculate_statistics`.
    """

    def __init__(
        self,
        LoaderStrategy: type[RawDataLoader],
        merge_num: int = 1,
        chunk_size: int = 100,
        preprocessor: dict[str, ImagesQbpmProcessor] | None = None,
        logger: Logger | None = None,
    ) -> None:
        self.LoaderStrategy = LoaderStrategy
        self.merge_num = merge_num
        self.chunk_size = chunk_size
        self.preprocessor = preprocessor or {"no_processing": lambda x: x}
        self.logger = logger or setup_logger()
        self.config: ExpConfig = ConfigManager.load_config()

        self.logger.info(f"Loader: {self.LoaderStrategy.__name__}")
        self.logger.info(f"Merge Num: {self.merge_num}")
        self.logger.info(f"Chunk Size: {self.chunk_size}")
        
        self._result = None

    def run(self, scan_dir: str | Path) -> dict[str, dict[str, npt.NDArray]]:
        scan_dir = Path(scan_dir)
        self.logger.info(f"Starting scan integration for: {scan_dir}")

        final_results_list = defaultdict(lambda: defaultdict(list))
        
        hdf5_files = sorted(scan_dir.glob("*.h5"), key=lambda file: int(file.stem[1:]))
        batches = list(batched(hdf5_files, self.merge_num))

        pbar = tqdm(batches, desc=f"Processing {scan_dir.name}", unit="group")

        for h5_batch in pbar:
            h5_batch_dirs = [scan_dir / file for file in h5_batch]

            with ExitStack() as stack:
                try:
                    loaders = [stack.enter_context(self.LoaderStrategy(f)) for f in h5_batch_dirs]
                except Exception as e:
                    self.logger.warning(f"Skipping batch: {e}")
                    continue
                
                # [Optimization] Delegate processing to Loader
                batch_data = self._process_batch_via_loader_stats(loaders)

                # Collect Results
                for proc_name, data in batch_data.items():
                    for delay, content in data.items():
                        if "pon" in content: final_results_list[proc_name]["pon"].append(content["pon"])
                        if "poff" in content: final_results_list[proc_name]["poff"].append(content["poff"])
                        final_results_list[proc_name]["delay"].append(delay)

            gc.collect()
            
        self.logger.info(f"Completed processing: {scan_dir}")
        self._result = self._stack_results(final_results_list)
        return self._result

    def _process_batch_via_loader_stats(
        self,
        loaders: list[RawDataLoader],
    ) -> dict[str, dict[float, dict[str, Any]]]:
        """
        Aggregates statistics (Sum/Count) calculated by each Loader.
        """
        # Accumulators: { proc_name: { 'pon_sum': ..., 'pon_cnt': ... } }
        accumulators = defaultdict(lambda: {
            "pon_sum": None, "pon_count": 0,
            "poff_sum": None, "poff_count": 0
        })

        representative_delay = None

        for i, loader in enumerate(loaders):
            # 1. Get Delay (First file only)
            if i == 0:
                # We can access delay cheaply via loader properties
                # (Assuming get_data or init already populated it)
                try:
                    representative_delay = loader.delay
                except:
                    pass

            # 2. Run Preprocessors via Loader
            for name, preprocessor in self.preprocessor.items():
                acc = accumulators[name]
                
                # [Delegate] Ask loader to compute stats for this preprocessor
                # This keeps RAM usage low inside the loader's method
                file_stats = loader.calculate_statistics(preprocessor, self.chunk_size)

                # 3. Aggregate Global Sum/Count
                if file_stats["pon_sum"] is not None:
                    if acc["pon_sum"] is None: acc["pon_sum"] = file_stats["pon_sum"]
                    else: acc["pon_sum"] += file_stats["pon_sum"]
                    acc["pon_count"] += file_stats["pon_count"]

                if file_stats["poff_sum"] is not None:
                    if acc["poff_sum"] is None: acc["poff_sum"] = file_stats["poff_sum"]
                    else: acc["poff_sum"] += file_stats["poff_sum"]
                    acc["poff_count"] += file_stats["poff_count"]

        # 4. Compute Final Mean for the Batch
        if representative_delay is None or np.isnan(representative_delay):
            return {}

        final_delay = round(float(representative_delay), 6)
        batch_output = defaultdict(dict)

        for name, acc in accumulators.items():
            res = {}
            if acc["pon_count"] > 0:
                res["pon"] = acc["pon_sum"] / acc["pon_count"]
            if acc["poff_count"] > 0:
                res["poff"] = acc["poff_sum"] / acc["poff_count"]
            
            if res:
                batch_output[name][final_delay] = res

        return batch_output

    def _stack_results(self, results_list: dict) -> dict:
        stacked = {}
        for name, data in results_list.items():
            stacked[name] = {}
            delays = np.array(data["delay"])
            sort_idx = np.argsort(delays)
            
            stacked[name]["delay"] = delays[sort_idx]
            if data["pon"]:
                stacked[name]["pon"] = np.stack(data["pon"])[sort_idx]
            if data["poff"]:
                stacked[name]["poff"] = np.stack(data["poff"])[sort_idx]
        return stacked

    @property
    def result(self): return self._result

    def save(self, saver: SaverStrategy, run_n, scan_n):
        if not self._result: raise ValueError("No result")
        for name, data in self._result.items():
            saver.save(run_n, scan_n, data, comment=name)