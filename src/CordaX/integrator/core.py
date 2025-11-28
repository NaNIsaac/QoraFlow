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
    
    Logic:
    1. Group files by 'merge_num' (Experimental Repeats).
    2. Load ALL images from these files (e.g., 3 files * 300 frames = 900 frames).
    3. Group internal data by Delay (Time).
    4. Preprocess (RANSAC, etc.) & Average -> Single Representative Image.
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
        self.logger.info(f"Merge Num: {self.merge_num} (Merging {self.merge_num} files into 1 dataset)")
        
        self._result: dict[str, dict[str, npt.NDArray]] | None = None

    def run(self, scan_dir: str | Path) -> dict[str, dict[str, npt.NDArray]]:
        scan_dir = Path(scan_dir)
        self.logger.info(f"Starting scan integration for: {scan_dir}")

        # Structure: { 'preprocessor_name': { 'pon': [avg_img1, avg_img2...], 'delay': [t1, t2...] } }
        final_results_list: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
        
        hdf5_files: list[Path] = sorted(
            scan_dir.glob("*.h5"), key=lambda file: int(file.stem[1:])
        )

        # Bundle files in merge_num unit and process
        batches = list(batched(hdf5_files, self.merge_num))
        desc = f"Processing {scan_dir.name} (bundles={self.merge_num})"

        pbar = tqdm(batches, desc=desc, unit="group")

        for h5_batch in pbar:
            h5_batch_dirs = [scan_dir / file for file in h5_batch]

            # Load all of the data in one batch
            with ExitStack() as stack:
                try:
                    loaders = [
                        stack.enter_context(self.LoaderStrategy(h5_dir)) 
                        for h5_dir in h5_batch_dirs
                    ]
                except Exception as e:
                    self.logger.warning(f"Skipping batch due to load error: {e}")
                    continue
                
                # Return variables: { 'proc_name': { 'pon': 2D_Avg_Img, 'delay': Scalar } }
                # [OOM Critical Point 1] Large raw data is processed here
                batch_averaged_data = self._process_batch_group(loaders)

                # Save result
                for proc_name, data_map in batch_averaged_data.items():
                    # There might be multiple results per delay (usually 1)
                    for delay_key, result_content in data_map.items():
                        if "pon" in result_content:
                            final_results_list[proc_name]["pon"].append(result_content["pon"])
                        if "poff" in result_content:
                            final_results_list[proc_name]["poff"].append(result_content["poff"])
                        
                        # Store delay information
                        final_results_list[proc_name]["delay"].append(delay_key)

            # 5. Memory release (Very Important: delete several GB of Raw data)
            del batch_averaged_data
            gc.collect()
            
        self.logger.info(f"Completed processing: {scan_dir}")

        # [Final] Convert list -> Numpy Stack
        final_result_stack = self._stack_results(final_results_list)
        self._result = final_result_stack
        
        return final_result_stack

    def _process_batch_group(
        self,
        loaders: list[RawDataLoader],
    ) -> dict[str, dict[float, dict[str, Any]]]:
        """
        [Modified]
        Merges ALL data from the provided loaders into a single dataset,
        ignoring whether the internal delays are different.
        
        The representative delay for the result will be the MEAN of all delays in the batch.
        """
        
        # 1. Containers for ALL data in this batch
        batch_pon: list[npt.NDArray] = []
        batch_pon_qbpm: list[npt.NDArray] = []
        batch_poff: list[npt.NDArray] = []
        batch_poff_qbpm: list[npt.NDArray] = []
        
        batch_delays: list[float] = []

        # 2. Collect data from all loaders
        for loader in loaders:
            d = loader.get_data()
            
            # Collect delay for averaging later
            if "delay" in d:
                batch_delays.append(float(d["delay"]))
            
            if "pon" in d:
                batch_pon.append(d["pon"])
                # Handle missing QBPM by filling with ones
                batch_pon_qbpm.append(d.get("pon_qbpm", np.ones(len(d["pon"]))))
            
            if "poff" in d:
                batch_poff.append(d["poff"])
                batch_poff_qbpm.append(d.get("poff_qbpm", np.ones(len(d["poff"]))))

        # If no delay info found, default to 0.0 or handle error
        if not batch_delays:
            return {}
        
        # Calculate representative delay (Mean of the batch)
        rep_delay = float(np.mean(batch_delays))
        rep_delay = round(rep_delay, 6) # Precision control

        # 3. Concatenate (Merge)
        merged_pon = None
        merged_pon_qbpm = None
        if batch_pon:
            # [OOM Risk] Concatenate all frames in the batch
            merged_pon = np.concatenate(batch_pon, axis=0)
            merged_pon_qbpm = np.concatenate(batch_pon_qbpm, axis=0)

            # [Optimization] Immediate memory release
            batch_pon = None 
            batch_pon_qbpm = None

        merged_poff = None
        merged_poff_qbpm = None
        if batch_poff:
            merged_poff = np.concatenate(batch_poff, axis=0)
            merged_poff_qbpm = np.concatenate(batch_poff_qbpm, axis=0)

            # [Optimization] Immediate memory release
            batch_poff = None
            batch_poff_qbpm = None

        # 4. Preprocess & Average
        batch_output = defaultdict(dict)

        for name, preprocessor in self.preprocessor.items():
            res = {}

            # Process PON
            if merged_pon is not None:
                # Preprocess the entire merged stack
                proc_pon, _ = preprocessor((merged_pon, merged_pon_qbpm))

                if proc_pon.size > 0:
                    res["pon"] = proc_pon.mean(axis=0)

            # Process POFF
            if merged_poff is not None:
                proc_poff, _ = preprocessor((merged_poff, merged_poff_qbpm))

                if proc_poff.size > 0:
                    res["poff"] = proc_poff.mean(axis=0)

            # Save result using the representative delay
            if res:
                batch_output[name][rep_delay] = res

        return batch_output

    def _stack_results(self, results_list: dict) -> dict:
        """Convert 2D images collected in list to 3D stack in chronological order."""
        stacked = {}
        for proc_name, data_map in results_list.items():
            stacked[proc_name] = {}
            
            # Group data to sort by Delay (Delay, Index)
            delays = np.array(data_map["delay"])
            # Create sort index
            sort_idx = np.argsort(delays)
            
            stacked[proc_name]["delay"] = delays[sort_idx]
            
            if "pon" in data_map and data_map["pon"]:
                # Convert list -> array and index in sorted order
                pon_stack = np.stack(data_map["pon"], axis=0)
                stacked[proc_name]["pon"] = pon_stack[sort_idx]
                
            if "poff" in data_map and data_map["poff"]:
                poff_stack = np.stack(data_map["poff"], axis=0)
                stacked[proc_name]["poff"] = poff_stack[sort_idx]
                
        return stacked

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
            self.logger.info(f"Saved file '{saver.file}'")