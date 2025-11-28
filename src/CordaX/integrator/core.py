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
        chunk_size: int = 100,
        preprocessor: dict[str, ImagesQbpmProcessor] | None = None,
        logger: Logger | None = None,
    ) -> None:
        self.LoaderStrategy: type[RawDataLoader] = LoaderStrategy
        self.merge_num: int = merge_num
        self.chunk_size = chunk_size
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
        [Corrected]
        Incrementally processes chunks from loaders and accumulates sums.
        Memory usage remains low regardless of total file size.
        """
        # 1. 누적기 초기화
        accumulators = defaultdict(lambda: {
            "pon_sum": None, "pon_count": 0,
            "poff_sum": None, "poff_count": 0,
            "delays": []
        })

        # 2. 모든 로더에 대해 반복
        for loader in loaders:
            # [핵심] 청크 단위(예: 100장)로 조금씩 가져옴
            for chunk_data in loader.get_chunked_data(chunk_size=self.chunk_size):
                
                if "delay" not in chunk_data:
                    continue
                current_delay = float(chunk_data["delay"])
                
                # 전처리기별 수행
                for name, preprocessor in self.preprocessor.items():
                    acc = accumulators[name]
                    acc["delays"].append(current_delay)

                    # --- Pump On ---
                    if "pon" in chunk_data:
                        pon_img = chunk_data["pon"]
                        # QBPM 없으면 1로 채움
                        pon_qbpm = chunk_data.get("pon_qbpm", np.ones(len(pon_img)))
                        
                        # 전처리 (100장 단위)
                        proc_pon, _ = preprocessor((pon_img, pon_qbpm))
                        
                        if proc_pon.size > 0:
                            # 합계(Sum)만 누적 (메모리 절약)
                            batch_sum = proc_pon.sum(axis=0)
                            batch_cnt = proc_pon.shape[0]

                            if acc["pon_sum"] is None:
                                acc["pon_sum"] = batch_sum.astype(np.float64)
                            else:
                                acc["pon_sum"] += batch_sum
                            acc["pon_count"] += batch_cnt

                    # --- Pump Off ---
                    if "poff" in chunk_data:
                        poff_img = chunk_data["poff"]
                        poff_qbpm = chunk_data.get("poff_qbpm", np.ones(len(poff_img)))
                        
                        proc_poff, _ = preprocessor((poff_img, poff_qbpm))
                        
                        if proc_poff.size > 0:
                            batch_sum = proc_poff.sum(axis=0)
                            batch_cnt = proc_poff.shape[0]

                            if acc["poff_sum"] is None:
                                acc["poff_sum"] = batch_sum.astype(np.float64)
                            else:
                                acc["poff_sum"] += batch_sum
                            acc["poff_count"] += batch_cnt

        # 3. 최종 평균 계산 (Finalize)
        # 누적된 합계(Sum)를 개수(Count)로 나눠서 평균 이미지 생성
        batch_output = defaultdict(dict)

        for name, acc in accumulators.items():
            if not acc["delays"]:
                continue
            
            # 대표 딜레이 (평균값 사용)
            rep_delay = float(np.mean(acc["delays"]))
            rep_delay = round(rep_delay, 6)

            res = {}
            if acc["pon_count"] > 0:
                res["pon"] = acc["pon_sum"] / acc["pon_count"]
            
            if acc["poff_count"] > 0:
                res["poff"] = acc["poff_sum"] / acc["poff_count"]

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