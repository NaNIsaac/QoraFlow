import gc
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from ..config import ConfigManager, ExpConfig
from ..functional import batched
from ..logger import Logger, setup_logger
from ..preprocessor.image_qbpm_preprocessor import ImagesQbpmProcessor
from .loader import RawDataLoader
from .saver import SaverStrategy


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
        self.logger.info(
            f"Merge Num: {self.merge_num} (Merging {self.merge_num} files into 1 dataset)"
        )

        self._result: dict[str, dict[str, npt.NDArray]] | None = None

    def run(self, scan_dir: str | Path) -> dict[str, dict[str, npt.NDArray]]:
        scan_dir = Path(scan_dir)
        self.logger.info(f"Starting scan integration for: {scan_dir}")

        # Structure: { 'preprocessor_name': { 'pon': [(t1, avg_img1), ...], 'poff': [(t1, avg_img1), ...] } }
        final_results_list: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))

        hdf5_files: list[Path] = sorted(scan_dir.glob("*.h5"), key=lambda file: int(file.stem[1:]))

        # Bundle files in merge_num unit and process
        batches = list(batched(hdf5_files, self.merge_num))
        desc = f"Processing {scan_dir.name} (bundles={self.merge_num})"

        pbar = tqdm(batches, desc=desc, unit="group")
        loaded_batches = 0

        for h5_batch in pbar:
            # Load all of the data in one batch
            with ExitStack() as stack:
                try:
                    loaders = [
                        stack.enter_context(self.LoaderStrategy(h5_file)) for h5_file in h5_batch
                    ]
                except Exception as e:
                    self.logger.warning(f"Skipping batch due to load error: {e}")
                    continue
                loaded_batches += 1

                # Return variables: { 'proc_name': { 'pon': 2D_Avg_Img, 'delay': Scalar } }
                batch_averaged_data = self._process_batch_group(loaders)

                # Save result
                for proc_name, data_map in batch_averaged_data.items():
                    # There might be multiple results per delay (usually 1)
                    for delay_key, result_content in data_map.items():
                        # Keep each image paired with its own delay; pump states can be missing per batch
                        for state in ("pon", "poff"):
                            if state in result_content:
                                final_results_list[proc_name][state].append(
                                    (delay_key, result_content[state])
                                )

            # Release the raw frames (several GB) before the next batch
            del batch_averaged_data
            gc.collect()

        if batches and not loaded_batches:
            raise RuntimeError(f"Every batch in {scan_dir} failed to load")

        self.logger.info(f"Completed processing: {scan_dir}")

        final_result_stack = self._stack_results(final_results_list)
        self._result = final_result_stack

        return final_result_stack

    def _process_batch_group(
        self,
        loaders: list[RawDataLoader],
    ) -> dict[str, dict[float, dict[str, Any]]]:
        """
        Incrementally processes chunks from loaders and accumulates sums.
        Memory usage remains low regardless of total file size.

        Frames are grouped per (preprocessor, delay). Delays that agree to
        6 decimal places are treated as the same nominal delay, so a merge
        batch spanning several delays yields one averaged image per delay.
        """
        accumulators = defaultdict(
            lambda: {
                "pon_sum": None,
                "pon_count": 0,
                "poff_sum": None,
                "poff_count": 0,
                "delays": [],
            }
        )

        for loader in loaders:
            for chunk_data in loader.get_chunked_data(chunk_size=self.chunk_size):
                if "delay" not in chunk_data:
                    continue
                current_delay = float(chunk_data["delay"])

                for name, preprocessor in self.preprocessor.items():
                    acc = accumulators[(name, round(current_delay, 6))]
                    acc["delays"].append(current_delay)

                    if "pon" in chunk_data:
                        pon_img = chunk_data["pon"]
                        # No QBPM signal: weight every frame by 1
                        pon_qbpm = chunk_data.get("pon_qbpm", np.ones(len(pon_img)))

                        proc_pon, _ = preprocessor((pon_img, pon_qbpm))

                        if proc_pon.size > 0:
                            # Accumulate only the sum to save memory
                            batch_sum = proc_pon.sum(axis=0)
                            batch_cnt = proc_pon.shape[0]

                            if acc["pon_sum"] is None:
                                acc["pon_sum"] = batch_sum.astype(np.float64)
                            else:
                                acc["pon_sum"] += batch_sum
                            acc["pon_count"] += batch_cnt

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

        batch_output = defaultdict(dict)

        for (name, _), acc in accumulators.items():
            if not acc["delays"]:
                continue

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
        """
        Convert (delay, image) pairs to delay-sorted 3D stacks per pump state.

        Uses a shared 'delay' key when pon and poff cover the same delays,
        otherwise 'pon_delay' and 'poff_delay'.
        """
        stacked = {}
        for proc_name, data_map in results_list.items():
            stacked[proc_name] = {}
            state_delays = {}

            for state in ("pon", "poff"):
                pairs = sorted(data_map.get(state, []), key=lambda pair: pair[0])
                if not pairs:
                    continue
                delays, images = zip(*pairs, strict=True)
                state_delays[state] = np.array(delays)
                stacked[proc_name][state] = np.stack(images, axis=0)

            match list(state_delays.values()):
                case [delays]:
                    stacked[proc_name]["delay"] = delays
                case [pon_delays, poff_delays] if np.array_equal(pon_delays, poff_delays):
                    stacked[proc_name]["delay"] = pon_delays
                case _:
                    self.logger.warning(f"{proc_name}: pon and poff delays differ")
                    for state, delays in state_delays.items():
                        stacked[proc_name][f"{state}_delay"] = delays

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
