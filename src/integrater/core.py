from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from src.config.config import ExpConfig, load_config
from src.integrater.loader import RawDataLoader
from src.integrater.saver import SaverStrategy
from src.logger import Logger, setup_logger
from src.preprocessor.image_qbpm_preprocessor import ImagesQbpmProcessor


class CoreIntegrater:
    """
    Use ETL Pattern
    """
    def __init__(
        self,
        LoaderStrategy: type[RawDataLoader],
        scan_dir: str | Path,
        preprocessor: Optional[dict[str, ImagesQbpmProcessor]] = None,
        logger: Optional[Logger] = None
    ) -> None:
        self.LoaderStrategy: type[RawDataLoader] = LoaderStrategy
        scan_dir = Path(scan_dir)
        self.preprocessor: dict[str, ImagesQbpmProcessor] = preprocessor or {"no_processing": lambda x: x}

        self.logger: Logger = logger if logger is not None else setup_logger()
        self.result: dict[str, defaultdict[str, npt.NDArray]] = self.scan(scan_dir)
        self.config: ExpConfig = load_config()

        self.logger.info(f"Meta Data:\n{self.config}")

    def scan(self, scan_dir: Path) -> None:
        """
        Processes a single scan directory.

        Parameters:
        - scan_dir (str): Directory path of the scan to process.

        Returns:
        - dict[str, npt.NDArray]: Dictionary containing stacked images from the scan.
        """
        self.logger.info(f"Starting scan: {scan_dir}")

        preprocessor_data_dict: dict[str, defaultdict[str, list]] = {
            pipeline_name: defaultdict(list)
            for pipeline_name in self.preprocessor
        }

        hdf5_files = scan_dir.glob("*.h5")
        hdf5_files: list[Path] = sorted(hdf5_files, key=lambda file: int(file.stem[1:]))

        pbar = tqdm(hdf5_files, total=len(hdf5_files), desc=scan_dir.name)
        for hdf5_file in pbar:
            loader_strategy = self.get_loader(scan_dir / hdf5_file)
            if loader_strategy is None:
                continue
            preprocessed_data = self.preprocess_data(loader_strategy)

            for preprocessor_name, data in preprocessed_data.items():
                for data_key, data_value in data.items():
                    preprocessor_data_dict[preprocessor_name][data_key].append(data_value)

        self.logger.info(f"Completed processing: {scan_dir}")

        result: dict[str, defaultdict[str, npt.NDArray]] = {}
        for preprocessor_name, data in preprocessor_data_dict.items():
            result[preprocessor_name] = {
                data_name: np.stack(data_list) for data_name, data_list in data.items()
            }
        return result

    def get_loader(self, hdf5_dir: str) -> Optional[RawDataLoader]:
        """
        Get Loader for the given HDF5 file.

        Parameters:
        - hdf5_dir (str): Path to the HDF5 file.

        Returns:
        - Optional[RawDataLoader]: Loader instance or None if loading fails.
        """
        try:
            return self.LoaderStrategy(hdf5_dir)
        except (KeyError, FileNotFoundError, ValueError) as e:
            self.logger.exception(f"{type(e)} happened in {hdf5_dir}")
            return None
        except Exception as e:
            self.logger.exception(f"Failed to load: {type(e)}: {str(e)}")
            return None
        # except Exception as e:
        #     self.logger.critical(f"{type(e)} happened in {hdf5_dir}")
        #     raise

    def preprocess_data(
        self,
        loader_strategy: RawDataLoader,
    ) -> dict[str, dict[str, Any]]:

        preprocessed_data: dict[str, dict[str, Any]] = {}
        for preprocessor_name, preprocessor in self.preprocessor.items():
            data: dict[str, Any] = {}
            loader_dict = loader_strategy.get_data()
            if "pon" in loader_dict:
                applied_pon_images: npt.NDArray = preprocessor((loader_dict['pon'], loader_dict['pon_qbpm']))[0]
                data['pon'] = applied_pon_images.mean(axis=0)
            if 'poff' in loader_dict:
                applied_poff_images: npt.NDArray = preprocessor((loader_dict['poff'], loader_dict['poff_qbpm']))[0]
                data['poff'] = applied_poff_images.mean(axis=0)
            data["delay"] = loader_dict['delay']
            preprocessed_data[preprocessor_name] = data

        return preprocessed_data

    def save(self, saver: SaverStrategy, run_n: int, scan_n: int):
        """
        Saves processed images using a specified saving strategy.

        Parameters:
        - saver (SaverStrategy): Saving strategy to use.
        - comment (str, optional): Comment to append to the file name.
        """
        self.logger.info(f"Start to save as {saver.file_type.capitalize()}")

        if not self.result:
            self.logger.error("Nothing to save")
            raise ValueError("Nothing to save")

        for pipeline_name, data_dict in self.result.items():

            saver.save(run_n, scan_n, data_dict)
            self.logger.info(f"Finished preprocessor: {pipeline_name}")
            self.logger.info(f"Data Dict Keys: {data_dict.keys()}")
            self.logger.info(f"Saved file '{saver.file}'")
