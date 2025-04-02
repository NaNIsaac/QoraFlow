import os
from typing import Optional
from pathlib import Path
from typing import Generator

import pandas as pd
import numpy as np
from roi_rectangle import RoiRectangle

from src.logger import setup_logger, Logger
from src.integrater.core import CoreIntegrater
from src.integrater.loader import PalXFELLoader
from src.integrater.saver import SaverStrategy, get_saver_strategy
from src.preprocessor.image_qbpm_preprocessor import (
    subtract_dark_background,
    create_pohang,
    create_threshold,
    ImagesQbpmProcessor
)
from src.gui.roi import get_hdf5_images, RoiSelector
from src.filesystem import get_run_scan_dir
from src.config.config import load_config, ExpConfig
from src.functional import compose


logger: Logger = setup_logger()
config: ExpConfig = load_config()

# TODO: Move to other directory
def get_scan_nums(run_num: int) -> list[int]:
    """Get Scan numbers from real directory"""
    run_dir: Path = get_run_scan_dir(config.path.load_dir, run_num)
    scan_folders: Generator[Path, None, None] = run_dir.iterdir()
    return [int(str(scan_dir.stem).split("=")[1]) for scan_dir in scan_folders]

# TODO: Move to other directory
def get_roi(scan_dir: str) -> RoiRectangle:
    """Get Roi for QBPM Normalization"""
    files = os.listdir(scan_dir)
    file: str = os.path.join(scan_dir, files[0])
    metadata = pd.read_hdf(file, key='metadata')
    roi_coord = np.array(
        metadata[
            f'detector_{config.param.hutch.value}_{config.param.detector.value}_parameters.ROI'
        ].iloc[0][0]
    )

    roi = np.array([
        roi_coord[config.param.x1],
        roi_coord[config.param.y1],
        roi_coord[config.param.x2],
        roi_coord[config.param.y2]
    ], dtype=np.int_)

    return RoiRectangle.from_tuple(roi)

# TODO: Move to other directory
def select_roi(scan_dir: str, index_mode: Optional[int] = None) -> RoiRectangle:
    """Get Roi for QBPM Normalization"""
    files = os.listdir(scan_dir)
    files.sort(key=lambda name: int(name[1:-3]))
    if index_mode is None:
        index = len(files) // 2
    else:
        index = index_mode

    file: str = os.path.join(scan_dir, files[index])
    image = get_hdf5_images(file, config).sum(axis=0)
    return RoiRectangle.from_tuple(RoiSelector().select_roi(np.log1p(image)))


# TODO: Move to other directory
def auto_roi(scan_dir: str, index_mode: Optional[int] = None):
    files = os.listdir(scan_dir)
    files.sort(key=lambda name: int(name[1:-3]))
    if index_mode is None:
        index = len(files) // 2
    else:
        index = index_mode

    file: str = os.path.join(scan_dir, files[index])
    image = get_hdf5_images(file, config).sum(axis=0)
    y, x, *_ = np.unravel_index(np.argmax(image, axis=None), image.shape)

    d = 20

    return RoiRectangle(x - d, y - d, x + d, y + d)


def setup_preprocessors(scan_dir: str) -> dict[str, ImagesQbpmProcessor]:
    """Return preprocessors"""

    # roi_rect = select_roi(scan_dir, None)
    # FIXME: Make better auto roi
    # roi_rect = auto_roi(scan_dir, None)

    # if roi_rect is None:
    #     raise ValueError(f"No ROI Rectangle Set for {scan_dir}")
    # logger.info(f"ROI rectangle: {roi_rect.to_tuple()}")

    roi_rect = None
    pohang = create_pohang(roi_rect)
    threshold4 = create_threshold(4)
    # compose make a function that exicuted from right to left
    standard = compose(
        threshold4,
        pohang,
        # subtract_dark_background
    )

    return {
        "standard": standard,
    }


def integrate_scan(run_n: int, scan_n: int) -> None:
    """Integrate Single Scan"""

    load_dir = config.path.load_dir
    scan_dir = get_run_scan_dir(load_dir, run_n, scan_n)

    preprocessors: dict[str, ImagesQbpmProcessor] = setup_preprocessors(scan_dir)

    # Logging
    for preprocessor_name in preprocessors:
        logger.info(f"preprocessor: {preprocessor_name}")

    processor: CoreIntegrater = CoreIntegrater(PalXFELLoader, scan_dir, preprocessors, logger)

    # Set SaverStrategy
    npz_saver: SaverStrategy = get_saver_strategy("npz")
    processor.save(npz_saver, run_n, scan_n)

    mat_saver: SaverStrategy = get_saver_strategy("mat")
    processor.save(mat_saver, run_n, scan_n)

    logger.info(f"Processing run={run_n}, scan={scan_n} is complete")


def main() -> None:
    """The entry point of the program."""

    run_nums: list[int] = config.runs
    logger.info(f"Runs to process: {run_nums}")

    for run_num in run_nums:  # pylint: disable=not-an-iterable
        logger.info(f"Run: {run_num}")
        scan_nums: list[int] = get_scan_nums(run_num)
        for scan_num in scan_nums:
            try:
                integrate_scan(run_num, scan_num)
            except Exception:
                logger.exception(f"Failed to process run={run_num}, scan={scan_num}")
                raise

    logger.info("All processing is complete")


if __name__ == "__main__":
    main()
