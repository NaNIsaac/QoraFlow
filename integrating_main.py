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
    # subtract_dark_background,
    create_pohang,
    create_threshold,
    ImagesQbpmProcessor
)
from gui.roi_core import get_hdf5_images, RoiSelector
from src.filesystem import get_run_scan_dir
from src.config.config import load_config, ExpConfig
from src.functional import compose


logger: Logger = setup_logger()
config: ExpConfig = load_config()


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
