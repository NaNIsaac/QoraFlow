from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tifffile
from roi_rectangle import RoiRectangle

from src.analyzer.core import DataAnalyzer
from src.analyzer.draw_figure import (
    draw_com_diff_figure,
    draw_com_figure,
    draw_intensity_diff_figure,
    draw_intensity_figure,
    patch_rectangle,
)
from src.config import load_config
from src.filesystem import get_run_scan_dir, make_run_scan_dir
from src.gui.roi_core import RoiSelector
from src.logger import Logger, setup_logger

if TYPE_CHECKING:
    from pandas import DataFrame

    from src.config import ExpConfig


def main() -> None:
    """Entry point"""
    config: ExpConfig = load_config()
    logger: Logger = setup_logger()

    run_nums: list[int] = [162]
    logger.info(f"Data Analysing run={run_nums}")
    for run_num in run_nums:
        scan_num: int = 1
        now = datetime.now()
        roi_name: str = now.strftime("%Y%m%d_%H%M%S")

        processed_dir: str = config.path.processed_dir
        file_name: str = f"run={run_num:0>4}_scan={scan_num:0>4}"
        npz_file: Path = get_run_scan_dir(
            processed_dir, run_num, scan_num, sub_path=file_name
        ).with_suffix(".npz")
        logger.info(f"NPZ file: {npz_file}")

        if not npz_file.exists():
            err_msg: str = f"The file {npz_file} does not exist."
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        logger.info(f"Run DataAnalyzer run={run_num:0>3} scan={scan_num:0>3}")

        # Initialize MeanDataProcessor
        processor: DataAnalyzer = DataAnalyzer(npz_file)
        poff_images: npt.NDArray = processor.poff_images
        pon_images: npt.NDArray = processor.pon_images

        # Select ROI using GUI
        if roi := RoiSelector().select_roi(np.log1p(poff_images[0])):
            err_msg: str = f"No ROI Rectangle Set for run={run_num}, scan={scan_num}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        logger.info(f"ROI rectangle: {roi}")
        roi_rect: RoiRectangle = RoiRectangle.from_tuple(roi)

        # Analyze data within the selected ROI
        data_df: DataFrame = processor.analyze_by_roi(roi_rect)

        # Define save directory
        output_dir: Path = make_run_scan_dir(
            config.path.output_dir, run_num, scan_num, sub_path=roi_name
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Slice images to ROI
        roi_poff_images: npt.NDArray = roi_rect.slice(poff_images)
        roi_pon_images: npt.NDArray = roi_rect.slice(pon_images)

        # Save Images and Data
        data_file: Path = output_dir / "data.csv"
        data_df.to_csv(data_file)
        logger.info(f"Saved CSV '{data_file}'")

        images_to_save = {
            "poff.tif": poff_images,
            "pon.tif": pon_images,
            "roi_poff.tif": roi_poff_images,
            "roi_pon.tif": roi_pon_images,
        }
        for filename, image_data in images_to_save.items():
            file_path = output_dir / filename
            tifffile.imwrite(file_path, image_data.astype(np.float32))
            logger.info(f"Saved TIF '{file_path}'")

        # Save Figures
        figures_to_save = {
            "log_image.png": patch_rectangle(
                np.log1p(processor.poff_images.sum(axis=0)), *roi_rect.to_tuple()
            ),
            "delay-intensity.png": draw_intensity_figure(data_df),
            "delay-intensity_diff.png": draw_intensity_diff_figure(data_df),
            "delay-com.png": draw_com_figure(data_df),
            "delay-com_diff.png": draw_com_diff_figure(data_df),
        }
        for filename, fig in figures_to_save.items():
            file_path = output_dir / filename
            fig.savefig(file_path)
            logger.info(f"Saved PNG '{file_path}'")

        logger.info(f"Run DataAnalyzer run={run_num:0>3} scan={scan_num:0>3} is Done.")
        plt.close("all")


if __name__ == "__main__":
    main()
