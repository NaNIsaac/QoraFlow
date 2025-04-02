from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import tifffile
from roi_rectangle import RoiRectangle

from src.gui.roi import RoiSelector
from src.filesystem import make_run_scan_dir
from src.config.config import load_config
from src.logger import setup_logger, Logger
from src.analyzer.draw_figure import (
    patch_rectangle,
    draw_com_figure,
    draw_intensity_figure,
    draw_intensity_diff_figure,
    draw_com_diff_figure
)
from src.analyzer.core import DataAnalyzer

if TYPE_CHECKING:
    from pandas import DataFrame
    from matplotlib.figure import Figure
    from src.config.config import ExpConfig


def main() -> None:
    config: ExpConfig = load_config()
    logger: Logger = setup_logger()

    run_nums: list[int] = [161, 162, 163, 164, 165, 166, 167]
    logger.info(f"Data Analysing run={run_nums}")
    for run_num in run_nums:  # pylint: disable=not-an-iterable
        # Define run and scan numbers
        scan_num: int = 1
        now = datetime.now()
        roi_name: str = now.strftime("%Y%m%d_%H%M%S")

        # Define file paths and names
        processed_dir: str = config.path.processed_dir
        file_name: str = f"run={run_num:0>4}_scan={scan_num:0>4}"
        npz_file: Path = make_run_scan_dir(processed_dir, run_num, scan_num, file_name).with_suffix(".npz")

        if not npz_file.exists():
            error_message = f"The file {npz_file} does not exist."
            logger.error(error_message)
            raise FileNotFoundError(error_message)

        logger.info(f"Run DataAnalyzer run={run_num:0>3} scan={scan_num:0>3}")

        # Initialize MeanDataProcessor
        processor: DataAnalyzer = DataAnalyzer(npz_file)

        # Extract images
        poff_images: npt.NDArray = processor.poff_images
        pon_images: npt.NDArray = processor.pon_images

        # Select ROI using GUI
        roi: Optional[tuple[int, int, int, int]] = RoiSelector().select_roi(
            np.log1p(poff_images[0])
        )
        if roi is None:
            logger.error(f"No ROI Rectangle Set for run={run_num}, scan={scan_num}")
            raise ValueError(f"No ROI Rectangle Set for run={run_num}, scan={scan_num}")

        logger.info(f"ROI rectangle: {roi}")
        roi_rect: RoiRectangle = RoiRectangle.from_tuple(roi)

        # Analyze data within the selected ROI
        data_df: DataFrame = processor.analyze_by_roi(roi_rect)

        # Define save directory
        output_dir: Path = make_run_scan_dir(config.path.output_dir, run_num, scan_num, roi_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Slice images to ROI
        roi_poff_images: npt.NDArray = roi_rect.slice(poff_images)
        roi_pon_images: npt.NDArray = roi_rect.slice(pon_images)

        # Save images as TIFF files
        tifffile.imwrite(output_dir / 'poff.tif', poff_images.astype(np.float32))
        logger.info(f"Saved TIF '{output_dir / 'poff.tif'}'")

        tifffile.imwrite(output_dir / 'pon.tif', pon_images.astype(np.float32))
        logger.info(f"Saved TIF '{output_dir / 'pon.tif'}'")

        tifffile.imwrite(output_dir / "roi_poff.tif", roi_poff_images.astype(np.float32))
        logger.info(f"Saved TIF '{output_dir / 'roi_poff.tif'}'")

        tifffile.imwrite(output_dir / "roi_pon.tif", roi_pon_images.astype(np.float32))
        logger.info(f"Saved TIF '{output_dir / 'roi_pon.tif'}'")

        # Save data as CSV
        data_file: Path = output_dir /"data.csv"
        data_df.to_csv(data_file)
        logger.info(f"Saved CSV '{data_file}'")

        # Create figures
        image_fig: Figure = patch_rectangle(
            np.log1p(processor.poff_images.sum(axis=0)),
            *roi_rect.to_tuple()
        )
        intensity_fig: Figure = draw_intensity_figure(data_df)
        intensity_diff_fig: Figure = draw_intensity_diff_figure(data_df)
        com_fig: Figure = draw_com_figure(data_df)
        com_diff_fig: Figure = draw_com_diff_figure(data_df)

        # Save figures as PNG files
        image_fig.savefig(output_dir / "log_image.png")
        logger.info(f"Saved PNG '{output_dir / 'log_image.png'}'")

        intensity_fig.savefig(output_dir /"delay-intensity.png")
        logger.info(f"Saved PNG '{output_dir / 'delay-intensity.png'}'")

        intensity_diff_fig.savefig(output_dir / "delay-intensity_diff.png")
        logger.info(f"Saved PNG '{output_dir / 'delay-intensity_diff.png'}'")

        com_fig.savefig(output_dir / "delay-com.png")
        logger.info(f"Saved PNG '{output_dir /'delay-com.png'}'")

        com_diff_fig.savefig(output_dir / "delay-com_diff.png")
        logger.info(f"Saved PNG '{output_dir / 'delay-com_diff.png'}'")

        logger.info(f"Run DataAnalyzer run={run_num:0>3} scan={scan_num:0>3} is Done.")
        plt.close("all")


if __name__ == "__main__":
    main()
