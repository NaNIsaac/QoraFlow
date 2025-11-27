from pathlib import Path

from roi_rectangle import RoiRectangle

from CordaX.config import ExpConfig, ConfigManager
from CordaX.filesystem import get_run_scan_dir, get_scan_nums
from CordaX.functional import compose
from CordaX.gui.select_roi import auto_roi
from CordaX.integrator.core import CoreIntegrator
from CordaX.integrator.loader import PalXFELLoader
from CordaX.integrator.saver import SaverStrategy, get_saver_strategy
from CordaX.logger import Logger, setup_logger
from CordaX.preprocessor.image_qbpm_preprocessor import (
    ImagesQbpmProcessor,
    make_qbpm_roi_normalizer,
    make_thresholder,
    subtract_dark_background,
)


logger: Logger = setup_logger()
ConfigManager.initialize("config.yaml")
config: ExpConfig = ConfigManager.load_config()


def setup_preprocessors(scan_dir: Path) -> dict[str, ImagesQbpmProcessor]:
    """Return preprocessors"""

    # roi_rect: RoiRectangle = select_roi(scan_dir, config, None)
    roi_rect: RoiRectangle = auto_roi(scan_dir, config, None)

    filter_and_normalize_by_qbpm = make_qbpm_roi_normalizer(roi_rect)
    threshold4 = make_thresholder(4)
    # compose make a function that exicuted from right to left
    standard = compose(
        threshold4,
        filter_and_normalize_by_qbpm,
        subtract_dark_background
    )

    return {
        "standard": standard,
    }


def integrate_scan(run_n: int, scan_n: int) -> None:
    """Integrate Single Scan"""

    scan_dir: Path = get_run_scan_dir(config.path.load_dir, run_n, scan_n)

    preprocessors: dict[str, ImagesQbpmProcessor] = setup_preprocessors(scan_dir)

    # Logging
    for preprocessor_name in preprocessors:
        logger.info(f"preprocessor: {preprocessor_name}")

    integrator: CoreIntegrator = CoreIntegrator(
        PalXFELLoader,
        merge_num=config.param.merge_num, 
        preprocessor=preprocessors, 
        logger=logger
    )
    integrator.run_integration(scan_dir)
    # Set and use SaverStrategy
    for saver_type in ["npz", "mat"]:
        saver: SaverStrategy = get_saver_strategy(saver_type)
        integrator.save(saver, run_n, scan_n)

    logger.info(f"Processing run={run_n}, scan={scan_n} is complete")


def main() -> None:
    """The entry point of the program."""

    logger.info(f"Runs to integrate: {config.runs}")

    for run_n in config.runs:
        logger.info(f"Run: {run_n}")
        scan_nums: list[int] = get_scan_nums(config.path.load_dir, run_n)
        for scan_n in scan_nums:
            try:
                integrate_scan(run_n, scan_n)
            except Exception:
                logger.exception(f"Failed to process run={run_n}, scan={scan_n}")
                raise

    logger.info("All integration is complete")


if __name__ == "__main__":
    main()
