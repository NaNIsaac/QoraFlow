from pathlib import Path
import numpy as np
from roi_rectangle import RoiRectangle
import click

from src.gui.roi import RoiSelector
from src.config.config import load_config
from src.analyzer.converter import load_npz
from src.filesystem import get_run_scan_dir


def load_image(run_n: int) -> np.ndarray:
    """Load image data from npz file."""
    config = load_config()
    scan_n: int = 1
    npz_file: Path = get_run_scan_dir(config.path.processed_dir, run_n, scan_n, f'run={run_n:04}_scan={scan_n:04}.npz')

    # Check if the file exists
    if not npz_file.exists():
        raise click.ClickException(f"File '{npz_file}' not found.")

    data = load_npz(npz_file)

    if 'poff' in data:
        image = data['poff'].mean(0)
    else:
        image = data['pon'].mean(0)
    return image


def process_image(image: np.ndarray, is_log: bool, vmin: float, vmax: float) -> np.ndarray:
    """Process image with specified log scaling and clipping."""
    adjusted_image = np.clip(image, vmin, vmax)

    if is_log:
        adjusted_image = np.log1p(adjusted_image)

    return adjusted_image


def select_roi_from_image(image: np.ndarray) -> tuple[int, int, int, int]:
    """Select ROI from the processed image."""
    return RoiSelector().select_roi(image)


@click.command()
@click.argument('run_n', type=int)
@click.option('--roi', is_flag=True, help="Flag to trigger ROI selection.")
@click.option('--log', is_flag=True, help="Flag to apply log1p to the image.")
@click.option('--vmin', type=float, default=0.0, help="Minimum value for image clipping.")
@click.option('--vmax', type=float, default=np.inf, help="Maximum value for image clipping.")
def gui_cli(run_n: int, roi: bool, log: bool, vmin: float, vmax: float) -> None:
    """
    Command-line interface to process and optionally display ROI of an image.

    Args:
        run_n (int): The run number.
        roi (bool): Flag to trigger ROI selection.
        log (bool): Flag to apply log1p to the image.
        vmin (float): Minimum value for image clipping.
        vmax (float): Maximum value for image clipping.
    """
    image = load_image(run_n)
    processed_image = process_image(image, log, vmin, vmax)

    if roi:
        roi_coords = select_roi_from_image(processed_image)
        roi_rect = RoiRectangle.from_tuple(roi_coords)
        click.echo(str(roi_rect))
    else:
        select_roi_from_image(processed_image)


if __name__ == '__main__':
    gui_cli()
