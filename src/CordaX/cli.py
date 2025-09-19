import sys
from pathlib import Path

import click
import numpy as np
from roi_rectangle import RoiRectangle

from .gui.roi_core import RoiSelector
from .analyzer.converter import load_npz
from .config import load_config
from .filesystem import get_run_scan_dir


class CliError(click.ClickException):
    """Generic CLI error for graceful termination."""

    def __init__(self, message):
        super().__init__(message)


def load_image(run_n: int, scan_n: int) -> np.ndarray:
    """Loads image data for a specific run and scan."""
    config = load_config()
    processed_dir = config.path.processed_dir

    npz_file: Path = get_run_scan_dir(
        processed_dir, run_n, scan_n, sub_path=f"run={run_n:04}_scan={scan_n:04}.npz"
    )

    if not npz_file.exists():
        raise click.FileError(filename=str(npz_file), hint="File not found.")

    try:
        data = load_npz(npz_file)
    except Exception as e:
        raise CliError(f"Failed to load npz file: {e}") from e

    if "poff" in data:
        image = data["poff"].mean(0)
    elif "pon" in data:
        image = data["pon"].mean(0)
    else:
        raise CliError(f"Missing required keys 'poff' or 'pon' in {npz_file}")

    return image


def process_image(
    image: np.ndarray, log: bool, vmin: float | None, vmax: float | None
) -> np.ndarray:
    """Applies processing (log, clip) to the image."""
    img = np.nan_to_num(image.copy())
    eff_vmin = vmin if vmin is not None else np.min(img)
    eff_vmax = vmax if vmax is not None else np.max(img)

    if eff_vmin > eff_vmax:
        click.echo(f"Warning: vmin {eff_vmin} > vmax {eff_vmax}", err=True)
        eff_vmin = eff_vmax

    if log:
        img = np.log1p(img)
        if vmin is None:
            eff_vmin = np.min(img)
        if vmax is None:
            eff_vmax = np.max(img)
        if eff_vmin > eff_vmax:
            eff_vmin = eff_vmax

    return np.clip(img, eff_vmin, eff_vmax)


def interactive_roi_selection(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Handles the GUI interaction for ROI selection ONLY."""
    click.echo("Launching ROI selector GUI...", err=True)
    try:
        selector = RoiSelector()
        coords = selector.select_roi(image)
        return coords
    except Exception as e:
        raise CliError(f"ROI selection failed: {e}") from e


# --- Decorators ---
def data_source_options(func):
    func = click.argument("run_n", type=int)(func)
    func = click.option("--scan", "scan_n", type=int, default=1, show_default=True)(
        func
    )
    return func


def processing_options(func):
    func = click.option("--log", is_flag=True, help="Apply log1p transformation.")(func)
    func = click.option(
        "--vmin",
        type=float,
        default=None,
        help="Min value for clipping (default: auto).",
    )(func)
    func = click.option(
        "--vmax",
        type=float,
        default=None,
        help="Max value for clipping (default: auto).",
    )(func)
    return func


@click.group()
def cli():
    """Unified CLI for data processing and visualization."""


@cli.command(name="get-roi")
@data_source_options
@processing_options
def get_roi(run_n: int, scan_n: int, log: bool, vmin: float | None, vmax: float | None):
    """
    Load, process, and interactively select an ROI.
    Outputs selected coordinates (x0, y0, x1, y1).
    """
    try:
        image = load_image(run_n, scan_n)
        processed_image = process_image(image, log, vmin, vmax)

        if np.ptp(processed_image) == 0:
            click.echo("Warning: Processed image is constant.", err=True)

        roi_coords = interactive_roi_selection(processed_image)

        if roi_coords:
            roi_rect = RoiRectangle.from_tuple(roi_coords)
            click.echo(str(roi_rect))
        else:
            raise CliError("ROI selection cancelled or failed.")

    except click.ClickException as e:
        e.show()  # Proper click-style error output
        sys.exit(e.exit_code)
