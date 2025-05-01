import sys
from pathlib import Path

import click
import numpy as np
from roi_rectangle import RoiRectangle

from src.gui.roi_core import RoiSelector
from src.analyzer.converter import load_npz
from src.config import load_config
from src.filesystem import get_run_scan_dir


def load_image(run_n: int, scan_n: int) -> np.ndarray:
    """Loads image data for a specific run and scan."""

    config = load_config()

    processed_dir = config.path.processed_dir

    npz_file: Path = get_run_scan_dir(
        processed_dir, run_n, scan_n, sub_path=f"run={run_n:04}_scan={scan_n:04}.npz"
    )
    if not npz_file.exists():
        raise FileNotFoundError(f"File not found: {npz_file}")
    data = load_npz(npz_file)
    if "poff" in data:
        image = data["poff"].mean(0)
    elif "pon" in data:
        image = data["pon"].mean(0)
    else:
        raise KeyError(f"Keys 'poff' or 'pon' not in {npz_file}")
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
        # Recalculate bounds if defaults were used, as log changes scale
        if vmin is None:
            eff_vmin = np.min(img)
        if vmax is None:
            eff_vmax = np.max(img)
        if eff_vmin > eff_vmax:
            eff_vmin = eff_vmax  # Re-check after log

    return np.clip(img, eff_vmin, eff_vmax)


def interactive_roi_selection(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Handles the GUI interaction for ROI selection ONLY."""
    click.echo("Launching ROI selector GUI...", err=True)  # Info to stderr
    try:
        selector = RoiSelector()
        # Assuming select_roi blocks and returns tuple or raises on cancel/close
        coords = selector.select_roi(image)
        return coords
    except Exception as e:
        click.echo(f"ROI selection cancelled or GUI error: {e}", err=True)
        return None


# --- Reusable Option Decorators ---
# Decorator for common data identification arguments
def data_source_options(func):
    func = click.argument("run_n", type=int)(func)
    func = click.option(
        "--scan", "scan_n", type=int, default=1, show_default=True, help="Scan number."
    )(func)
    return func


# Decorator for common image processing options
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


# --- CLI Structure ---
@click.group()
def cli():
    """Unified CLI for data processing and visualization."""


@cli.command()
@data_source_options  # Add RUN_N arg and --scan option
@processing_options  # Add --log, --vmin, --vmax options
def view(run_n: int, scan_n: int, log: bool, vmin: float | None, vmax: float | None):
    """
    Load, process, and display an image in the GUI viewer.

    This command shows the image but does NOT perform interactive ROI selection.
    Use 'get-roi' command for interactive selection.
    """
    try:
        image = load_image(run_n, scan_n)
        processed_image = process_image(image, log, vmin, vmax)
        if np.ptp(processed_image) == 0:
            click.echo("Warning: Processed image is constant.", err=True)

        # Use the same selector tool, but just for viewing (modify if needed)
        click.echo("Launching image viewer GUI...", err=True)
        selector = RoiSelector()
        # Call a hypothetical 'view' method, or adapt 'select_roi' if it just displays
        # For now, assume calling select_roi and ignoring its output works for viewing.
        selector.select_roi(processed_image)
        click.echo("Viewer closed.", err=True)

    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        click.ClickException,
        ImportError,
    ) as e:
        click.echo(f"Error: {e}", err=True)
    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
        raise


@cli.command(name="get-roi")  # Explicit command name
@data_source_options  # Add RUN_N arg and --scan option
@processing_options  # Add --log, --vmin, --vmax options
def get_roi(run_n: int, scan_n: int, log: bool, vmin: float | None, vmax: float | None):
    """
    Load, process, and interactively select an ROI, printing the result.

    Displays the processed image in a GUI window for ROI selection.
    Outputs the selected ROI coordinates (x0, y0, x1, y1) to standard output.
    """
    try:
        image = load_image(run_n, scan_n)
        processed_image = process_image(image, log, vmin, vmax)
        if np.ptp(processed_image) == 0:
            click.echo("Warning: Processed image is constant.", err=True)

        # Perform interactive selection
        roi_coords = interactive_roi_selection(processed_image)

        if roi_coords:
            roi_rect = RoiRectangle.from_tuple(roi_coords)
            click.echo(str(roi_rect))  # Output result to stdout
        else:
            # Inform user selection wasn't completed (error already printed)
            # Exit with error status if ROI is considered mandatory here

            sys.exit(1)  # Exit indicating failure

    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        click.ClickException,
        ImportError,
    ) as e:
        click.echo(f"Error: {e}", err=True)

        sys.exit(1)
    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)

        sys.exit(1)


# Potential future command for non-GUI processing
# @cli.command()
# @data_source_options
# @processing_options
# @click.option('--output', type=click.Path(), help='Save processed data to file.')
# def process(run_n, scan_n, log, vmin, vmax, output):
#    """Load and process data, save to output file (no GUI)."""
#    # ... implementation ...
#    pass


if __name__ == "__main__":
    cli()
