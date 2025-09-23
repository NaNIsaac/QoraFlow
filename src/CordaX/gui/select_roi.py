from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from roi_rectangle import RoiRectangle

from ..config import ExpConfig
from ..filesystem import get_run_scan_dir
from .roi_core import RoiSelector
from ..integrator.loader import get_hdf5_images


def get_metadata_roi(scan_dir: str | Path, config: ExpConfig) -> RoiRectangle:
    """Get Roi from metadata"""
    scan_dir = Path(scan_dir)
    files = list(scan_dir.glob("*.h5"))
    file: Path = files[0]
    metadata = pd.read_hdf(file, key="metadata")
    roi_coord = np.array(
        metadata[
            f"detector_{config.param.hutch.value}_{config.param.detector.value}_parameters.ROI"
        ].iloc[0][0]
    )

    roi = np.array(
        [
            roi_coord[config.param.x1],
            roi_coord[config.param.y1],
            roi_coord[config.param.x2],
            roi_coord[config.param.y2],
        ],
        dtype=np.int_,
    )

    return RoiRectangle.from_tuple(roi)


def select_roi(
    scan_dir: str | Path, config: ExpConfig, index_mode: Optional[int] = None
) -> RoiRectangle:
    """Select ROI from GUI"""
    scan_dir = Path(scan_dir)
    files = list(scan_dir.glob("*.h5"))
    files.sort(key=lambda name: int(name.stem[1:]))
    if index_mode is None:
        index = len(files) // 2
    else:
        index = index_mode

    file: Path = scan_dir / files[index]
    image = get_hdf5_images(file, config).sum(axis=0)
    return RoiRectangle.from_tuple(RoiSelector().select_roi(np.log1p(image)))


def auto_roi(scan_dir: str | Path, config: ExpConfig, index_mode: Optional[int] = None):
    """Get Roi based on the maximum value of the image"""
    scan_dir = Path(scan_dir)
    files = list(scan_dir.glob("*.h5"))
    files.sort(key=lambda file: int(file.stem[1:]))
    if index_mode is None:
        index = len(files) // 2
    else:
        index = index_mode

    file: Path = scan_dir / files[index]
    image = get_hdf5_images(file, config).sum(axis=0)

    # Normalize image
    img = np.nan_to_num(image)
    img = img - np.min(img)
    if np.max(img) != 0:
        img = img / np.max(img)

    # Threshold to remove background noise (e.g., keep top 10%)
    threshold = np.percentile(img, 90)
    mask = img >= threshold
    masked_img = img * mask

    # Compute centroid (intensity-weighted)
    total = masked_img.sum()

    indices = np.indices(masked_img.shape)
    y_c = int(np.sum(indices[0] * masked_img) / total)
    x_c = int(np.sum(indices[1] * masked_img) / total)

    d = 20  # Half-side of ROI box
    return RoiRectangle(x_c - d, y_c - d, x_c + d, y_c + d)


if __name__ == "__main__":
    from CordaX.config import ConfigManager

    config: ExpConfig = ConfigManager.load_config()
    scan_dir = get_run_scan_dir(config.path.load_dir, 114, 1)
    print(config)

    roi1 = get_metadata_roi(scan_dir, config)
    print(roi1)
    roi2 = select_roi(scan_dir, config, 0)
    print(roi2)
    roi3 = auto_roi(scan_dir, config, 0)
    print(roi3)
