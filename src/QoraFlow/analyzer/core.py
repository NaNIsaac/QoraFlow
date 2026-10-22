from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from roi_rectangle import RoiRectangle
from scipy.ndimage import rotate
from scipy.optimize import curve_fit

from ..mathematics import gaussian, mul_delta_q

Background = Literal["edge", "plane"] | None


def edge_background(
    images: npt.NDArray, method: Literal["edge", "plane"], width: int = 3
) -> npt.NDArray:
    """Background of each image in a stack (n, h, w), estimated from its outer ``width`` pixels.

    "edge" is the border median, a flat level. "plane" is the least-squares plane a + b·x + c·y
    through the border, which also removes a gradient. Both read the peak's own tail as
    background when the tail reaches the border: the median then oversubtracts a little, while
    the plane takes the tail for a gradient and pulls the centre of mass back, which is worse
    for a peak that sits off the ROI centre. A gradient biases the median the most. Keep the
    ROI half-width several peak widths and the peak near its centre.
    """
    border = np.ones(images.shape[1:], bool)
    border[width:-width, width:-width] = False
    if method == "edge":
        return np.median(images[:, border], axis=1)[:, None, None]
    if method == "plane":
        y, x = np.indices(images.shape[1:])
        design = np.column_stack([np.ones(border.sum()), x[border], y[border]])
        a, b, c = np.linalg.lstsq(design, images[:, border].T, rcond=None)[0][:, :, None, None]
        return a + b * x + c * y
    raise ValueError(f"unknown background method {method!r}, expected 'edge' or 'plane'")


class DataAnalyzer:
    """
    Analyzes data from a given file and performs various operations on the images.

    Args:
        file (str): The path to the file containing the data.
        angle (int, optional): The angle to rotate the images. Defaults to 0.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not contain the required keys.
    """

    def __init__(self, file: str | Path, angle: int = 0) -> None:
        file = Path(file)
        if not file.exists():
            raise FileNotFoundError(f"The file {file} does not exist.")

        data: Mapping[str, npt.NDArray] = np.load(file)
        if "delay" not in data:
            raise ValueError(f"The file {file} does not contain the required keys: 'delay'")

        self.delay: npt.NDArray = data["delay"]

        if "pon" not in data:
            self.poff_images: npt.NDArray = data["poff"]
            self.pon_images: npt.NDArray = np.full_like(data["poff"], 1)
        elif "poff" not in data:
            self.pon_images: npt.NDArray = data["pon"]
            self.poff_images: npt.NDArray = np.full_like(data["pon"], 1)
        else:
            self.poff_images: npt.NDArray = data["poff"]
            self.pon_images: npt.NDArray = data["pon"]
        if angle:
            self.poff_images = rotate(self.poff_images, angle, axes=(1, 2), reshape=False)
            self.pon_images = rotate(self.pon_images, angle, axes=(1, 2), reshape=False)

        self.poff_images = np.maximum(0, self.poff_images)
        self.pon_images = np.maximum(0, self.pon_images)

    def get_summed_image(self) -> tuple[npt.NDArray, npt.NDArray]:
        """
        return:
            pump off image, pump on image
        """
        return self.poff_images.sum(axis=0), self.pon_images.sum(axis=0)

    def pon_subtract_by_poff(self):
        """Subtrack pump on images by pump off images"""
        return np.maximum(self.pon_images - self.poff_images, 0)

    def _roi_center_of_masses(
        self,
        roi_rect: RoiRectangle,
        images: npt.NDArray,
        background: Background = None,
        edge_width: int = 3,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        roi_images = roi_rect.slice(images)
        if background is not None:
            roi_images = roi_images - edge_background(roi_images, background, edge_width)
        height, width = roi_rect.height, roi_rect.width

        y_coords, x_coords = np.mgrid[:height, :width]

        total_mass = np.sum(roi_images, axis=(1, 2))
        x_centroids = np.sum(x_coords * roi_images, axis=(1, 2)) / total_mass
        y_centroids = np.sum(y_coords * roi_images, axis=(1, 2)) / total_mass

        return x_centroids, y_centroids

    def _roi_gaussian(
        self, roi_rect: RoiRectangle, images: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:

        roi_images = roi_rect.slice(images)
        height, width = roi_rect.height, roi_rect.width

        intensities = []
        com_xs = []
        com_ys = []

        for image in roi_images:
            max_y, max_x = np.unravel_index(np.argmax(image), image.shape)  # pylint: disable=unbalanced-tuple-unpacking

            x: npt.NDArray = np.arange(0, width)
            y: npt.NDArray = np.arange(0, height)

            x_data = image.sum(axis=0)
            y_data = image.sum(axis=1)

            initial_guess_x = [
                x_data[max_x],
                max_x,
                (np.max(x_data) - np.min(x_data)) / 4,
            ]
            try:
                params_x = curve_fit(gaussian, x, x_data, p0=initial_guess_x)[0]
            except RuntimeError as e:
                print(e, ": x")
                params_x = [np.nan, np.nan, np.nan]

            initial_guess_y = [
                y_data[max_y],
                max_y,
                (np.max(y_data) - np.min(y_data)) / 4,
            ]
            try:
                params_y = curve_fit(gaussian, y, y_data, p0=initial_guess_y)[0]
            except RuntimeError as e:
                print(e, ": y")
                params_y = [np.nan, np.nan, np.nan]

            gaussian_a_x, gaussain_com_x, _ = params_x
            gaussian_a_y, gaussain_com_y, _ = params_y

            gaussian_a = np.sqrt(gaussian_a_x * gaussian_a_y)
            intensities.append(gaussian_a)
            com_xs.append(gaussain_com_x)
            com_ys.append(gaussain_com_y)

        return np.stack(intensities), np.stack(com_xs), np.stack(com_ys)

    def _roi_intensities(self, roi_rect: RoiRectangle, images: npt.NDArray):
        roi_images = roi_rect.slice(images)

        return roi_images.mean(axis=(1, 2))

    def analyze_by_roi(
        self, roi_rect: RoiRectangle, background: Background = None, edge_width: int = 3
    ) -> pd.DataFrame:
        """ROI centre of mass and mean intensity per delay, relative to the first delay.

        Args:
            roi_rect: The region of interest.
            background: What to subtract from each ROI image before the centre of mass, estimated
                from its outer ``edge_width`` pixels: None (nothing, the default), "edge" (their
                median, a flat background) or "plane" (a least-squares plane a + b·x + c·y).
                The ROI intensities are not changed. See ``edge_background`` for when each fails.
            edge_width: Width in pixels of the ROI border used to estimate the background.
        """
        poff_com_x, poff_com_y = self._roi_center_of_masses(
            roi_rect, self.poff_images, background, edge_width
        )
        poff_intensity = self._roi_intensities(roi_rect, self.poff_images)
        pon_com_x, pon_com_y = self._roi_center_of_masses(
            roi_rect, self.pon_images, background, edge_width
        )
        pon_intensity = self._roi_intensities(roi_rect, self.pon_images)

        # poff_guassain_intensity, poff_gussian_com_x, poff_gussian_com_y = self._roi_gaussian(
        #     roi_rect, self.poff_images
        # )
        # pon_guassain_intensity, pon_gussian_com_x, pon_gussian_com_y = self._roi_gaussian(
        #     roi_rect, self.pon_images
        # )

        roi_df = pd.DataFrame(
            data={
                "poff_com_x": mul_delta_q(poff_com_x - poff_com_x[0]),
                "poff_com_y": mul_delta_q(poff_com_y - poff_com_y[0]),
                "poff_intensity": poff_intensity / poff_intensity[0],
                "pon_com_x": mul_delta_q(pon_com_x - pon_com_x[0]),
                "pon_com_y": mul_delta_q(pon_com_y - pon_com_y[0]),
                "pon_intensity": pon_intensity / pon_intensity[0],
                # "poff_gussian_com_x": poff_gussian_com_x,
                # "poff_gussian_com_y": poff_gussian_com_y,
                # "poff_guassain_intensity": poff_guassain_intensity / poff_guassain_intensity[0],
                # "pon_gussian_com_x": pon_gussian_com_x,
                # "pon_gussian_com_y": pon_gussian_com_y,
                # "pon_guassain_intensity": pon_guassain_intensity / pon_guassain_intensity[0],
            }
        )
        roi_df = roi_df.set_index(self.delay)
        return roi_df
