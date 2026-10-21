import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from roi_rectangle import RoiRectangle
from scipy.ndimage import rotate
from scipy.optimize import curve_fit

from ..mathematics import gaussian, mul_delta_q


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

        states = {state: data[state] for state in ("poff", "pon") if state in data}
        if not states:
            raise ValueError(f"The file {file} contains neither 'poff' nor 'pon' images")
        # A run without a pump (pump_setting 0HZ) legitimately has only 'poff'. A missing state
        # stays None instead of being invented, and its ROI columns come out NaN.
        for missing in {"poff", "pon"} - states.keys():
            warnings.warn(
                f"{file.name} has no '{missing}' images; '{missing}' results will be NaN "
                "(expected for a pump_setting 0HZ run)",
                UserWarning,
                stacklevel=2,
            )

        def prepare(images: npt.NDArray | None) -> npt.NDArray | None:
            if images is None:
                return None
            if angle:
                images = rotate(images, angle, axes=(1, 2), reshape=False)
            return np.maximum(0, images)

        self.poff_images: npt.NDArray | None = prepare(states.get("poff"))
        self.pon_images: npt.NDArray | None = prepare(states.get("pon"))

    def _both_states(self) -> tuple[npt.NDArray, npt.NDArray]:
        missing = [name for name in ("poff", "pon") if getattr(self, f"{name}_images") is None]
        if missing:
            raise ValueError(f"needs both pump states; this file has no {' or '.join(missing)}")
        return self.poff_images, self.pon_images

    def get_summed_image(self) -> tuple[npt.NDArray | None, npt.NDArray | None]:
        """
        return:
            pump off image, pump on image (None for a state the file does not have)
        """
        return tuple(
            None if images is None else images.sum(axis=0)
            for images in (self.poff_images, self.pon_images)
        )

    def pon_subtract_by_poff(self):
        """Subtrack pump on images by pump off images"""
        poff, pon = self._both_states()
        return np.maximum(pon - poff, 0)

    def _roi_center_of_masses(
        self, roi_rect: RoiRectangle, images: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray]:
        roi_images = roi_rect.slice(images)
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

    def analyze_by_roi(self, roi_rect: RoiRectangle) -> pd.DataFrame:
        columns = {}
        for state in ("poff", "pon"):
            images = getattr(self, f"{state}_images")
            if images is None:
                nan = np.full(len(self.delay), np.nan)
                columns |= {f"{state}_com_x": nan, f"{state}_com_y": nan, f"{state}_intensity": nan}
                continue
            com_x, com_y = self._roi_center_of_masses(roi_rect, images)
            intensity = self._roi_intensities(roi_rect, images)
            columns |= {
                f"{state}_com_x": mul_delta_q(com_x - com_x[0]),
                f"{state}_com_y": mul_delta_q(com_y - com_y[0]),
                f"{state}_intensity": intensity / intensity[0],
            }
        order = [
            "poff_com_x",
            "poff_com_y",
            "poff_intensity",
            "pon_com_x",
            "pon_com_y",
            "pon_intensity",
        ]
        roi_df = pd.DataFrame(data={key: columns[key] for key in order})
        roi_df = roi_df.set_index(self.delay)
        return roi_df
