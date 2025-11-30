from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit
from sklearn.linear_model import RANSACRegressor

from ..config import ConfigManager
from ..logger import setup_logger

# Logger setup at module level
logger = setup_logger()

def ransac_regression(
    y: np.ndarray, x: np.ndarray, min_samples: int | None = None
) -> tuple[npt.NDArray[np.bool_], npt.NDArray, npt.NDArray]:
    """
    Perform RANSAC (Random Sample Consensus) regression to identify inliers and estimate the regression model.

    Parameters:
    - y (np.ndarray): The target variable array.
    - x (np.ndarray): The feature variable array.
    - min_samples (int, optional): The minimum number of samples to fit the model. Default is 3.

    Returns:
    - tuple[npt.NDArray[np.bool_], npt.NDArray, npt.NDArray]:
        A tuple containing the inlier mask, coefficient, and intercept of the linear model.
    """
    # Safety check for small datasets
    if len(y) < 2:
        return np.ones(len(y), dtype=bool), np.array([0]), 0.0

    X = x[:, np.newaxis]
    # Ensure min_samples is valid
    ms = min_samples if min_samples is not None else max(2, int(len(y) * 0.1))
    
    try:
        ransac = RANSACRegressor(min_samples=ms).fit(X, y)
        inlier_mask = ransac.inlier_mask_
        return inlier_mask, ransac.estimator_.coef_, ransac.estimator_.intercept_
    except ValueError:
        logger.warning("RANSAC fitting failed (too few samples or noise). Returning all True mask.")
        return np.ones(len(y), dtype=bool), np.array([0]), 0.0


def get_linear_regression_confidence_bounds(
    y: npt.NDArray, x: npt.NDArray, sigma: float
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Get lower and upper bounds for data points based on their confidence interval in a linear regression model.

    Statistical Explanation:
    This function applies the principle of propagation of uncertainty. It fits a linear
    regression model y = mx + b, calculates the standard errors of the model parameters
    m and b, and uses these to generate prediction intervals for identifying outliers.

    The prediction interval is calculated using the formula:
    y ± sqrt((m_err * x)^2 + b_err^2) * sigma
    where m_err and b_err are the standard errors of m and b respectively.

    Parameters:
    y (NDArray): Dependent variable data. Shape: (N,)
    x (NDArray): Independent variable data. Shape: (N,)
    sigma (float): Number of standard deviations for the confidence interval. Default is 3.0.

    Returns:
    lowerbound (NDArray): Lower bounds of y
    upperbound (NDArray): Upper bounds of y
    y_fit (NDArray): Fitted y

    Note:
    This method assumes a linear relationship in the data. For strong non-linearities,
    a different approach may be necessary.
    """
    if len(y) < 2:
        return y, y, y

    def linear_model(x, m, b):
        return m * x + b

    try:
        params, covars = curve_fit(linear_model, x, y)
        m, b = params
        m_err, b_err = np.sqrt(np.diag(covars))
        y_fit = linear_model(x, m, b)

        # Calculate upper and lower bounds considering both slope and intercept errors
        error = np.sqrt((m_err * x) ** 2 + b_err**2)
        upper_bound = y_fit + error * sigma
        lower_bound = y_fit - error * sigma

        return lower_bound, upper_bound, y_fit
    except Exception as e:
        logger.warning(f"Linear regression failed: {e}")
        # Fail-safe: Return infinite bounds so no data is filtered
        return y - np.inf, y + np.inf, y


def filter_images_qbpm_by_linear_model(
    images: npt.NDArray, qbpm: npt.NDArray, sigma: float
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Filter images based on the confidence interval of their intensities
    using a linear regression model with QBPM values.

    This function computes the total intensity of each image, applies a linear regression model
    to the intensity and QBPM values, and generates a mask to filter out images whose intensities
    fall outside the specified confidence interval.

    Parameters:
    images (NDArray): Array of images. Shape: (N, H, W), where N is the number of images,
        and H and W are the height and width of each image.
    qbpm (NDArray): Array of QBPM (Quadrature Balanced Photodetector Measurements) values. Shape: (N,)
    sigma (float): Number of standard deviations for the confidence interval.

    Returns:
    tuple[NDArray, NDArray]: Tuple of filtered intensities and QBPM values.
        - Filtered intensities (NDArray):
            Array of intensities within the confidence interval. Shape: (M,), where M <= N.
        - Filtered qbpm (NDArray):
            Array of QBPM values corresponding to the filtered intensities. Shape: (M,), where M <= N.

    Note:
    This method uses the `get_linear_regression_confidence_lower_upper_bound` function to generate the mask based
    on the linear regression model and confidence interval.
    """
    intensites = images.sum(axis=(1, 2))
    lower_bound, upper_bound, _ = get_linear_regression_confidence_bounds(
        intensites, qbpm, sigma
    )
    mask = np.logical_and(intensites >= lower_bound, intensites <= upper_bound)

    return images[mask], qbpm[mask]


def div_images_by_qbpm(images: npt.NDArray, qbpm: npt.NDArray) -> npt.NDArray:
    """
    Divide images by qbpm.

    Parameters:
    images (NDArray): Array of images.
        shape=(N, H, W), where N is the number of images,
        and H and W are the height and width of each image.
    qbpm (NDArray): Array of QBPM (Quadrature Balanced Photodetector Measurements) values. Shape: (N,)

    Returns:
    NDArray: Images that divided by qbpm
    """
    # Prevent division by zero
    safe_qbpm = np.where(qbpm == 0, 1, qbpm)
    return images * safe_qbpm.mean() / safe_qbpm[:, np.newaxis, np.newaxis]


@lru_cache(maxsize=1)
def _load_cached_dark_image(dark_file_path: Path) -> Optional[np.ndarray]:
    """
    [Optimization] Load dark image only ONCE and cache it in RAM.
    This prevents IO bottleneck when processing thousands of chunks.
    """
    if not dark_file_path.exists():
        logger.warning(f"No dark image file found: \"{dark_file_path}\"")
        return None
    try:
        dark_images = np.load(dark_file_path)
        # If multiple dark frames exist, average them
        if dark_images.ndim == 3:
            return np.mean(dark_images, axis=0)
        return dark_images
    except Exception as e:
        logger.error(f"Failed to load dark image: {e}")
        return None


def subtract_dark(images: npt.NDArray) -> npt.NDArray:
    """
    Subtract dark background. Uses caching for performance.

    Parameters:
    - images (npt.NDArray): Input images.

    Returns:
    - npt.NDArray: Background subtracted images.
    """
    config = ConfigManager.load_config()
    dark_file = config.path.analysis_dir / "dark_images" / "dark.npy"

    # Use cached loader
    dark = _load_cached_dark_image(dark_file)

    if dark is None:
        return images

    none_negative_dark = np.maximum(dark, 0)
    
    # Broadcasting subtraction: (N, H, W) - (1, H, W)
    return np.maximum(0, images - none_negative_dark[np.newaxis, :, :])


def add_bias(images: npt.NDArray):
    bias = np.min(images)
    return images - bias


def equalize_brightness(images: np.ndarray) -> np.ndarray:
    """
    Equalize the brightness of each image in the 3D array while maintaining the overall average brightness.

    Parameters:
    - images: np.ndarray, 3D array of images (num_images, height, width).

    Returns:
    - np.ndarray: The brightness-equalized images.
    """
    intensites = images.sum(axis=(1, 2))
    mean_intensity = intensites.mean()
    
    if mean_intensity == 0:
        return images
        
    overal_normed_intensites = intensites / mean_intensity
    
    # Prevent division by zero
    overal_normed_intensites = np.where(
        overal_normed_intensites == 0, 1, overal_normed_intensites
    )
    
    equalized_images = images / overal_normed_intensites[:, np.newaxis, np.newaxis]

    return equalized_images