from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.integrate import dblquad, quad

from src.config import load_config

FWHM_COEFFICIENT: Final[float] = 2.35482  # FWHM_COEFFICIENT = 2 * np.sqrt(2 * np.log(2))
WAVELENGTH_COEFFICIENT: Final[float] = 12.398419843320025


def reverse_axis(array: npt.NDArray):
    return np.transpose(array, axes=range(array.ndim)[::-1])


def gaussian(x: npt.NDArray, a: float, mu: float, sig: float) -> npt.NDArray:
    return a * np.exp(-(x - mu) ** 2 / (2 * sig ** 2))


def integrate_FWHM(a: float, mu: float, sig: float) -> float:
    fwhm = FWHM_COEFFICIENT * np.abs(sig)
    result, _ = quad(gaussian, mu - 0.5 * fwhm, mu + 0.5 * fwhm, args=(a, mu, sig))
    return result


def gaussian2d(
    xy, amplitude: float,
    x0: float, y0: float,
    sigma_x: float, sigma_y: float,
    theta: float, offset: float
) -> npt.NDArray:
    """
    Calculate the 2D Gaussian distribution at the given coordinates.

    Parameters:
        xy (tuple of arrays): A tuple containing two arrays 'x' and 'y' representing the 2D coordinates.
        amplitude (float): The amplitude (peak value) of the Gaussian.
        x0 (float): The x-coordinate of the center of the Gaussian.
        y0 (float): The y-coordinate of the center of the Gaussian.
        sigma_x (float): The standard deviation in the x-direction.
        sigma_y (float): The standard deviation in the y-direction.
        theta (float): The rotation angle in radians. The Gaussian will be rotated counterclockwise by this angle.
        offset (float): The constant offset added to the Gaussian distribution.

    Returns:
        numpy.ndarray: A 1D array containing the values of the 2D Gaussian distribution flattened into a 1D array.
    """
    x, y = xy
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = offset + amplitude * np.exp(-(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2)))
    return g.ravel()


def integrate_fwhm_2d(
    amplitude: float,
    xo: float, yo: float,
    sigma_x: float, sigma_y: float,
    theta: float, offset: float
) -> float:
    """
    Calculate the integral of a 2D Gaussian with an offset over its FWHM
    """
    def integrand(y, x):
        xy = np.meshgrid(x, y)
        return gaussian2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset)

    # Calculate FWHM in x and y directions
    fwhm_x = FWHM_COEFFICIENT * np.abs(sigma_x)
    fwhm_y = FWHM_COEFFICIENT * np.abs(sigma_y)

    x0 = xo
    y0 = yo
    # Define integration limits based on FWHM
    x_lower = x0 - fwhm_x * 0.5
    x_upper = x0 + fwhm_x * 0.5
    y_lower = y0 - fwhm_y * 0.5
    y_upper = y0 + fwhm_y * 0.5

    # Perform the double integration
    result, _ = dblquad(integrand, y_lower, y_upper, lambda x: x_lower, lambda x: x_upper)
    return result


def get_wavelength(beam_energy: float) -> float:
    return WAVELENGTH_COEFFICIENT / beam_energy


def pixel_to_del_q(pixels: npt.NDArray) -> npt.NDArray:

    config = load_config()
    del_pixels = pixels - pixels[0]
    del_two_theta = np.arctan2(config.param.dps, config.param.sdd * del_pixels)
    wavelength = get_wavelength(config.param.beam_energy)
    return 4 * np.pi / wavelength * np.sin(del_two_theta / 2)


def pixel_to_q(pixels: npt.NDArray) -> npt.NDArray:
    """
    two_theta = arctan(dps * pixels / sdd)
    Q = (4 * pi / wavelength) * sin(two_theta / 2)
      = (4 * pi / wavelength) * two_theta / 2
      = (4 * pi / wavelength) * arctan(dps * pixels / sdd) / 2
      = pixels * (4 * pi / wavelength) * arctan(dps / sdd) / 2
    """
    config = load_config()
    wavelength = get_wavelength(config.param.beam_energy)
    two_theta = np.arctan2(config.param.dps, config.param.sdd * pixels)
    return 4 * np.pi / wavelength * np.sin(two_theta / 2)


def mul_delta_q(pixels: npt.NDArray) -> npt.NDArray:
    config = load_config()
    two_theta = np.arctan2(config.param.dps, config.param.sdd)
    wavelength = get_wavelength(config.param.beam_energy)
    delta_q = (4 * np.pi / wavelength) * two_theta
    del_pixels = pixels - pixels[0]
    return del_pixels * delta_q


def non_outlier_indices_percentile(
    arr: npt.NDArray,
    lower_percentile: float,
    upper_percentile: float
) -> npt.NDArray[np.bool_]:
    """
    Get the indices of non-outliers in a NumPy array.

    Parameters:
    arr (np.ndarray): Input NumPy array containing data.

    Returns:
    np.ndarray: Boolean array with the same shape as 'arr' where True indicates non-outlier data points.
    """

    # Calculate the first quartile (Q1) and third quartile (Q3)
    q1 = np.percentile(arr, lower_percentile)
    q3 = np.percentile(arr, upper_percentile)

    # Calculate the Interquartile Range (IQR)
    iqr = q3 - q1

    # Define the lower and upper bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Create a boolean array to identify non-outliers
    conditions = np.logical_and(arr > lower_bound, arr < upper_bound)

    return conditions


if __name__ == '__main__':
    arr = np.array([1, 2, 3])
    print(mul_delta_q(arr))
    print(pixel_to_q(arr))
    print(pixel_to_del_q(arr))
