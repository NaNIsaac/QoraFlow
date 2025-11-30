from collections.abc import Callable
from functools import partial

import numpy as np
import numpy.typing as npt
from roi_rectangle import RoiRectangle

from .generic_preprocessors import (
    add_bias,
    div_images_by_qbpm,
    equalize_brightness,
    filter_images_qbpm_by_linear_model,
    ransac_regression,
    subtract_dark,
)

ImagesQbpm = tuple[npt.NDArray, npt.NDArray]
ImagesQbpmProcessor = Callable[[ImagesQbpm], ImagesQbpm]


# --- [Top-Level Helper Functions for Pickling] ---

def _filter_and_normalize_by_qbpm_impl(roi_rect: RoiRectangle | None, images_qbpm: ImagesQbpm) -> ImagesQbpm:
    """Implementation for QBPM filtering and normalization."""
    images, qbpm = images_qbpm
    
    if roi_rect is None:
        roi_images = images
    else:
        roi_images = roi_rect.slice(images)

    # intensity sum within ROI
    roi_intensities = roi_images.sum((1, 2))

    # 1. Filter by QBPM range (mean +/- 2*std)
    qbpm_mask = np.logical_and(
        qbpm > qbpm.mean() - qbpm.std() * 2,
        qbpm < qbpm.mean() + qbpm.std() * 2,
    )

    # 2. Filter by Signal/QBPM Ratio (Outliers)
    signal_ratios = roi_intensities[qbpm_mask] / qbpm[qbpm_mask]
    
    valid = np.logical_and(
        signal_ratios < np.median(signal_ratios) + np.std(signal_ratios) * 0.3,
        signal_ratios > np.median(signal_ratios) - np.std(signal_ratios) * 0.3,
    )

    valid_qbpm = qbpm[qbpm_mask][valid]
    
    # Normalize images
    # Image_new = Image * (Mean_QBPM / Current_QBPM)
    valid_images = images[qbpm_mask][valid] / valid_qbpm[:, np.newaxis, np.newaxis] * np.mean(valid_qbpm)

    return valid_images, valid_qbpm


def _threshold_impl(n: float, images_qbpm: ImagesQbpm) -> ImagesQbpm:
    """Implementation for thresholding."""
    images, qbpm = images_qbpm
    return np.where(images > n, images, 0), qbpm


def _remove_ransac_roi_outliers_impl(roi_rect: RoiRectangle, images_qbpm: ImagesQbpm) -> ImagesQbpm:
    """Implementation for RANSAC outlier removal within ROI."""
    roi_image = roi_rect.slice(images_qbpm[0])
    mask = ransac_regression(
        roi_image.sum(axis=(1, 2)), images_qbpm[1], min_samples=2
    )[0]
    return images_qbpm[0][mask], images_qbpm[1][mask]


# --- [Factory Functions] ---

def make_qbpm_roi_normalizer(roi_rect: RoiRectangle | None) -> ImagesQbpmProcessor:
    """Returns a picklable processor for QBPM normalization."""
    return partial(_filter_and_normalize_by_qbpm_impl, roi_rect)


def make_thresholder(n: float) -> ImagesQbpmProcessor:
    """Returns a picklable processor for thresholding."""
    return partial(_threshold_impl, n)


def create_ransac_roi_outlier_remover(roi_rect: RoiRectangle) -> ImagesQbpmProcessor:
    """Returns a picklable processor for RANSAC outlier removal."""
    return partial(_remove_ransac_roi_outliers_impl, roi_rect)


def create_linear_model_outlier_remover(sigma: float) -> ImagesQbpmProcessor:
    """Returns a picklable processor for Linear Model outlier removal."""
    return partial(filter_images_qbpm_by_linear_model, sigma=sigma)


# --- [Standard Processors] ---

def no_negative(images_qbpm: ImagesQbpm) -> ImagesQbpm:
    """No value below zero"""
    return np.maximum(images_qbpm[0], 0), images_qbpm[1]


def shift_to_positive(images_qbpm: ImagesQbpm) -> ImagesQbpm:
    """Shift images to ensure non-negative values."""
    return add_bias(images_qbpm[0]), images_qbpm[1]


def subtract_dark_background(images_qbpm: ImagesQbpm) -> ImagesQbpm:
    """Remove dark background."""
    return subtract_dark(images_qbpm[0]), images_qbpm[1]


def normalize_images_by_qbpm(images_qbpm: ImagesQbpm) -> ImagesQbpm:
    """Normalize images by QBPM values."""
    return div_images_by_qbpm(images_qbpm[0], images_qbpm[1]), images_qbpm[1]


def remove_outliers_using_ransac(images_qbpm: ImagesQbpm) -> ImagesQbpm:
    """Remove outliers using RANSAC on total intensity."""
    mask = ransac_regression(
        images_qbpm[0].sum(axis=(1, 2)), images_qbpm[1], min_samples=2
    )[0]
    return images_qbpm[0][mask], images_qbpm[1][mask]


def equalize_intensities(images_qbpm: ImagesQbpm) -> ImagesQbpm:
    """Equalize image brightness."""
    return equalize_brightness(images_qbpm[0]), images_qbpm[1]