import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.widgets import Slider

from src.config import load_config
from src.filesystem import get_run_scan_dir
from src.integrator.loader import PalXFELLoader
from src.preprocessor.generic_preprocessors import (
    get_linear_regression_confidence_bounds, ransac_regression)


def find_outliers_gui(y: npt.NDArray, x: npt.NDArray) -> float:
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9)

    # Initial plot
    sigma_init = 3.0
    lb, ub, yf = get_linear_regression_confidence_bounds(y, x, sigma_init)
    within_bounds = (y >= lb) & (y <= ub)

    normal_points, = ax.plot(
        x[within_bounds], y[within_bounds],
        'o', color='blue', markersize=3,
        label='Normal Points'
    )
    outlier_points, = ax.plot(
        x[~within_bounds], y[~within_bounds],
        'o', color='red', markersize=3,
        label='Outliers'
    )
    line_fit, = ax.plot(x, yf, color='black', label='Fitted Line')
    ax.fill_between(x, lb, ub, color='gray', alpha=0.2, label='Confidence Interval')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Linear Regression with Confidence Interval for Outlier Detection')
    y_range = np.max(y) - np.min(y)
    ax.set_ylim([np.min(y) - y_range * 0.1, np.max(y) + y_range * 0.1])
    ax.legend(loc='lower right')

    # Add sigma value text
    sigma_text = ax.text(0.02, 0.98, f'Sigma: {sigma_init:.1f}', transform=ax.transAxes, verticalalignment='top')

    # Add outlier count text
    outlier_count = np.sum(~within_bounds)
    outlier_text = ax.text(
        0.02, 0.93,
        f'Outliers: {outlier_count} ({outlier_count / len(y):.1%})',
        transform=ax.transAxes, verticalalignment='top'
    )

    axsigma = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    sigma_slider = Slider(axsigma, 'Sigma', 0.1, 20.0, valinit=sigma_init, valstep=0.1)

    def update(val):
        sigma = sigma_slider.val
        lb, ub, yf = get_linear_regression_confidence_bounds(y, x, sigma)
        line_fit.set_ydata(yf)

        # Clear previous fill_between collection
        for coll in ax.collections:
            if isinstance(coll, plt.matplotlib.collections.PolyCollection):
                coll.remove()

        ax.fill_between(x, lb, ub, color='gray', alpha=0.2)

        within_bounds = (y >= lb) & (y <= ub)
        normal_points.set_data(x[within_bounds], y[within_bounds])
        outlier_points.set_data(x[~within_bounds], y[~within_bounds])

        # Update sigma text
        sigma_text.set_text(f'Sigma: {sigma:.1f}')

        # Update outlier count
        outlier_count = np.sum(~within_bounds)
        outlier_text.set_text(f'Outliers: {outlier_count} ({outlier_count / len(y):.1%})')

        fig.canvas.draw_idle()

    sigma_slider.on_changed(update)

    plt.show()

    return round(sigma_slider.val, 1)


def find_outliers_run_scan_gui(run: int, scan: int) -> float:

    config = load_config()
    scan_dir = get_run_scan_dir(config.path.load_dir, run, scan)
    files = os.listdir(scan_dir)
    file = os.path.join(scan_dir, files[len(files) // 2])

    rr = PalXFELLoader(file)
    images = rr.images
    qbpm = rr.qbpm

    return find_outliers_gui(images.sum(axis=(1, 2)), qbpm)


def RANSAC_regression_gui(run: int, scan: int) -> None:
    config = load_config()
    scan_dir = get_run_scan_dir(config.path.load_dir, run, scan)
    files = os.listdir(scan_dir)
    file = os.path.join(scan_dir, files[len(files) // 2])

    rr = PalXFELLoader(file)
    images = rr.images
    qbpm = rr.qbpm

    intensities = images.sum(axis=(1, 2))
    mask, *_ = ransac_regression(intensities, qbpm)
    plt.scatter(qbpm[mask], intensities[mask], color="blue", label="Inliers")
    plt.scatter(qbpm[~mask], intensities[~mask], color="red", label="Outliers")
    plt.title("RANSAC - outliers vs inliers")

    plt.show()


if __name__ == "__main__":
    # Example data
    np.random.seed(0)  # For reproducibility
    x = np.random.normal(10, 5, 100)
    x.sort()
    y = 2.5 * x + np.random.normal(0, 2, 100)
    y[0] += 10  # Add an outlier
    y[-1] -= 10  # Add another outlier

    sigma = find_outliers_gui(y, x)
    print(sigma)
