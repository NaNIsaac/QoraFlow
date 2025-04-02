import os

from typing import Optional

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import numpy.typing as npt
from roi_rectangle import RoiRectangle

from src.integrater.loader import get_hdf5_images
from src.filesystem import get_run_scan_dir
from src.config.config import load_config, ExpConfig


class RoiSelector:
    def __init__(self):
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.fx, self.fy = -1, -1
        self.rect = None
        self.ax = None

    def on_mouse_press(self, event):

        if event.inaxes is not None:
            if event.button == 1:
                self.drawing = True
                self.ix, self.iy = int(event.xdata), int(event.ydata)
                self.fx, self.fy = self.ix, self.iy
                if self.rect is not None:
                    self.rect.remove()
                self.rect = patches.Rectangle((self.ix, self.iy), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
                self.ax.add_patch(self.rect)
                plt.draw()

    def on_mouse_release(self, event):

        if event.inaxes is not None and self.drawing:
            self.drawing = False
            self.fx, self.fy = int(event.xdata), int(event.ydata)
            if self.rect is not None:
                self.rect.set_width(self.fx - self.ix)
                self.rect.set_height(self.fy - self.iy)
                plt.draw()

    def on_mouse_move(self, event):

        if event.inaxes is not None and self.drawing:
            self.fx, self.fy = int(event.xdata), int(event.ydata)
            if self.rect is not None:
                self.rect.set_width(self.fx - self.ix)
                self.rect.set_height(self.fy - self.iy)
                plt.draw()

    def select_roi(self, image: npt.NDArray) -> Optional[tuple[int, int, int, int]]:
        if image.ndim != 2:
            raise TypeError(f"Invalid shape {image.shape} for image data")
        fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.imshow(image)

        fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        plt.show()

        if self.ix == -1 or self.iy == -1 or self.fx == -1 or self.fy == -1:
            return None
        x1, y1 = min(self.ix, self.fx), min(self.iy, self.fy)
        x2, y2 = max(self.ix, self.fx), max(self.iy, self.fy)
        return (x1, y1, x2, y2)


def select_roi_by_run_scan(run: int, scan: int, index_mode: Optional[int] = None) -> Optional[RoiRectangle]:
    config: ExpConfig = load_config()
    load_dir = config.path.load_dir
    scan_dir = get_run_scan_dir(load_dir, run, scan)
    files = os.listdir(scan_dir)

    if index_mode is None:
        index = len(files) // 2
    elif isinstance(index_mode, int):
        index = index_mode

    images = get_hdf5_images(os.path.join(scan_dir, files[index]), config)
    image = np.log1p(images.sum(axis=0))
    image = np.maximum(0, image)
    roi = RoiSelector().select_roi(image)
    if roi is None:
        return None
    return RoiRectangle(*roi)


def get_roi_auto(
    image,
    width: int = 5,
) -> RoiRectangle:
    """get roi_rect by max pixel"""
    center = np.unravel_index(np.argmax(image), image.shape)[::-1]
    return RoiRectangle(center[0] - width, center[1] - width, center[0] + width, center[1] + width)


if __name__ == "__main__":
    roi_rect = select_roi_by_run_scan(144, 1, 0)
    print(roi_rect)
