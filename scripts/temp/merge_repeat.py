import os

import numpy as np
from roi_rectangle import RoiRectangle
from scipy.io import loadmat, savemat
from scipy.ndimage import center_of_mass
from tifffile import imwrite

from CordaX.gui.roi_core import RoiSelector
from CordaX.config import ConfigManager


def shift_image(arr, dx, dy):
    dx = round(dx)
    dy = round(dy)
    arr = np.roll(arr, dy, axis=-2)
    arr = np.roll(arr, dx, axis=-1)
    if dy > 0:
        arr[..., :dy, :] = 0
    elif dy < 0:
        arr[..., dy:, :] = 0
    if dx > 0:
        arr[..., :, :dx] = 0
    elif dx < 0:
        arr[..., :, dx:] = 0
    return arr


def main() -> None:

    config = ConfigManager.load_config()
    runs = [85, 87]

    arr_list = []

    for run in runs:
        matfile = os.path.join(config.path.mat_dir, f"run={run:04d}_scan=0001_poff.mat")
        mat_arr = loadmat(matfile)["data"]
        arr = np.transpose(mat_arr, [2, 0, 1])
        arr_list.append(arr)

    first_img_sum = arr_list[0].sum(0)
    roi = RoiSelector().select_roi(first_img_sum)

    roi_rect = RoiRectangle.from_tuple(roi)
    roi_arr = roi_rect.slice(arr)
    x0, y0 = center_of_mass(roi_arr.sum(0))

    shifted_arr_list = [arr_list[0]]
    for arr in arr_list[1:]:
        roi_arr = roi_rect.slice(arr)
        x, y = center_of_mass(roi_arr.sum(0))
        dx, dy = x0 - x, y0 - y
        shifted_arr = shift_image(arr, dx, dy)
        shifted_arr_list.append(shifted_arr)
        print(shifted_arr.shape)

    merged_img = np.mean(np.stack(shifted_arr_list, axis=0), axis=0)

    mat_merged_img = np.transpose(merged_img, [1, 2, 0])

    merged_img_name = "_".join(map(str, runs))
    merged_mat_name = merged_img_name + ".mat"
    merged_mat_file = os.path.join(config.path.mat_dir, merged_mat_name)
    savemat(merged_mat_file, {"data": mat_merged_img})

    merged_tif_name = merged_img_name + ".tif"
    merged_tif_file = os.path.join(config.path.mat_dir, merged_tif_name)
    imwrite(merged_tif_file, merged_img)


if __name__ == "__main__":
    main()
