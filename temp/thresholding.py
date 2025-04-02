import os

from scipy.io import loadmat, savemat
import numpy as np

from src.config.config import load_config

def main() -> None:
    config = load_config()
    run = 22

    matfile = os.path.join(config.path.mat_dir, f"run={run:04d}_scan=0001_poff.mat")
    mat_img = loadmat(matfile)['data']
    print(mat_img.shape)
    new_mat_img = np.where(mat_img > 4, mat_img, 0)

    new_mat_file = os.path.join(config.path.mat_dir, f"run={run:04d}_scan=0001_poff_threshold4.mat")

    savemat(new_mat_file, {'data':new_mat_img})


if __name__ == '__main__':
    main()