import os

import pandas as pd
import h5py


def get_file_status(root: str) -> dict:

    status: dict = {}

    runs: list[str] = os.listdir(root)
    for run in runs:
        path = os.path.join(root, run)
        scans = os.listdir(path)
        for scan in scans:
            path = os.path.join(path, scan)
            files = os.listdir(path)

            name = "_".join(path[-2:])[:-2]

            nums = {int(file[1:-3]) for file in files}
            max_num = max(nums)
            missing_nums = set(range(1, max_num + 1)) - nums
            status[name] = [max_num, missing_nums]

    return status


def h5_tree(val, pre: None = '') -> None:
    """
    with h5py.File(file) as hf:
        print(hf)
        h5_tree(hf)
    """
    items_cnt = len(val)
    for key, val in val.items():
        items_cnt -= 1
        if items_cnt == 0:
            # the last item
            if isinstance(val, h5py._hl.group.Group):
                print(f"{pre}└── {key}")
                h5_tree(val, f'{pre}    ')
            else:
                try:
                    if h5py.check_string_dtype(val.dtype):
                        print(f"{pre}└── {key} ({val})")
                    else:
                        print(f"{pre}└── {key} ({val.shape})")
                except TypeError:
                    print(f"{pre}└── {key} (scalar)")
        else:
            if isinstance(val, h5py._hl.group.Group):
                print(f"{pre}├── {key}")
                h5_tree(val, f"{pre}│   ")
            else:
                try:
                    if h5py.check_string_dtype(val.dtype):
                        print(f"{pre}├── {key} ({val})")
                    else:
                        print(f"{pre}├── {key} ({val.shape})")
                except TypeError:
                    print(f"{pre}├── {key} (scalar)")


def load_matdata(h5file: str) -> pd.DataFrame:
    """Return Dataframe that contains metadata"""
    return pd.read_hdf(h5file, 'metadata')


if __name__ == "__main__":

    from src.filesystem import get_run_scan_dir
    from src.config.config import load_config
    import matplotlib.pyplot as plt
    config = load_config()
    load_dir: str = config.path.load_dir

    file = get_run_scan_dir(load_dir, 1, 1, 110)
    # file = 'Z:\\241103_DFXM\\dat\\241108-241109\\20241108_Cu_80nm_01093.h5'
    with h5py.File(file) as hf:
        # print(hf)
        # h5_tree(hf)
        img = hf['run/scan00001/det/eh2hama_img/rawData'][()][0, 0, 0, 0]
        plt.imshow(img)

    plt.show()
