import os

import h5py
import pandas as pd


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


def h5_tree(val, pre: None = "") -> None:
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
                h5_tree(val, f"{pre}    ")
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
    return pd.read_hdf(h5file, "metadata")


if __name__ == "__main__":
    from CordaX.config import ConfigManager
    from CordaX.filesystem import get_run_scan_dir

    ConfigManager.initialize("config.yaml")
    config = ConfigManager.load_config()
    load_dir: str = config.path.load_dir

    file = get_run_scan_dir(load_dir, 162, 1, sub_path="p0050.h5")
    print(pd.DataFrame(pd.read_hdf(file, "metadata")))

    with h5py.File(file, "r") as hf:
        print(hf)
        h5_tree(hf)
