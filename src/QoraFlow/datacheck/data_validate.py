from pathlib import Path

import numpy as np


def find_outliers_mad(data, threshold=3):
    """Indices (as from np.where) of points whose modified z-score exceeds `threshold`."""
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    # 0.6745 = Phi^-1(0.75): makes the MAD-based z comparable to a normal z-score
    modified_z = 0.6745 * (data - median) / mad
    return np.where(np.abs(modified_z) > threshold)


def validata_sizes(path: str | Path):
    path = Path(path)
    files = path.glob("p*.h5")
    sizes = [file.stat().st_size for file in files]
    print(find_outliers_mad(sizes))


if __name__ == "__main__":
    path: Path = Path("data/example/raw_data/h5/type=raw/run=005/scan=001")
    validata_sizes(path)
