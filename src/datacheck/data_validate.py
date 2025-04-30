from pathlib import Path

import numpy as np


def find_outliers_mad(data, threshold=3):
    """MAD를 이용하여 이상치를 찾습니다.

    Args:
        data (pd.Series 또는 np.array): 데이터 배열.
        threshold (float): 이상치로 간주할 MAD 배수 (기본값은 3).

    Returns:
        pd.Series: 이상치로 판단되는 데이터의 boolean Series.
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z = (
        0.6745 * (data - median) / mad
    )  # 정규 분포 가정 시 MAD 기반 Z-점수 근사
    return np.where(np.abs(modified_z) > threshold)


def validata_sizes(path: str | Path):
    path = Path(path)
    files = path.glob("p*.h5")
    sizes = [file.stat().st_size for file in files]
    print(find_outliers_mad(sizes))


if __name__ == "__main__":
    path: Path = Path("Y:\\230518_FXS\\raw_data\\h5\\type=raw\\run=005\\scan=001")
    validata_sizes(path)
