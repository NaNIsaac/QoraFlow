from pathlib import Path
from typing import Generator, Optional


def get_run_scan_dir(mother: str | Path, run: int, scan: Optional[int] = None, *, sub_path: Optional[str] = None) -> Path:
    """
    Generate the directory for a given run and scan number.
    """
    mother = Path(mother)

    if scan is None and sub_path is None:
        return mother / f"run={run:0>3}"
    if scan is not None and sub_path is None:
        return mother / f"run={run:0>3}" / f"scan={scan:0>3}"
    if scan is not None and sub_path is not None:
        return mother / f"run={run:0>3}" / f"scan={scan:0>3}" / sub_path


def make_run_scan_dir(mother: str | Path, run: int, scan: int, *, sub_path: str | Path = None) -> Path:
    """
    Create a nested directory structure for the given run and scan numbers.
    Sub_path will not be created, but it will be returned as a path.
    """
    path = Path(mother) / f'run={run:0>3d}' / f'scan={scan:0>3d}'
    path.mkdir(parents=True, exist_ok=True)

    return path / sub_path if sub_path else path


def get_run_nums(path: str | Path) -> list[int]:
    """Get Run numbers from real directory"""
    run_folders: Generator[Path, None, None] = Path(path).iterdir()
    return [int(run_dir.stem.split("=")[1]) for run_dir in run_folders]


def get_scan_nums(path: str | Path, run_n: int) -> list[int]:
    """Get Scan numbers from real directory"""
    run_dir: Path = get_run_scan_dir(path, run_n)
    scan_folders: Generator[Path, None, None] = run_dir.iterdir()
    return [int(scan_dir.stem.split("=")[1]) for scan_dir in scan_folders]


if __name__ == "__main__":
    mother: Path = Path()
    path: Path = make_run_scan_dir(mother, 1, 2, sub_path='test')
    print('path:', path)
