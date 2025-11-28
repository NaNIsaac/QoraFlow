from pathlib import Path
from typing import Generator


def get_run_scan_dir(
    mother: str | Path,
    run: int,
    scan: int | None = None,
    *,
    sub_path: str | None = None,
) -> Path:
    """
    Generate the directory for a given run and scan number.
    Robustly handles all combinations of scan and sub_path.
    """
    # 1. Start with the Run directory (This is always required)
    # We cast run to int to ensure string formatting (:03d) works even if a string "1" is passed
    target_path = Path(mother) / f"run={int(run):03d}"

    # 2. Append Scan directory if it exists
    if scan is not None:
        target_path /= f"scan={int(scan):03d}"

    # 3. Append sub_path if it exists
    if sub_path:
        # Security/Bug fix: lstrip("/") prevents sub_path from being treated 
        # as an absolute path (which would reset the path in pathlib)
        target_path /= sub_path.lstrip("/")

    return target_path


def make_run_scan_dir(
    mother: str | Path, run: int, scan: int, *, sub_path: str | Path | None = None
) -> Path:
    """
    Create a nested directory structure for the given run and scan numbers.
    Sub_path will not be created, but it will be returned as a path.
    """
    path = Path(mother) / f"run={run:0>3d}" / f"scan={scan:0>3d}"
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
    path: Path = make_run_scan_dir(mother, 1, 2, sub_path="test")
    print("path:", path)
