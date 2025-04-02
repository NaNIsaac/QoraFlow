from pathlib import Path
from typing import Optional, Generator
from src.config.config import ExpConfig


def get_run_scan_dir(mother: str | Path, run: int, scan: Optional[int] = None, file_name: Optional[str] = None) -> Path:
    """
    Generate the directory for a given run and scan number, optionally with a file number.

    Parameters:
        mother (str | Path): The base directory or path where the path will be generated.
        run (int): The run number for which the path will be generated.
        scan (int, optional): The scan number for which the path will be generated.
            If not provided, only the run directory path will be returned.
        file_num (int, optional): The file number for which the path will be generated.
            If provided, both run and scan directories will be included in the path.

    Returns:
        Path: The path representing the specified run, scan, and file number (if applicable).
    """
    mother = Path(mother)

    if scan is None and file_name is None:
        return mother / f"run={run:0>3}"
    if scan is not None and file_name is None:
        return mother / f"run={run:0>3}" / f"scan={scan:0>3}"
    if scan is not None and file_name is not None:
        return mother / f"run={run:0>3}" / f"scan={scan:0>3}" / file_name


def make_run_scan_dir(mother: str | Path, run: int, scan: int, file_name: str | Path) -> Path:
    """
    Create a nested directory structure for the given run and scan numbers.

    Parameters:
        dir (str | Path): The base directory where the nested structure will be created.
        run (int): The run number for which the directory will be created.
        scan (int): The scan number for which the directory will be created.

    Returns:
        Path: The path of the created nested directory.
    """
    mother = Path(mother)
    mother.mkdir(parents=True, exist_ok=True)

    path = mother / f'run={run:0>3d}'
    path.mkdir(parents=True, exist_ok=True)

    path = path / f'scan={scan:0>3d}'
    path.mkdir(parents=True, exist_ok=True)

    path = path / file_name if file_name is not None else path
    return path


def get_scan_nums(run_num: int, config: ExpConfig) -> list[int]:
    """Get Scan numbers from real directory"""
    run_dir: Path = get_run_scan_dir(config.path.load_dir, run_num)
    scan_folders: Generator[Path, None, None] = run_dir.iterdir()
    return [int(scan_dir.stem.split("=")[1]) for scan_dir in scan_folders]
