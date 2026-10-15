from pathlib import Path
import zarr
import polars as pl

def read_zarr(file: Path | str, parquet_file: Path | str):
    with zarr.open(file, mode='r') as root:
        det = root['det-eh1-jungfrau2']
        qbpm = root['qbpm-eh1-qbpm1']
    df = pl.read_parquet(parquet_file)
    return det, qbpm, df