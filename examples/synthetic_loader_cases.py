"""Deterministic miniature PAL-XFEL HDF5 scenarios with machine-readable truth.

This tests file/metadata behavior, not a physical XFEL beam or detector.
No measured data or local credentials are used.

Example:
    python examples/synthetic_loader_cases.py --output /tmp/qoraflow-cases
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

N = 8
PUMP_COLUMN = "timestamp_info.RATE_HX_10HZ"
CASES = (
    "normal",
    "shuffled_metadata",
    "missing_pump_flags",
    "partial_metadata",
    "no_qbpm_timestamp_overlap",
)


def write_case(root: Path, scenario: str) -> tuple[Path, dict]:
    """Write one PAL-shaped file plus truth JSON; no QoraFlow imports required."""
    if scenario not in CASES:
        raise ValueError(f"Unknown scenario: {scenario!r}")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{scenario}.h5"

    timestamps = np.arange(1000, 1000 + N, dtype=np.int64)
    order = np.arange(N)
    if scenario == "shuffled_metadata":
        order = np.array([6, 2, 7, 0, 5, 1, 4, 3])
    elif scenario == "partial_metadata":
        order = np.array([0, 2, 4, 6])

    flags = np.array([bool(i % 2) for i in order], dtype=bool)
    if scenario == "missing_pump_flags":
        flags = flags.astype(float)
        flags[2] = np.nan
        flags[5] = np.nan

    metadata = pd.DataFrame(
        {PUMP_COLUMN: list(flags), "delay_value": 1.5},
        index=timestamps[order],
    )
    metadata.to_hdf(path, key="metadata", mode="w")

    # Each frame carries its source frame index in every pixel.
    # The sum of the four QBPM channels is 10*(frame_index+1).
    with h5py.File(path, "a") as hf:
        images = hf.create_group("detector/eh1/jungfrau1/image")
        images["block0_items"] = timestamps
        images["block0_values"] = np.broadcast_to(
            np.arange(N, dtype=np.float32)[:, None, None], (N, 3, 4)
        ).copy()

        for channel in range(1, 5):
            qbpm = hf.create_group(f"qbpm/eh1/qbpm1/waveforms.ch{channel}")
            qbpm["axis1"] = timestamps + (
                5000 if scenario == "no_qbpm_timestamp_overlap" else 0
            )
            qbpm["block0_values"] = (
                (np.arange(N, dtype=np.float32) + 1)[:, None] * 2.5
            )

    valid = {int(i) for i in order}
    if scenario == "missing_pump_flags":
        valid.difference_update({2, 5})
    if scenario == "no_qbpm_timestamp_overlap":
        valid.clear()
    truth = {
        "scenario": scenario,
        "expected_frame_indices": sorted(valid),
        "expected_pump_on": {str(i): bool(i % 2) for i in sorted(valid)},
        "expected_qbpm": {str(i): float(10 * (i + 1)) for i in sorted(valid)},
        "expected_delay": 1.5,
        "expect_loader_error": scenario == "no_qbpm_timestamp_overlap",
        "provenance": "synthetic, not measured data",
    }
    (root / f"{scenario}.json").write_text(
        json.dumps(truth, indent=2) + "\n", encoding="utf-8"
    )
    return path, truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=CASES)
    args = parser.parse_args()
    for name in ([args.scenario] if args.scenario else CASES):
        path, truth = write_case(args.output, name)
        print(f"{path}: {len(truth['expected_frame_indices'])} expected frames")


if __name__ == "__main__":
    main()
