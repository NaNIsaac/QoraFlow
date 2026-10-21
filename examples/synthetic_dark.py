"""Write a synthetic JUNGFRAU dark frame for the notebooks and scripts/temp.

    uv run python examples/synthetic_dark.py [output.npy]

Default output: notebooks/dark.npy. The frame is generated, not measured: a flat
pedestal with Gaussian read noise. The offset and noise level are arbitrary.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SHAPE = (514, 1030)  # one JUNGFRAU 0.5M module including the double-sized pixels


def synthetic_dark(
    shape: tuple[int, int] = SHAPE, offset: float = 0.5, noise: float = 0.1, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(offset, noise, shape).astype(np.float32)


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "notebooks" / "dark.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, synthetic_dark())
    print(f"Synthetic dark frame {SHAPE} written to {out}")


if __name__ == "__main__":
    main()
