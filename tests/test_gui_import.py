"""QoraFlow.gui imports without a working Qt binding (#29).

A headless machine may have PyQt6 installed but no display libraries, so importing any Qt
binding fails there. The GUI modules must still import; only opening the ROI selector needs Qt.
"""

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

BLOCK_QT = """
import importlib.abc
import sys

class NoQt(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"PyQt6", "PySide6", "PyQt5", "PySide2"}:
            raise ImportError(f"{name} blocked by the test: no Qt here")
        return None

sys.meta_path.insert(0, NoQt())
"""


def test_gui_modules_import_without_qt():
    # pyplot first, as a plotting script does: then matplotlib.use("QtAgg") switches at once
    # and imports Qt, instead of only recording the choice.
    code = BLOCK_QT + (
        "import matplotlib.pyplot\n"
        "import QoraFlow.gui.roi_core, QoraFlow.gui.select_roi, QoraFlow.gui.preprocess_gui\n"
        "from QoraFlow.gui.roi_core import get_roi_auto\n"
        "import numpy as np\n"
        "image = np.zeros((20, 20)); image[7, 12] = 1.0\n"
        "print([int(v) for v in get_roi_auto(image, half_width=2).to_tuple()])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        # pytest finds the package through its own pythonpath setting; the child needs it too
        env={k: v for k, v in os.environ.items() if k != "MPLBACKEND"} | {"PYTHONPATH": str(SRC)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[5, 9, 10, 14]"
