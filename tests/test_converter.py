import numpy as np
from scipy.io import loadmat

from QoraFlow.analyzer.converter import save_mat
from QoraFlow.mathematics import axis_mat2np


def test_save_mat_writes_y_x_frame_like_the_integrator(tmp_path):
    stack = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)
    path = tmp_path / "stack.mat"

    save_mat(str(path), {"poff": stack, "delay": np.arange(3.0)})

    mat = loadmat(path)
    assert mat["data"].shape == (4, 5, 3)
    np.testing.assert_array_equal(axis_mat2np(mat["data"]), stack)
    np.testing.assert_array_equal(mat["delay"].ravel(), np.arange(3.0))
