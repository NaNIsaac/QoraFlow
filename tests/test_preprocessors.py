import numpy as np
import pytest
from roi_rectangle import RoiRectangle

from QoraFlow.preprocessor import generic_preprocessors as gp
from QoraFlow.preprocessor import image_qbpm_preprocessor as ip


def test_add_bias_shifts_minimum_to_zero():
    images = np.array([[[-2.0, 1.0], [3.0, 0.0]]])
    out = gp.add_bias(images)
    assert out.min() == 0
    np.testing.assert_array_equal(out - images, np.full(images.shape, 2.0))


def test_div_images_by_qbpm_preserves_mean_scale():
    images = np.ones((3, 2, 2))
    qbpm = np.array([1.0, 2.0, 3.0])
    out = gp.div_images_by_qbpm(images, qbpm)
    np.testing.assert_allclose(out[:, 0, 0], qbpm.mean() / qbpm)


def test_equalize_brightness_equalizes_and_keeps_mean():
    rng = np.random.default_rng(0)
    images = rng.uniform(1, 2, size=(4, 3, 3)) * np.array([1, 2, 3, 4])[:, None, None]
    out = gp.equalize_brightness(images)
    sums = out.sum(axis=(1, 2))
    np.testing.assert_allclose(sums, np.full(4, images.sum(axis=(1, 2)).mean()))


def test_ransac_regression_flags_the_outlier():
    x = np.arange(20, dtype=float)
    y = 2 * x + 1
    y[7] += 100
    mask, coef, intercept = gp.ransac_regression(y, x, min_samples=5)
    assert not mask[7]
    assert mask.sum() == 19
    assert coef[0] == pytest.approx(2.0)
    assert intercept == pytest.approx(1.0)


def test_thresholder_zeroes_values_at_or_below_threshold():
    images = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    qbpm = np.array([1.0])
    out, q = ip.make_thresholder(2.0)((images, qbpm))
    np.testing.assert_array_equal(out, [[[0.0, 0.0], [3.0, 4.0]]])
    assert q is qbpm


def test_no_negative_and_shift_to_positive():
    images = np.array([[[-1.0, 2.0]]])
    qbpm = np.array([1.0])
    np.testing.assert_array_equal(ip.no_negative((images, qbpm))[0], [[[0.0, 2.0]]])
    np.testing.assert_array_equal(ip.shift_to_positive((images, qbpm))[0], [[[0.0, 3.0]]])


def test_lsb_quantization_floors_to_multiples_of_lsb():
    images = np.array([[[9.7, 9.8, 20.0]]])
    out, _ = ip.lsb_quantization((images, np.ones(1)))
    np.testing.assert_allclose(out, [[[0.0, 9.8, 19.6]]])


def test_normalize_images_by_qbpm_matches_generic():
    images = np.arange(8, dtype=float).reshape(2, 2, 2) + 1
    qbpm = np.array([1.0, 3.0])
    out, q = ip.normalize_images_by_qbpm((images, qbpm))
    np.testing.assert_array_equal(out, gp.div_images_by_qbpm(images, qbpm))
    np.testing.assert_array_equal(q, qbpm)


def test_qbpm_roi_normalizer_drops_qbpm_and_signal_outliers():
    n = 40
    rng = np.random.default_rng(1)
    qbpm = rng.normal(100.0, 1.0, n)
    images = np.ones((n, 4, 4)) * qbpm[:, None, None] * rng.uniform(0.9, 1.1, n)[:, None, None]
    qbpm[3] = 1000.0
    images[10] *= 3.0
    out_images, out_qbpm = ip.make_qbpm_roi_normalizer(None)((images, qbpm))
    assert 0 < len(out_images) == len(out_qbpm) < n
    assert 1000.0 not in out_qbpm
    ratios = out_images.sum((1, 2)) / out_qbpm
    assert ratios.std() / ratios.mean() < 0.15


def test_qbpm_roi_normalizer_uses_roi_slice():
    rect = RoiRectangle(y1=0, y2=2, x1=0, x2=2)
    n = 30
    rng = np.random.default_rng(2)
    qbpm = np.linspace(95, 105, n)
    images = np.ones((n, 4, 4))
    images[:, :2, :2] = qbpm[:, None, None] * rng.uniform(0.9, 1.1, n)[:, None, None]
    out_images, out_qbpm = ip.make_qbpm_roi_normalizer(rect)((images, qbpm))
    assert 0 < len(out_images) == len(out_qbpm) < n
    assert out_images.shape[1:] == (4, 4)
