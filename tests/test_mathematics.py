import numpy as np
import pytest
from scipy.special import erf

from QoraFlow import mathematics as m


def test_min_max_scale():
    out = m.min_max_scale(np.array([2.0, 4.0, 10.0]))
    np.testing.assert_allclose(out, [0.0, 0.25, 1.0])


def test_axis_round_trip_and_shape():
    arr = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    mat = m.axis_np2mat(arr)
    assert mat.shape == (4, 5, 2, 3)
    np.testing.assert_array_equal(m.axis_mat2np(mat), arr)


def test_axis_conversion_passes_through_low_dimensions():
    arr = np.arange(4)
    assert m.axis_np2mat(arr) is arr
    assert m.axis_mat2np(arr) is arr


def test_gaussian_peak_and_symmetry():
    x = np.array([-1.0, 0.0, 1.0]) + 2.0
    out = m.gaussian(x, a=3.0, mu=2.0, sig=0.5)
    assert out[1] == pytest.approx(3.0)
    assert out[0] == pytest.approx(out[2])


def test_integrate_fwhm_matches_analytic_erf():
    a, mu, sig = 2.0, 1.0, 0.7
    half_width = m.FWHM_COEFFICIENT * sig / 2
    expected = a * sig * np.sqrt(2 * np.pi) * erf(half_width / (np.sqrt(2) * sig))
    assert m.integrate_FWHM(a, mu, sig) == pytest.approx(expected, rel=1e-8)


def test_gaussian2d_peak_offset_and_flattening():
    x, y = np.meshgrid(np.linspace(-3, 3, 7), np.linspace(-2, 2, 5))
    out = m.gaussian2d((x, y), 5.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0)
    assert out.shape == (35,)
    assert out.max() == pytest.approx(6.0)
    assert out.reshape(x.shape)[2, 3] == pytest.approx(6.0)


def test_gaussian2d_quarter_turn_swaps_sigmas():
    x, y = np.meshgrid(np.linspace(-3, 3, 13), np.linspace(-3, 3, 13))
    plain = m.gaussian2d((x, y), 1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0)
    turned = m.gaussian2d((x, y), 1.0, 0.0, 0.0, 2.0, 1.0, np.pi / 2, 0.0)
    np.testing.assert_allclose(turned, plain, atol=1e-12)


def test_integrate_fwhm_2d_of_circular_gaussian_is_separable():
    sigma = 1.0
    half_width = m.FWHM_COEFFICIENT * sigma / 2
    one_d = sigma * np.sqrt(2 * np.pi) * erf(half_width / (np.sqrt(2) * sigma))
    result = m.integrate_fwhm_2d(1.0, 0.0, 0.0, sigma, sigma, 0.0, 0.0)
    assert result == pytest.approx(one_d**2, rel=1e-6)


def test_get_wavelength():
    assert m.get_wavelength(12.398419843320025) == pytest.approx(1.0)
    assert m.get_wavelength(10.0) == pytest.approx(1.2398419843320025)


def test_non_outlier_indices_percentile():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    mask = m.non_outlier_indices_percentile(arr, 25, 75)
    np.testing.assert_array_equal(mask, [True, True, True, True, True, False])
