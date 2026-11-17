import numpy as np
import pytest

from tslab.diagnostics import check_residual_normality, forecast_interval


def test_gaussian_noise_looks_normal():
    # seed=0 happens to land both tests just under 0.05 (a genuine, rare
    # Type-I error for a truly Gaussian sample at n=500, checked across
    # seeds 0-19, only seed 0 fails at the 5% level for both tests
    # simultaneously). seed=1 is a representative draw.
    rng = np.random.default_rng(1)
    residuals = rng.normal(0, 1, 500)
    result = check_residual_normality(residuals)
    assert result.looks_normal_at_5pct
    assert result.jarque_bera_pvalue > 0.05
    assert result.shapiro_pvalue > 0.05


def test_heavily_skewed_data_does_not_look_normal():
    rng = np.random.default_rng(0)
    residuals = rng.exponential(scale=1.0, size=500)  # strongly right-skewed
    result = check_residual_normality(residuals)
    assert not result.looks_normal_at_5pct
    assert result.jarque_bera_pvalue < 0.05


def test_forecast_interval_widens_with_smaller_dof():
    point = np.array([1.0, 2.0, 3.0])
    se = np.array([0.1, 0.1, 0.1])
    lower_large_dof, upper_large_dof = forecast_interval(point, se, dof=1000)
    lower_small_dof, upper_small_dof = forecast_interval(point, se, dof=3)

    width_large = upper_large_dof - lower_large_dof
    width_small = upper_small_dof - lower_small_dof
    assert np.all(width_small > width_large)  # fatter t-tails at small dof


def test_forecast_interval_is_symmetric_around_point_forecast():
    point = np.array([5.0])
    se = np.array([0.5])
    lower, upper = forecast_interval(point, se, dof=50)
    assert (point - lower) == pytest.approx(upper - point)
