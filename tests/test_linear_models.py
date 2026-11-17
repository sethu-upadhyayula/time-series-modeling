import numpy as np
import pandas as pd
import pytest

from tslab.linear_models import fit_ols_three_ways, forecast_linear_recursive, make_design_matrix


def synthetic_ar1_series(n=100, phi=0.6, intercept=2.0, noise_std=0.3, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = np.zeros(n)
    values[0] = intercept
    for t in range(1, n):
        values[t] = intercept * (1 - phi) + phi * values[t - 1] + rng.normal(0, noise_std)
    idx = pd.date_range("2000-01-01", periods=n, freq="QS")
    return pd.Series(values, index=idx, name="y")


def test_three_fitting_methods_agree_on_coefficients():
    y = synthetic_ar1_series()
    X, y_aligned = make_design_matrix(y, lags=(1,), trend=False)
    candidate = fit_ols_three_ways("lag1", X, y_aligned)

    assert candidate.coeffs_numpy == pytest.approx(candidate.coeffs_sklearn, abs=1e-8)
    assert candidate.coeffs_numpy == pytest.approx(candidate.coeffs_statsmodels, abs=1e-8)


def test_recovers_known_ar1_coefficient_on_synthetic_data():
    """AR(1) with phi=0.6: a lag-1 regression should recover a slope near
    0.6, a known-ground-truth check, not just an internal-consistency
    one. n=3000 is chosen purely to shrink OLS sampling noise on the slope
    estimate (empirically ~0.05 std at n=300 for this phi/noise
    combination, confirmed by simulation across seeds) to something a
    tight tolerance can rely on, it isn't meant to represent a
    realistic quarterly-economic-series length."""
    y = synthetic_ar1_series(n=3000, phi=0.6, noise_std=0.2, seed=1)
    X, y_aligned = make_design_matrix(y, lags=(1,), trend=False)
    candidate = fit_ols_three_ways("lag1", X, y_aligned)

    slope = candidate.coeffs_statsmodels[1]  # [intercept, lag1_coef]
    assert slope == pytest.approx(0.6, abs=0.05)
    assert candidate.r2 > 0.25  # phi=0.6 with noise_std=0.2 implies R^2 = phi^2 ~ 0.36 in population


def test_make_design_matrix_drops_leading_nans_from_lagging():
    y = synthetic_ar1_series(n=20)
    X, y_aligned = make_design_matrix(y, lags=(1, 4), trend=True)
    assert len(X) == len(y) - 4  # longest lag determines how many rows are dropped
    assert not X.isna().any().any()
    assert len(X) == len(y_aligned)


def test_r2_and_adjusted_r2_are_consistent():
    y = synthetic_ar1_series(n=150, phi=0.7, seed=2)
    X, y_aligned = make_design_matrix(y, lags=(1,), trend=True)
    candidate = fit_ols_three_ways("lag1+trend", X, y_aligned)
    assert 0.0 <= candidate.r2 <= 1.0
    assert candidate.adjusted_r2 <= candidate.r2


def test_forecast_linear_recursive_matches_noiseless_deterministic_recursion():
    """Noise-free AR(1): y_t = 2*(1-0.9) + 0.9*y_{t-1} = 0.2 + 0.9*y_{t-1}.
    Starting away from the fixed point (2.0) so the series actually varies
    over time (starting exactly at the fixed point would make it constant
    and R^2 undefined, 0/0). OLS on noiseless data recovers the
    coefficients exactly, so the recursive multi-step forecast should
    exactly reproduce the true deterministic continuation, a strong,
    exact ground-truth check rather than a statistical one."""
    n_total = 30
    idx = pd.date_range("2000-01-01", periods=n_total, freq="QS")
    values = np.zeros(n_total)
    values[0] = 5.0
    for t in range(1, n_total):
        values[t] = 0.2 + 0.9 * values[t - 1]
    full_series = pd.Series(values, index=idx)

    n_train = 20
    train = full_series.iloc[:n_train]
    test_dates = full_series.index[n_train:]

    X, y = make_design_matrix(train, lags=(1,), trend=False, trend_index=full_series.index)
    candidate = fit_ols_three_ways("lag1", X, y)
    assert candidate.r2 == pytest.approx(1.0, abs=1e-10)  # noiseless -> perfect fit

    forecast = forecast_linear_recursive(
        candidate, history=train, full_index=full_series.index, lags=(1,), trend=False, forecast_dates=test_dates
    )
    true_continuation = full_series.loc[test_dates]
    assert forecast.to_numpy() == pytest.approx(true_continuation.to_numpy(), abs=1e-6)


def test_forecast_linear_recursive_uses_own_predictions_not_future_actuals():
    """If the recursive forecast were (incorrectly) allowed to peek at the
    real held-out values for its lag features, perturbing those held-out
    values wouldn't change a multi-step-ahead forecast that only depends on
    step 1's own lag input. Confirms the forecast is actually recursive."""
    y = synthetic_ar1_series(n=60, phi=0.5, noise_std=0.1, seed=7)
    train = y.iloc[:40]
    test_dates = y.index[40:]

    X, y_aligned = make_design_matrix(train, lags=(1,), trend=False, trend_index=y.index)
    candidate = fit_ols_three_ways("lag1", X, y_aligned)

    forecast = forecast_linear_recursive(
        candidate, history=train, full_index=y.index, lags=(1,), trend=False, forecast_dates=test_dates
    )
    # Step 2's forecast must equal slope*step1_forecast + intercept, i.e.
    # derived from the model's OWN step-1 prediction, not from the real
    # (different) actual value at test_dates[0].
    intercept, slope = candidate.coeffs_statsmodels
    expected_step2 = intercept + slope * forecast.iloc[0]
    assert forecast.iloc[1] == pytest.approx(expected_step2, abs=1e-8)
    assert forecast.iloc[0] != pytest.approx(y.loc[test_dates[0]])  # true value has noise; forecast shouldn't match it exactly
