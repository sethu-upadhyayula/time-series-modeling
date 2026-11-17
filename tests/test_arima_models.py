import numpy as np
import pandas as pd
import pytest

from tslab.arima_models import ARIMACandidate, grid_search_arima, residual_diagnostics, check_stationarity, select_by_aic


def white_noise(n=200, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=n, freq="QS")
    return pd.Series(rng.normal(0, 1, n), index=idx)


def random_walk(n=200, seed=0) -> pd.Series:
    idx = pd.date_range("2000-01-01", periods=n, freq="QS")
    return pd.Series(np.cumsum(white_noise(n, seed).to_numpy()), index=idx)


def ar1_series(n=200, phi=0.7, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = np.zeros(n)
    for t in range(1, n):
        values[t] = phi * values[t - 1] + rng.normal(0, 1)
    idx = pd.date_range("2000-01-01", periods=n, freq="QS")
    return pd.Series(values, index=idx)


def test_white_noise_is_stationary():
    result = check_stationarity(white_noise())
    assert result.is_stationary_at_5pct
    assert result.recommended_d == 0


def test_random_walk_is_not_stationary_but_its_difference_is():
    result = check_stationarity(random_walk())
    assert not result.is_stationary_at_5pct
    assert result.recommended_d == 1
    assert result.diff_p_value is not None
    assert result.diff_p_value < 0.05  # the difference of a random walk is white noise


def test_arima_grid_search_prefers_ar_dynamics_on_ar1_data():
    """Not a tautology: the search is over p in {0,1,2}, q in {0,1} at d=0,
    and only knows what AIC tells it. Data is genuinely AR(1), so an order
    with p>=1 should win over the pure-noise order (0,0,0)."""
    series = ar1_series(n=250, phi=0.7, seed=3)
    candidates = grid_search_arima(series, d=0, p_range=range(0, 3), q_range=range(0, 2))

    assert len(candidates) > 0
    aics = [c.aic for c in candidates]
    assert aics == sorted(aics)  # returned best-first

    best = candidates[0]
    assert best.order[0] >= 1  # picks up the AR(1) structure
    assert all(c.order[1] == 0 for c in candidates)  # respects the fixed d


def test_residual_diagnostics_flags_autocorrelated_residuals_as_not_white():
    strongly_autocorrelated = ar1_series(n=200, phi=0.9, seed=4)
    diag = residual_diagnostics(strongly_autocorrelated, lags=10)
    assert not diag.residuals_look_white
    assert diag.ljung_box_pvalue < 0.05


def test_residual_diagnostics_passes_white_noise():
    diag = residual_diagnostics(white_noise(n=200, seed=5), lags=10)
    assert diag.residuals_look_white
    assert diag.ljung_box_pvalue > 0.05


def _fake_candidate(order, aic):
    return ARIMACandidate(order=order, aic=aic, bic=aic, results=None)


def test_select_by_aic_prefers_parsimony_within_delta():
    # (2,1,2) is 0.03 AIC worse than (4,1,4) but has 4 fewer parameters.
    # Within delta=2.0, parsimony should win.
    candidates = sorted(
        [_fake_candidate((4, 1, 4), 463.53), _fake_candidate((2, 1, 2), 463.56), _fake_candidate((0, 1, 1), 480.0)],
        key=lambda c: c.aic,
    )
    chosen = select_by_aic(candidates, delta=2.0)
    assert chosen.order == (2, 1, 2)


def test_select_by_aic_falls_back_to_strict_argmin_when_no_close_competitor():
    candidates = sorted(
        [_fake_candidate((3, 1, 3), 400.0), _fake_candidate((1, 1, 1), 420.0)], key=lambda c: c.aic
    )
    chosen = select_by_aic(candidates, delta=2.0)
    assert chosen.order == (3, 1, 3)  # no candidate within 2 AIC units, so strict best wins


def test_select_by_aic_rejects_empty_list():
    with pytest.raises(ValueError):
        select_by_aic([])
