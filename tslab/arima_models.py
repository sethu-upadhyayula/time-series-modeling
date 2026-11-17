"""ARIMA model selection by AIC (Box-Jenkins style): the differencing order
`d` is chosen from an Augmented Dickey-Fuller stationarity test rather than
searched blindly, and `(p, q)` are then grid-searched by AIC at that fixed
`d`.

This split matters, not just for speed: AIC values are only comparable
across models fit on the *same* data. An ARIMA(p, 0, q) model's likelihood
is computed on the level series; ARIMA(p, 1, q)'s is computed on the
differenced series. Those are different data, so their AIC values are not
directly comparable to each other, picking `d` first via a stationarity
test, then treating AIC search as "compare (p, q) at fixed d" avoids that
trap. (See README for how this project also avoids comparing ARIMA AIC to
linear-regression AIC across a differencing boundary.)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, adfuller, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox


@dataclass
class StationarityResult:
    adf_statistic: float
    p_value: float
    is_stationary_at_5pct: bool
    diff_p_value: float | None  # ADF p-value of the first difference, if the level wasn't stationary
    recommended_d: int


def check_stationarity(series: pd.Series, alpha: float = 0.05) -> StationarityResult:
    """Augmented Dickey-Fuller test. If the level series isn't stationary
    at `alpha`, also tests the first difference for transparency, but
    recommends d=1 either way, this project's search space caps at d=1;
    tbilrate doesn't need more, and higher-order differencing on a rate
    series is rarely sensible economically. If even the first difference
    fails to reject the unit-root null, that's reported (diff_p_value) but
    d=1 is used anyway, since d=2 on a rate series would be uninterpretable."""
    stat, pvalue, *_ = adfuller(series.dropna(), autolag="AIC")
    if pvalue < alpha:
        return StationarityResult(stat, pvalue, True, diff_p_value=None, recommended_d=0)

    _, diff_pvalue, *_ = adfuller(series.diff().dropna(), autolag="AIC")
    return StationarityResult(stat, pvalue, False, diff_p_value=diff_pvalue, recommended_d=1)


@dataclass
class ARIMACandidate:
    order: tuple[int, int, int]
    aic: float
    bic: float
    results: object  # statsmodels ARIMAResultsWrapper


def grid_search_arima(
    series: pd.Series, d: int, p_range: range = range(0, 5), q_range: range = range(0, 5)
) -> list[ARIMACandidate]:
    """Fits ARIMA(p, d, q) for every (p, q) in the grid at fixed d, and
    returns every candidate that converged, sorted by AIC ascending (best
    first). Convergence failures and near-singular fits (common at the
    high end of a (4,4) grid on ~180 quarters of data) are skipped rather
    than allowed to crash the search.
    """
    candidates: list[ARIMACandidate] = []
    for p in p_range:
        for q in q_range:
            if p == 0 and q == 0 and d == 0:
                continue  # degenerate (no dynamics at all)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(series, order=(p, d, q))
                    results = model.fit()
                if not np.isfinite(results.aic):
                    continue
                candidates.append(ARIMACandidate(order=(p, d, q), aic=results.aic, bic=results.bic, results=results))
            except Exception:
                continue  # non-convergent order; skip rather than fail the whole search
    return sorted(candidates, key=lambda c: c.aic)


def select_by_aic(candidates: list[ARIMACandidate], delta: float = 2.0) -> ARIMACandidate:
    """Picks the most parsimonious candidate within `delta` AIC of the
    minimum, rather than the strict argmin.

    This matters in practice, not just in theory: on this project's real
    data, ARIMA(4,1,4) and ARIMA(2,1,2) land within 0.03 AIC units of each
    other, but (2,1,2) has 4 fewer parameters on a 183-observation sample
    and a BIC nearly 13 points better. Burnham & Anderson's "delta AIC < 2"
    rule of thumb treats models that close as statistically
    indistinguishable by AIC alone, in which case the tie should go to the
    simpler model, not whichever one happened to shave the last 0.03 off
    the training likelihood.
    """
    if not candidates:
        raise ValueError("no candidates to select from")
    best_aic = candidates[0].aic  # candidates is already sorted ascending by aic
    within_delta = [c for c in candidates if c.aic - best_aic <= delta]
    return min(within_delta, key=lambda c: sum(c.order))


@dataclass
class ResidualDiagnostics:
    ljung_box_pvalue: float  # H0: no autocorrelation left in residuals
    residuals_look_white: bool
    acf_values: np.ndarray
    pacf_values: np.ndarray


def residual_diagnostics(residuals: pd.Series, lags: int = 10) -> ResidualDiagnostics:
    lb = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    pvalue = float(lb["lb_pvalue"].iloc[0])
    return ResidualDiagnostics(
        ljung_box_pvalue=pvalue,
        residuals_look_white=pvalue > 0.05,
        acf_values=acf(residuals, nlags=lags, fft=True),
        pacf_values=pacf(residuals, nlags=lags),
    )
