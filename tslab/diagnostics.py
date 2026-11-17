"""Residual diagnostics and forecast intervals built directly on
`scipy.stats`, rather than statsmodels' equivalents (which exist and are
used elsewhere in this project), kept as a separate module specifically
to exercise scipy directly for the pieces where that's the natural tool.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class NormalityResult:
    jarque_bera_statistic: float
    jarque_bera_pvalue: float
    shapiro_statistic: float
    shapiro_pvalue: float
    looks_normal_at_5pct: bool  # both tests fail to reject normality


def check_residual_normality(residuals: np.ndarray) -> NormalityResult:
    """Two independent normality tests on the same residuals: Jarque-Bera
    (skewness + kurtosis based, the standard econometrics choice) and
    Shapiro-Wilk (generally more powerful for smaller samples, the
    standard statistics choice). Requiring both to fail to reject H0
    before calling residuals "normal" is a deliberately conservative bar, OLS/ARIMA standard errors lean on this assumption, so it's worth more
    than one look before trusting it.
    """
    residuals = np.asarray(residuals, dtype=float)
    jb_stat, jb_p = stats.jarque_bera(residuals)
    sw_stat, sw_p = stats.shapiro(residuals)
    return NormalityResult(
        jarque_bera_statistic=float(jb_stat),
        jarque_bera_pvalue=float(jb_p),
        shapiro_statistic=float(sw_stat),
        shapiro_pvalue=float(sw_p),
        looks_normal_at_5pct=(jb_p > 0.05) and (sw_p > 0.05),
    )


def forecast_interval(point_forecast: np.ndarray, forecast_se: np.ndarray, dof: int, alpha: float = 0.05):
    """(lower, upper) prediction bands using the Student-t critical value
    at `dof` degrees of freedom (via `scipy.stats.t.ppf`) rather than a
    fixed z=1.96, matters when `dof` is small, which it is here (this
    project's holdout sets and residual samples are a few hundred points at
    most, not the asymptotic regime where t and normal critical values are
    indistinguishable).
    """
    point_forecast = np.asarray(point_forecast, dtype=float)
    forecast_se = np.asarray(forecast_se, dtype=float)
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=dof)
    lower = point_forecast - t_crit * forecast_se
    upper = point_forecast + t_crit * forecast_se
    return lower, upper
