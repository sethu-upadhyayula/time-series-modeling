"""Linear-regression candidates for the time series, each fit three
independent ways, closed-form normal equations (NumPy), scikit-learn's
`LinearRegression`, and `statsmodels.OLS`, so a coding mistake in any one
implementation would show up as a mismatch against the other two rather
than silently producing a wrong "right answer" (tests/test_linear_models.py
checks all three agree to numerical precision). statsmodels' OLS is the
one actually used downstream, since it's the only one of the three that
exposes AIC/BIC directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


def make_design_matrix(
    target: pd.Series,
    lags: tuple[int, ...] = (),
    trend: bool = True,
    exog: pd.DataFrame | None = None,
    trend_index: pd.Index | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Builds a feature matrix from a target series: an optional linear
    time trend, an optional set of autoregressive lags of the target
    itself, and optional exogenous regressors, then aligns everything
    and drops the leading rows that lagging makes NaN.

    `trend_index`: if given, the trend value for each date is its
    *position in this reference index* rather than 0..len(target)-1. This
    matters when `target` is a training-period slice of a longer series, fitting trend on 0..len(train)-1 and then, when forecasting the test
    period, continuing the trend from len(train) onward requires knowing
    where the training slice sits in the full timeline. Pass the full
    series' index here and both the fit and the forecast step
    (`forecast_linear_recursive`) agree on what "trend" means for any date.
    """
    reference = trend_index if trend_index is not None else target.index
    frame = pd.DataFrame(index=target.index)
    if trend:
        frame["trend"] = [float(reference.get_loc(d)) for d in target.index]
    for lag in lags:
        frame[f"lag{lag}"] = target.shift(lag)
    if exog is not None:
        frame = frame.join(exog)

    frame["_y"] = target
    frame = frame.dropna()
    y = frame.pop("_y")
    return frame, y


@dataclass
class OLSCandidate:
    name: str
    n_features: int
    coeffs_numpy: np.ndarray
    coeffs_sklearn: np.ndarray
    coeffs_statsmodels: np.ndarray
    sm_results: object  # statsmodels RegressionResultsWrapper
    r2: float
    adjusted_r2: float
    aic: float
    bic: float
    fitted_values: pd.Series = field(repr=False)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X_design = sm.add_constant(X, has_constant="add")
        preds = self.sm_results.predict(X_design)
        return preds


def fit_ols_three_ways(name: str, X: pd.DataFrame, y: pd.Series) -> OLSCandidate:
    # 1) Closed-form normal equations: beta = (X'X)^-1 X'y, via NumPy.
    X_np = np.column_stack([np.ones(len(X)), X.to_numpy()])
    y_np = y.to_numpy()
    coeffs_numpy, *_ = np.linalg.lstsq(X_np, y_np, rcond=None)

    # 2) scikit-learn.
    sk_model = LinearRegression()
    sk_model.fit(X, y)
    coeffs_sklearn = np.concatenate([[sk_model.intercept_], sk_model.coef_])

    # 3) statsmodels (this is the one whose .aic/.bic/.rsquared_adj we use).
    X_sm = sm.add_constant(X, has_constant="add")
    sm_results = sm.OLS(y, X_sm).fit()
    coeffs_statsmodels = sm_results.params.to_numpy()

    return OLSCandidate(
        name=name,
        n_features=X.shape[1],
        coeffs_numpy=coeffs_numpy,
        coeffs_sklearn=coeffs_sklearn,
        coeffs_statsmodels=coeffs_statsmodels,
        sm_results=sm_results,
        r2=sm_results.rsquared,
        adjusted_r2=sm_results.rsquared_adj,
        aic=sm_results.aic,
        bic=sm_results.bic,
        fitted_values=sm_results.fittedvalues,
    )


def forecast_linear_recursive(
    candidate: OLSCandidate,
    history: pd.Series,
    full_index: pd.Index,
    lags: tuple[int, ...],
    trend: bool,
    forecast_dates: pd.Index,
) -> pd.Series:
    """Genuine multi-step-ahead forecast for an AR-style linear candidate
    (trend + lags of the target only, no exogenous regressors), matching
    how ARIMA's own multi-step forecast works: each step's prediction feeds
    forward as the "known" value for later steps' lag features, rather than
    using the real (held-out) future values.

    This is what makes the linear-vs-ARIMA out-of-sample race fair: a
    one-step lookup that's allowed to see the actual next quarter's rate
    would make the linear model's forecast trivially better than any real
    model could do prospectively.
    """
    extended = history.copy()
    predictions = {}
    for date in forecast_dates:
        row = {}
        if trend:
            row["trend"] = float(full_index.get_loc(date))
        for lag in lags:
            lag_pos = full_index.get_loc(date) - lag
            lag_date = full_index[lag_pos]
            if lag_date not in extended.index:
                raise ValueError(f"no history available for lag date {lag_date} (forecasting {date})")
            row[f"lag{lag}"] = extended.loc[lag_date]

        X_row = pd.DataFrame([row], index=[date])
        pred = float(candidate.predict(X_row).iloc[0])
        predictions[date] = pred
        extended.loc[date] = pred  # feed forward for subsequent lags

    return pd.Series(predictions).reindex(forecast_dates)
