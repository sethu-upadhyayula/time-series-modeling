# tslab: Time Series Model Selection, Linear vs. ARIMA

A tested Python library that fits linear regression and ARIMA models to a macroeconomic time series, selects the best candidate in each family (Adjusted R² for linear models, AIC with a parsimony rule for ARIMA), and compares them out of sample.

Stack: `numpy`, `pandas`, `scipy`, `statsmodels`, `scikit-learn`, `matplotlib`.

## Design principles

A linear vs. ARIMA comparison has several pitfalls that can distort results without raising any errors.

- **AIC is only comparable across models fit to the same data.** An ARIMA(2,1,2) fit on differenced data cannot be ranked against an ARIMA(1,0,1) fit on levels. The library fixes the differencing order `d` first with an ADF stationarity test and then grid-searches `(p, q)` at that `d`, so every AIC in the comparison is on the same footing.
- **The minimum-AIC model is not always the best choice.** Models within about 2 AIC points of each other are statistically indistinguishable (the Burnham and Anderson rule of thumb), and the more parsimonious one usually generalizes better. `select_by_aic` applies this rule explicitly.
- **A fair forecast comparison needs recursive forecasting on both sides.** Feeding the linear model actual future lag values would give it an unfair advantage. The linear model is forecast recursively, with each prediction feeding the next step's lag features, which matches how ARIMA's dynamic forecast works.
- **Every regression is fit three independent ways** (`numpy.linalg.lstsq`, scikit-learn's `LinearRegression` and `statsmodels.OLS`) and the results are checked for agreement. The ARIMA forecast confidence interval is also cross-checked against an independently coded `scipy.stats.t` band.

## Data

`statsmodels.datasets.macrodata` contains US quarterly macroeconomic data from 1959Q1 to 2009Q3 (203 observations). It ships with statsmodels, so the analysis is fully reproducible without internet access. The target variable is `tbilrate`, the 3-month Treasury bill rate. Candidate exogenous features are `infl` (inflation) and `unemp` (unemployment).

The series is split in time order, without shuffling, into a training set (1959Q1 to 2004Q3, 183 quarters) and a 20-quarter holdout (2004Q4 to 2009Q3). The holdout is simply the last 20 quarters of the dataset, and it spans the 2008 financial crisis, which strongly affects the results.

## Method

**Linear regression.** `make_design_matrix` builds candidate feature sets from a linear trend and lags of the target, optionally with exogenous regressors. Each candidate is fit three ways and scored by R², Adjusted R², AIC and BIC. Candidates fall into two groups: forecastable models (trend and own lags only), which can be rolled forward recursively, and exogenous-augmented models, which are kept as a reference upper bound because forecasting `infl` and `unemp` is outside the scope of this project.

**ARIMA.** `check_stationarity` runs an Augmented Dickey-Fuller test on the level series. If it fails to reject the unit-root null at 5%, it recommends `d=1`, capped at 1 because a second difference of a rate series is rarely economically meaningful. `grid_search_arima` then fits every `ARIMA(p,1,q)` for `p, q ∈ {0,...,4}`, and `select_by_aic` picks the most parsimonious model within 2 AIC points of the best.

**Diagnostics.** Ljung-Box tests residual whiteness. Jarque-Bera and Shapiro-Wilk test residual normality, and both must pass at 5% for residuals to be classed as normal.

**Out-of-sample evaluation.** Both models forecast the 20-quarter holdout recursively (dynamic multi-step forecasts, not one-step-ahead) and are scored on RMSE, MAE, MAPE and R², against a naive baseline that carries the last value forward.

## Results

All figures are produced by `scripts/run_analysis.py`.

**Model selection:**

| | Model | AIC | BIC | R² | Adj. R² |
|---|---|---|---|---|---|
| Linear (forecastable) | `trend + lag1 + lag4` | 473.04 | 485.79 | 0.897 | 0.895 |
| Linear (with exogenous, reference) | `trend + lag1 + infl + unemp` | 447.44 | 460.25 | 0.914 | 0.912 |
| ARIMA | (2,1,2) | 463.56 | 479.58 | n/a | n/a |

The ARIMA grid search shows why the parsimony rule matters. The strict AIC minimum was ARIMA(4,1,4) at 463.53, only 0.03 below ARIMA(2,1,2) at 463.56, a statistically meaningless difference. ARIMA(2,1,2) uses 4 fewer parameters and has a much better BIC (479.58 vs. 492.37), so `select_by_aic` chooses it.

**Residual diagnostics:**

| Model | Ljung-Box p (whiteness) | Normal residuals? |
|---|---|---|
| Linear | 0.0001 (fails, still autocorrelated) | No |
| ARIMA(2,1,2) | 0.375 (passes, white noise) | No |

The statsmodels `.conf_int()` interval and the independently coded `scipy.stats.t` band agree closely, with a mean width difference of 0.07 percentage points across the 20-quarter forecast.

**Out-of-sample (20-quarter holdout spanning the 2008 rate collapse):**

| Model | RMSE | MAE | MAPE | R² (OOS) |
|---|---|---|---|---|
| Linear (recursive) | **1.990** | 1.798 | 441% | -0.293 |
| ARIMA(2,1,2) | 2.132 | 1.841 | 233% | -0.484 |
| Naive (last value) | 2.115 | 1.829 | 237% | -0.461 |

The linear model has the lowest RMSE: 6.7% below ARIMA and 5.9% below the naive baseline.

## Discussion

**ARIMA has cleaner in-sample residuals but loses out of sample.** ARIMA(2,1,2) passes the Ljung-Box whiteness test in sample (p=0.375), while the linear model's residuals remain autocorrelated (p=0.0001). By classical diagnostics ARIMA should be the better model, yet it forecasts worse on this holdout. In-sample residual diagnostics and out-of-sample accuracy answer different questions.

**All three models have negative out-of-sample R².** Each does worse than predicting the training mean for every holdout point. This happens because a structural break, the 2008 financial crisis, falls inside the holdout but outside the training distribution. The rate falls from about 5% to about 0.15% in roughly six quarters, a move that has no precedent in the 1959 to 2004 training data. The forecast chart (`results/forecast_comparison.png`) shows both models drifting near the last training value while the actual rate first rises and then collapses.

**MAPE is unreliable for this target.** The linear model's MAPE is 441% because MAPE divides by the actual value, and `tbilrate` approaches zero in 2008 and 2009 (as low as 0.12%). Even small absolute errors then become very large percentage errors. For this reason RMSE is the headline metric.

## Design notes and limitations

- The exogenous-augmented linear model (`trend + lag1 + infl + unemp`, R² of 0.914) fits better in sample than the forecastable model but is excluded from the out-of-sample comparison, since it would require forecasting `infl` and `unemp` as well. It is reported as a reference upper bound only.
- The ARIMA forecast is dynamic: each step conditions on ARIMA's own prior forecasts rather than actual values, matching `forecast_linear_recursive` on the linear side. This keeps the comparison fair, but both forecasts flatten over a long horizon, as the forecast chart shows.
- `check_stationarity` caps the recommended differencing order at `d=1`. A second difference is unlikely to help for a bounded rate series and would make the model harder to interpret.
- The grid search is exhaustive over `p, q ∈ {0,...,4}` (25 fits) rather than a heuristic search such as `pmdarima.auto_arima`, which keeps dependencies light and the selection transparent.

## Project layout

```
tslab/
  data.py            macrodata loader and time-ordered train/test split
  linear_models.py   design matrix builder, 3-way OLS fit, recursive forecasting
  arima_models.py    stationarity test, ARIMA grid search, AIC parsimony selection
  diagnostics.py     normality tests and t-distribution forecast intervals
  metrics.py         R², Adjusted R², RMSE, MAE, MAPE
tests/               27 pytest tests
scripts/
  run_analysis.py    runs the full pipeline and writes results/*.csv
  plot_results.py    renders results/*.png from the CSVs
results/             generated CSVs and PNG charts
```

## Running it

```bash
pip install -e ".[dev]"
python3 -m pytest -q                  # 27 tests
python3 scripts/run_analysis.py       # writes results/*.csv and results/summary.txt
python3 scripts/plot_results.py       # writes results/*.png
```
