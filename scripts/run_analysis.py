"""End-to-end model-selection pipeline: fits a family of linear-regression
candidates and a grid of ARIMA candidates to the 3-month T-bill rate
(1959Q1-2004Q4), selects the best of each family by AIC, forecasts the
held-out last 20 quarters (2005Q1-2009Q3, includes the 2008 crisis rate
collapse, real regime stress for the models to be evaluated against), and
writes every result to results/*.csv.

Run from the project root:
    python3 scripts/run_analysis.py
"""
import pathlib
import warnings

import numpy as np
import pandas as pd

from tslab import (
    check_residual_normality,
    check_stationarity,
    fit_ols_three_ways,
    forecast_interval,
    grid_search_arima,
    load_macrodata,
    make_design_matrix,
    mae,
    mape,
    r_squared,
    residual_diagnostics,
    rmse,
    select_by_aic,
    time_ordered_split,
)
from tslab.linear_models import forecast_linear_recursive

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

TEST_QUARTERS = 20
EXOG_COLS = ["infl", "unemp"]


def build_linear_candidates(split, full_index):
    specs = {
        "L1_trend_only": dict(lags=(), trend=True, exog=None),
        "L2_trend_lag1": dict(lags=(1,), trend=True, exog=None),
        "L3_trend_lag1_lag4": dict(lags=(1, 4), trend=True, exog=None),
        "L4_lag1_infl_unemp": dict(lags=(1,), trend=False, exog=split.train_exog),
        "L5_trend_lag1_infl_unemp": dict(lags=(1,), trend=True, exog=split.train_exog),
    }
    candidates = {}
    for name, spec in specs.items():
        X, y = make_design_matrix(split.train, trend_index=full_index, **spec)
        candidates[name] = (fit_ols_three_ways(name, X, y), spec)
    return candidates


def main():
    print("=" * 70)
    print("Loading data: statsmodels macrodata (real US macro panel, 1959Q1-2009Q3)")
    df = load_macrodata()
    split = time_ordered_split(df, "tbilrate", EXOG_COLS, test_quarters=TEST_QUARTERS)
    print(f"Train: {split.train.index[0].date()} .. {split.train.index[-1].date()}  (n={len(split.train)})")
    print(f"Test:  {split.test.index[0].date()} .. {split.test.index[-1].date()}  (n={len(split.test)})")

    # --- Stationarity -> pick ARIMA's d ---
    stationarity = check_stationarity(split.train)
    print(f"\nADF on level series: stat={stationarity.adf_statistic:.3f}, p={stationarity.p_value:.4f} "
          f"-> {'stationary' if stationarity.is_stationary_at_5pct else 'NOT stationary'} at 5%")
    if stationarity.diff_p_value is not None:
        print(f"ADF on first difference: p={stationarity.diff_p_value:.6f}")
    print(f"-> using d={stationarity.recommended_d} for the ARIMA grid search")

    # --- Linear candidates ---
    print("\nFitting linear-regression candidates (numpy + sklearn + statsmodels, cross-checked)...")
    linear_candidates = build_linear_candidates(split, df.index)

    rows = []
    for name, (cand, spec) in linear_candidates.items():
        rows.append({
            "family": "linear", "model": name, "n_params": cand.n_features,
            "r2": cand.r2, "adjusted_r2": cand.adjusted_r2, "aic": cand.aic, "bic": cand.bic,
        })

    # AIC-best among the FORECASTABLE linear candidates (target-history only,
    # L1-L3), L4/L5 use infl/unemp and can't be fairly recursively
    # forecast without also modeling those exogenous series, so they're
    # reported for discussion but excluded from the OOS race.
    forecastable_names = ["L1_trend_only", "L2_trend_lag1", "L3_trend_lag1_lag4"]
    best_linear_name = min(forecastable_names, key=lambda n: linear_candidates[n][0].aic)
    best_linear, best_linear_spec = linear_candidates[best_linear_name]
    print(f"Best forecastable linear candidate by AIC: {best_linear_name} "
          f"(AIC={best_linear.aic:.2f}, R^2={best_linear.r2:.4f}, Adj R^2={best_linear.adjusted_r2:.4f})")

    best_exog_name = min(["L4_lag1_infl_unemp", "L5_trend_lag1_infl_unemp"], key=lambda n: linear_candidates[n][0].aic)
    best_exog, _ = linear_candidates[best_exog_name]
    print(f"(For reference only, not forecast-eligible: best exog-augmented candidate is {best_exog_name}, "
          f"AIC={best_exog.aic:.2f}, R^2={best_exog.r2:.4f}. More information helps in-sample fit, "
          f"but using it for a real forecast would require also forecasting inflation/unemployment.)")

    # --- ARIMA grid search ---
    print(f"\nGrid-searching ARIMA(p,{stationarity.recommended_d},q) for p,q in 0..4 (25 fits)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arima_candidates = grid_search_arima(split.train, d=stationarity.recommended_d, p_range=range(0, 5), q_range=range(0, 5))
    print(f"{len(arima_candidates)}/25 orders converged.")
    for c in arima_candidates[:5]:
        rows.append({"family": "arima", "model": f"ARIMA{c.order}", "n_params": sum(c.order),
                     "r2": np.nan, "adjusted_r2": np.nan, "aic": c.aic, "bic": c.bic})
    best_arima = select_by_aic(arima_candidates, delta=2.0)
    strict_best_arima = arima_candidates[0]
    print(f"Strict AIC-minimum: ARIMA{strict_best_arima.order}  AIC={strict_best_arima.aic:.2f}")
    print(f"Selected (parsimony-within-2-AIC rule): ARIMA{best_arima.order}  "
          f"AIC={best_arima.aic:.2f}  BIC={best_arima.bic:.2f}")

    candidates_df = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    candidates_df.to_csv(OUT / "candidates_summary.csv", index=False)
    print(f"\nwrote {OUT / 'candidates_summary.csv'} ({len(candidates_df)} rows)")

    # --- In-sample residual diagnostics for both selected models ---
    linear_resid = split.train.loc[best_linear.fitted_values.index] - best_linear.fitted_values
    arima_resid = pd.Series(best_arima.results.resid)

    linear_lb = residual_diagnostics(linear_resid, lags=8)
    arima_lb = residual_diagnostics(arima_resid, lags=8)
    linear_norm = check_residual_normality(linear_resid.to_numpy())
    arima_norm = check_residual_normality(arima_resid.to_numpy())

    print(f"\n{best_linear_name} residuals: Ljung-Box p={linear_lb.ljung_box_pvalue:.4f} "
          f"({'white noise' if linear_lb.residuals_look_white else 'still autocorrelated'}), "
          f"JB p={linear_norm.jarque_bera_pvalue:.4f}, Shapiro p={linear_norm.shapiro_pvalue:.4f}")
    print(f"ARIMA{best_arima.order} residuals: Ljung-Box p={arima_lb.ljung_box_pvalue:.4f} "
          f"({'white noise' if arima_lb.residuals_look_white else 'still autocorrelated'}), "
          f"JB p={arima_norm.jarque_bera_pvalue:.4f}, Shapiro p={arima_norm.shapiro_pvalue:.4f}")

    diagnostics_df = pd.DataFrame([
        {"model": best_linear_name, "ljung_box_pvalue": linear_lb.ljung_box_pvalue,
         "residuals_white": linear_lb.residuals_look_white, "jb_pvalue": linear_norm.jarque_bera_pvalue,
         "shapiro_pvalue": linear_norm.shapiro_pvalue, "looks_normal": linear_norm.looks_normal_at_5pct},
        {"model": f"ARIMA{best_arima.order}", "ljung_box_pvalue": arima_lb.ljung_box_pvalue,
         "residuals_white": arima_lb.residuals_look_white, "jb_pvalue": arima_norm.jarque_bera_pvalue,
         "shapiro_pvalue": arima_norm.shapiro_pvalue, "looks_normal": arima_norm.looks_normal_at_5pct},
    ])
    diagnostics_df.to_csv(OUT / "residual_diagnostics.csv", index=False)
    print(f"wrote {OUT / 'residual_diagnostics.csv'}")

    # --- Out-of-sample forecasts (genuine multi-step-ahead, no peeking) ---
    print(f"\nForecasting {TEST_QUARTERS} quarters out-of-sample...")
    linear_forecast = forecast_linear_recursive(
        best_linear, history=split.train, full_index=df.index,
        lags=best_linear_spec["lags"], trend=best_linear_spec["trend"], forecast_dates=split.test.index,
    )

    arima_fc = best_arima.results.get_forecast(steps=TEST_QUARTERS)
    arima_forecast = pd.Series(arima_fc.predicted_mean.to_numpy(), index=split.test.index)
    arima_se = pd.Series(arima_fc.se_mean.to_numpy(), index=split.test.index)
    sm_ci = arima_fc.conf_int(alpha=0.05)
    # Cross-check statsmodels' own CI against an independent scipy.stats.t
    # band built from the same (mean, se), see diagnostics.forecast_interval.
    scipy_lower, scipy_upper = forecast_interval(
        arima_forecast.to_numpy(), arima_se.to_numpy(), dof=len(split.train) - sum(best_arima.order) - 1
    )

    naive_forecast = pd.Series(split.train.iloc[-1], index=split.test.index)  # last-value-carried-forward benchmark

    forecast_df = pd.DataFrame({
        "date": split.test.index,
        "actual": split.test.to_numpy(),
        "linear_forecast": linear_forecast.to_numpy(),
        "arima_forecast": arima_forecast.to_numpy(),
        "arima_ci_low_statsmodels": sm_ci.iloc[:, 0].to_numpy(),
        "arima_ci_high_statsmodels": sm_ci.iloc[:, 1].to_numpy(),
        "arima_ci_low_scipy": scipy_lower,
        "arima_ci_high_scipy": scipy_upper,
        "naive_forecast": naive_forecast.to_numpy(),
    })
    forecast_df.to_csv(OUT / "forecast_comparison.csv", index=False)
    print(f"wrote {OUT / 'forecast_comparison.csv'}")

    ci_width_diff = np.abs((sm_ci.iloc[:, 1] - sm_ci.iloc[:, 0]).to_numpy() - (scipy_upper - scipy_lower)).mean()
    print(f"Mean |CI width difference| (statsmodels vs. independent scipy.stats.t band): {ci_width_diff:.4f}")

    # --- Out-of-sample accuracy ---
    actual = split.test.to_numpy()
    metrics_rows = []
    for label, pred in [
        ("linear_recursive", linear_forecast.to_numpy()),
        ("arima", arima_forecast.to_numpy()),
        ("naive_last_value", naive_forecast.to_numpy()),
    ]:
        metrics_rows.append({
            "model": label, "rmse": rmse(actual, pred), "mae": mae(actual, pred),
            "mape": mape(actual, pred), "r2_oos": r_squared(actual, pred),
        })
    oos_df = pd.DataFrame(metrics_rows)
    oos_df.to_csv(OUT / "oos_metrics.csv", index=False)
    print(f"\nOut-of-sample accuracy ({TEST_QUARTERS}-quarter holdout):")
    print(oos_df.to_string(index=False))

    linear_rmse = oos_df.loc[oos_df.model == "linear_recursive", "rmse"].iloc[0]
    arima_rmse = oos_df.loc[oos_df.model == "arima", "rmse"].iloc[0]
    naive_rmse = oos_df.loc[oos_df.model == "naive_last_value", "rmse"].iloc[0]

    winner, loser, winner_rmse, loser_rmse = (
        ("ARIMA", "linear", arima_rmse, linear_rmse) if arima_rmse < linear_rmse
        else ("linear", "ARIMA", linear_rmse, arima_rmse)
    )
    improvement_vs_loser = (1 - winner_rmse / loser_rmse) * 100
    improvement_vs_naive = (1 - winner_rmse / naive_rmse) * 100
    print(f"\nWinner: {winner} (RMSE={winner_rmse:.4f}) beats {loser} (RMSE={loser_rmse:.4f}) "
          f"by {improvement_vs_loser:.1f}% lower RMSE")
    print(f"{winner} vs. naive last-value-carried-forward (RMSE={naive_rmse:.4f}): "
          f"{improvement_vs_naive:.1f}% lower RMSE")

    summary_path = OUT / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Best forecastable linear model: {best_linear_name}\n")
        f.write(f"  AIC={best_linear.aic:.3f}  R2={best_linear.r2:.4f}  Adj R2={best_linear.adjusted_r2:.4f}\n")
        f.write(f"Best ARIMA model: ARIMA{best_arima.order}\n")
        f.write(f"  AIC={best_arima.aic:.3f}  BIC={best_arima.bic:.3f}\n")
        f.write(f"Stationarity: ADF p={stationarity.p_value:.4f} on level -> d={stationarity.recommended_d}\n\n")
        f.write(f"Out-of-sample ({TEST_QUARTERS}q holdout, {split.test.index[0].date()}..{split.test.index[-1].date()}):\n")
        f.write(oos_df.to_string(index=False) + "\n\n")
        f.write(f"Winner: {winner} beats {loser} by {improvement_vs_loser:.1f}% lower RMSE\n")
        f.write(f"Winner vs. naive baseline: {improvement_vs_naive:.1f}% lower RMSE\n")
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
