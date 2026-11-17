"""Renders every chart from results/*.csv (written by run_analysis.py).

Run after scripts/run_analysis.py:
    python3 scripts/plot_results.py
"""
import pathlib

import matplotlib
import matplotlib.dates as mdates

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tslab import grid_search_arima, load_macrodata, time_ordered_split
from tslab.arima_models import acf, pacf

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_PRIMARY, "text.color": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
    "font.size": 11, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
})


def plot_series_with_split():
    df = load_macrodata()
    split = time_ordered_split(df, "tbilrate", ["infl", "unemp"], test_quarters=20)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(split.train.index, split.train.to_numpy(), color=BLUE, linewidth=1.6, label="Train (1959Q1-2004Q3)")
    ax.plot(split.test.index, split.test.to_numpy(), color=ORANGE, linewidth=1.6, label="Test / holdout (2004Q4-2009Q3)")
    ax.axvline(split.test.index[0], color=TEXT_SECONDARY, linewidth=1, linestyle="--", alpha=0.6)
    ax.set_ylabel("3-month T-bill rate (%)")
    ax.set_title("3-month Treasury bill rate, 1959-2009 (real data, statsmodels macrodata)", loc="left", fontsize=12)
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "series_train_test.png", dpi=160)
    plt.close(fig)


def plot_acf_pacf():
    df = load_macrodata()
    split = time_ordered_split(df, "tbilrate", ["infl", "unemp"], test_quarters=20)
    diffed = split.train.diff().dropna()

    nlags = 16
    acf_vals = acf(diffed, nlags=nlags, fft=True)
    pacf_vals = pacf(diffed, nlags=nlags)
    ci = 1.96 / np.sqrt(len(diffed))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, vals, title in [(axes[0], acf_vals, "ACF"), (axes[1], pacf_vals, "PACF")]:
        lags = np.arange(len(vals))
        ax.bar(lags[1:], vals[1:], color=BLUE, width=0.5)
        ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
        ax.axhspan(-ci, ci, color=GRID, alpha=0.6, zorder=0)
        ax.set_title(f"{title} of first-differenced train series", loc="left", fontsize=11)
        ax.set_xlabel("Lag (quarters)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Order-selection diagnostics (informs the ARIMA(p,1,q) search grid)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "acf_pacf.png", dpi=160)
    plt.close(fig)


def plot_aic_heatmap():
    df = load_macrodata()
    split = time_ordered_split(df, "tbilrate", ["infl", "unemp"], test_quarters=20)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        candidates = grid_search_arima(split.train, d=1, p_range=range(0, 5), q_range=range(0, 5))

    grid = np.full((5, 5), np.nan)
    for c in candidates:
        p, _, q = c.order
        grid[p, q] = c.aic

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    im = ax.imshow(grid, cmap=cmap, aspect="auto")
    for p in range(5):
        for q in range(5):
            if not np.isnan(grid[p, q]):
                is_best = (p, 1, q) == min(candidates, key=lambda c: c.aic).order
                ax.text(q, p, f"{grid[p, q]:.0f}", ha="center", va="center",
                        color="white" if grid[p, q] < np.nanmedian(grid) else TEXT_PRIMARY,
                        fontsize=9, fontweight="bold" if is_best else "normal")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xlabel("q (MA order)"); ax.set_ylabel("p (AR order)")
    ax.set_title("AIC across ARIMA(p,1,q) grid - lower is better (darker)", loc="left", fontsize=12)
    fig.colorbar(im, ax=ax, label="AIC", shrink=0.85)
    fig.tight_layout()
    fig.savefig(OUT / "aic_heatmap.png", dpi=160)
    plt.close(fig)


def plot_forecast_comparison():
    fc = pd.read_csv(OUT / "forecast_comparison.csv", parse_dates=["date"])
    df = load_macrodata()
    context = df["tbilrate"].loc["2000-01-01":fc["date"].iloc[0]]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(context.index, context.to_numpy(), color=TEXT_SECONDARY, linewidth=1.4, label="History (pre-holdout)")
    ax.plot(fc["date"], fc["actual"], color=TEXT_PRIMARY, linewidth=2, marker="o", markersize=4, label="Actual")
    ax.plot(fc["date"], fc["linear_forecast"], color=BLUE, linewidth=1.8, marker="o", markersize=4,
            markerfacecolor=SURFACE, label="Linear (recursive, best AIC)")
    ax.plot(fc["date"], fc["arima_forecast"], color=ORANGE, linewidth=1.8, marker="o", markersize=4,
            markerfacecolor=SURFACE, label="ARIMA(2,1,2)")
    ax.fill_between(fc["date"], fc["arima_ci_low_statsmodels"], fc["arima_ci_high_statsmodels"],
                     color=ORANGE, alpha=0.15, label="ARIMA 95% CI")
    ax.plot(fc["date"], fc["naive_forecast"], color=AQUA, linewidth=1.4, linestyle="--", label="Naive (last value)")
    ax.axvline(fc["date"].iloc[0], color=TEXT_SECONDARY, linewidth=1, linestyle=":", alpha=0.6)
    ax.set_ylabel("3-month T-bill rate (%)")
    ax.set_title("Out-of-sample forecasts vs. actual, 2004Q4-2009Q3 (includes the 2008 rate collapse)",
                 loc="left", fontsize=12)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "forecast_comparison.png", dpi=160)
    plt.close(fig)


def plot_oos_metrics_bar():
    oos = pd.read_csv(OUT / "oos_metrics.csv")
    labels = {"linear_recursive": "Linear\n(recursive)", "arima": "ARIMA(2,1,2)", "naive_last_value": "Naive\n(last value)"}
    colors = [BLUE, ORANGE, AQUA]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs = np.arange(len(oos))
    bars = ax.bar(xs, oos["rmse"], color=colors, width=0.55)
    for x, v in zip(xs, oos["rmse"]):
        ax.text(x, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10, color=TEXT_PRIMARY)
    ax.set_xticks(xs)
    ax.set_xticklabels([labels[m] for m in oos["model"]])
    ax.set_ylabel("Out-of-sample RMSE (percentage points)")
    ax.set_title("20-quarter holdout RMSE by model - lower is better", loc="left", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "oos_rmse_comparison.png", dpi=160)
    plt.close(fig)


def plot_residuals():
    df = load_macrodata()
    split = time_ordered_split(df, "tbilrate", ["infl", "unemp"], test_quarters=20)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        candidates = grid_search_arima(split.train, d=1, p_range=range(0, 5), q_range=range(0, 5))
    from tslab import select_by_aic
    best_arima = select_by_aic(candidates, delta=2.0)
    arima_resid = pd.Series(best_arima.results.resid, index=split.train.index)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(arima_resid.index, arima_resid.to_numpy(), color=BLUE, linewidth=1.2)
    axes[0].axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    axes[0].set_title(f"ARIMA{best_arima.order} residuals over time", loc="left", fontsize=11)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].hist(arima_resid.to_numpy(), bins=25, color=BLUE, alpha=0.85)
    axes[1].set_title("Residual distribution", loc="left", fontsize=11)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Residual diagnostics for the selected ARIMA model", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "residual_diagnostics.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    plot_series_with_split()
    plot_acf_pacf()
    plot_aic_heatmap()
    plot_forecast_comparison()
    plot_oos_metrics_bar()
    plot_residuals()
    for name in ["series_train_test", "acf_pacf", "aic_heatmap", "forecast_comparison", "oos_rmse_comparison", "residual_diagnostics"]:
        print(f"wrote {OUT / (name + '.png')}")
