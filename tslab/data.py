"""Loads the real US macroeconomic panel bundled with statsmodels
(Board of Governors / BEA / BLS data, 1959Q1-2009Q3, no network access
needed) and builds a proper datetime-indexed series for the 3-month
Treasury bill rate, the target series for this project.

Why tbilrate: it's the one series in this panel a rates-desk quant would
actually be asked to model, and it has real regime structure (the Volcker
disinflation spike in the early 1980s, the 2008 zero-rate collapse) that
gives both the linear and ARIMA candidates real work to do, a smoother
series would make the model comparison less informative.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm


def load_macrodata() -> pd.DataFrame:
    """Real quarterly US macro data, indexed by an actual datetime (quarter
    start date), built from the raw integer year/quarter columns via
    `datetime.date`, not pandas' PeriodIndex shortcut, so the date
    construction itself is explicit.
    """
    raw = sm.datasets.macrodata.load_pandas().data.copy()

    dates = [
        dt.date(int(year), int((quarter - 1) * 3 + 1), 1)
        for year, quarter in zip(raw["year"], raw["quarter"])
    ]
    raw.index = pd.DatetimeIndex(dates, name="date")
    raw.index.freq = "QS-JAN"
    return raw.drop(columns=["year", "quarter"])


@dataclass
class TrainTestSplit:
    train: pd.Series
    test: pd.Series
    train_exog: pd.DataFrame
    test_exog: pd.DataFrame
    split_date: dt.date


def time_ordered_split(df: pd.DataFrame, target: str, exog_cols: list[str], test_quarters: int) -> TrainTestSplit:
    """Holds out the last `test_quarters` observations as an out-of-sample
    test set. Time-ordered on purpose, a random shuffle split (sklearn's
    default `train_test_split`) would leak future information into the
    training set for a time series and is a textbook mistake this project
    is deliberately avoiding.
    """
    if test_quarters <= 0 or test_quarters >= len(df):
        raise ValueError("test_quarters must be in (0, len(df))")

    split_idx = len(df) - test_quarters
    split_date = df.index[split_idx].date()

    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    return TrainTestSplit(
        train=train[target],
        test=test[target],
        train_exog=train[exog_cols],
        test_exog=test[exog_cols],
        split_date=split_date,
    )
