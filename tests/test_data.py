import datetime as dt

import pytest

from tslab import load_macrodata, time_ordered_split


def test_load_macrodata_has_expected_shape_and_index():
    df = load_macrodata()
    assert len(df) == 203
    assert df.index.is_monotonic_increasing
    assert df.index[0] == dt.datetime(1959, 1, 1)
    assert "tbilrate" in df.columns


def test_time_ordered_split_respects_chronology():
    df = load_macrodata()
    split = time_ordered_split(df, "tbilrate", ["infl", "unemp"], test_quarters=20)
    assert len(split.test) == 20
    assert len(split.train) == len(df) - 20
    assert split.train.index.max() < split.test.index.min()


def test_time_ordered_split_rejects_out_of_range_test_size():
    df = load_macrodata()
    with pytest.raises(ValueError):
        time_ordered_split(df, "tbilrate", ["infl"], test_quarters=0)
    with pytest.raises(ValueError):
        time_ordered_split(df, "tbilrate", ["infl"], test_quarters=len(df))
