import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tslab.metrics import adjusted_r_squared, mae, mape, r_squared, rmse


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    y_true = np.linspace(1.0, 5.0, 50) + rng.normal(0, 0.1, 50)
    y_pred = y_true + rng.normal(0, 0.2, 50)
    return y_true, y_pred


def test_r_squared_matches_sklearn(sample):
    y_true, y_pred = sample
    assert r_squared(y_true, y_pred) == pytest.approx(r2_score(y_true, y_pred), abs=1e-10)


def test_rmse_matches_sklearn(sample):
    y_true, y_pred = sample
    expected = np.sqrt(mean_squared_error(y_true, y_pred))
    assert rmse(y_true, y_pred) == pytest.approx(expected, abs=1e-10)


def test_mae_matches_sklearn(sample):
    y_true, y_pred = sample
    assert mae(y_true, y_pred) == pytest.approx(mean_absolute_error(y_true, y_pred), abs=1e-10)


def test_r_squared_is_one_for_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r_squared(y, y) == pytest.approx(1.0)


def test_adjusted_r_squared_is_lower_than_r_squared_with_features():
    rng = np.random.default_rng(1)
    y_true = np.linspace(0, 10, 30)
    y_pred = y_true + rng.normal(0, 1.0, 30)
    r2 = r_squared(y_true, y_pred)
    adj = adjusted_r_squared(y_true, y_pred, n_features=5)
    assert adj < r2  # adding the features-penalty term must reduce it whenever R^2 < 1


def test_mape_ignores_zero_actuals():
    y_true = np.array([0.0, 10.0, 20.0])
    y_pred = np.array([5.0, 11.0, 18.0])
    # Only indices 1, 2 should count: |1|/10=0.1, |2|/20=0.1 -> mean 0.1
    assert mape(y_true, y_pred) == pytest.approx(0.1)
