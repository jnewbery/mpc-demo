import numpy as np
import pytest

from src.forecast import (
    SimulationParams,
    get_forecast_window,
)

def test_prices_non_negative(test_prices: np.ndarray):
    assert np.all(test_prices >= 0), "Prices contain negative values"


def test_prices_plausible_range(test_prices: np.ndarray):
    assert test_prices.mean() > 0, "Mean price should be positive"
    assert test_prices.max() < 1000, "Price spike seems unrealistically large"


def test_prices_shape(test_prices: np.ndarray):
    assert test_prices.shape == (365,)


def test_seasonal_winter_peak(test_prices: np.ndarray, test_demand: np.ndarray):
    """Prices and demand should be higher in winter (days 0–30) than summer (days 180–210)."""
    assert test_prices[:31].mean() > test_prices[180:211].mean(), \
        "Seasonal prices should peak in winter"
    assert test_demand[:31].mean() > test_demand[180:211].mean(), \
        "Seasonal demand should peak in winter"


def test_forecast_zero_noise_matches_linear_blend(
        test_prices: np.ndarray, test_seasonal_prices: np.ndarray):
    """With noise_scale=0, E=0 everywhere, so forecast equals the explicit linear blend:
    forecast[h] = w[h]*P_nominal[h] + (1-w[h])*seasonal[h]."""
    T, blend = 50, 20
    params = SimulationParams(seed=7, T=T, noise_scale=0.0, blend_horizon=blend)
    prices = test_prices[:T]
    seasonal = test_seasonal_prices[:T]
    forecast = get_forecast_window(0, prices, seasonal, params)

    h = np.arange(T, dtype=float)
    w = np.clip(1.0 - h / blend, 0.0, 1.0)
    expected = w * prices + (1 - w) * seasonal
    np.testing.assert_allclose(forecast, expected, rtol=1e-6,
        err_msg="Zero-noise forecast should equal the linear blend of nominal and seasonal")


def test_forecast_equals_seasonal_beyond_blend_horizon(
        test_prices: np.ndarray, test_seasonal_prices: np.ndarray):
    """With blend_horizon=1, w=0 for h≥1 so forecast equals seasonal from h=1 onward.
    At h=0: E[0]=0 and w[0]=1, so forecast[0] = true price[0]."""
    T = 60
    params = SimulationParams(seed=3, T=T, blend_horizon=1)
    prices = test_prices[:T]
    seasonal_prices = test_seasonal_prices[:T]
    forecast = get_forecast_window(0, prices, seasonal_prices, params)

    assert forecast[0] == pytest.approx(prices[0], rel=1e-6)
    np.testing.assert_allclose(forecast[1:], seasonal_prices[1:], rtol=1e-6,
        err_msg="Forecast should equal seasonal average when blend_horizon=1")
