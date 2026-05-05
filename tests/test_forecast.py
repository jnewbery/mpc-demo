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


def test_forecast_matches_true_price_at_short_horizons(test_prices: np.ndarray,
                                                       test_seasonal_prices: np.ndarray
                                                       ):
    """When forecast_short_term covers the full window, the blended forecast
    should equal the true price (α=1 everywhere, deviation = true deviation)."""
    T = 50
    params = SimulationParams(seed=7, T=T, forecast_short_term=T, forecast_long_term=T + 1)
    prices = test_prices[:T]
    seasonal_prices = test_seasonal_prices[:T]
    forecast = get_forecast_window(0, T, prices, seasonal_prices, params)
    np.testing.assert_allclose(forecast, prices, rtol=1e-6,
        err_msg="Forecast should match true prices when short-term horizon covers full window")


def test_forecast_equals_seasonal_at_long_horizons(test_prices: np.ndarray,
                                                       test_seasonal_prices: np.ndarray
                                                       ):
    """When forecast_long_term=1, α=0 for all h≥1, so the forecast from h=1
    onward should equal the seasonal average."""
    T = 60
    params = SimulationParams(seed=3, T=T, forecast_short_term=0, forecast_long_term=1)
    prices = test_prices[:T]
    seasonal_prices = test_seasonal_prices[:T]
    forecast = get_forecast_window(0, T, prices, seasonal_prices, params)

    # h=0: α=1 (short-term covers h=0), so forecast[0] = true price[0]
    assert forecast[0] == pytest.approx(prices[0], rel=1e-6)
    # h≥1: α=0, so forecast should equal seasonal
    np.testing.assert_allclose(forecast[1:], seasonal_prices[1:], rtol=1e-6,
        err_msg="Forecast should equal seasonal average when long-term horizon is 1")
