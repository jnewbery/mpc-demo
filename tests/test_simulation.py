import numpy as np
import pytest

from src.simulation import (
    SimulationParams,
    generate_price_series,
    generate_demand_series,
    generate_seasonal_prices,
    generate_seasonal_demand,
    get_forecast_window,
)


def test_prices_non_negative():
    params = SimulationParams(seed=0)
    prices = generate_price_series(params)
    assert np.all(prices >= 0), "Prices contain negative values"


def test_prices_plausible_range():
    params = SimulationParams(seed=0)
    prices = generate_price_series(params)
    assert prices.mean() == pytest.approx(params.price_mean, abs=20), \
        "Annual mean price is far from expected"
    assert prices.max() < 300, "Price spike seems unrealistically large"


def test_prices_shape():
    for T in (30, 100, 365):
        params = SimulationParams(T=T, seed=1)
        assert generate_price_series(params).shape == (T,)


def test_demand_non_negative():
    params = SimulationParams(seed=0)
    demand = generate_demand_series(params)
    assert np.all(demand >= 0), "Demand contains negative values"


def test_demand_plausible_range():
    params = SimulationParams(seed=0)
    demand = generate_demand_series(params)
    assert demand.mean() == pytest.approx(params.demand_mean, abs=15), \
        "Annual mean demand is far from expected"


def test_seasonal_winter_peak():
    """Prices and demand should be higher in winter (days 0–30) than summer (days 180–210)."""
    params = SimulationParams(seed=42, T=365)
    prices = generate_seasonal_prices(params)
    demand = generate_seasonal_demand(params)
    assert prices[:31].mean() > prices[180:211].mean(), \
        "Seasonal prices should peak in winter"
    assert demand[:31].mean() > demand[180:211].mean(), \
        "Seasonal demand should peak in winter"


def test_demand_weekend_dip():
    """Weekend demand should be lower than weekday demand on average."""
    params = SimulationParams(seed=42, T=365)
    demand = generate_demand_series(params)
    t = np.arange(365)
    weekday = demand[t % 7 < 5].mean()
    weekend = demand[t % 7 >= 5].mean()
    assert weekend < weekday, "Weekend demand should be lower than weekday demand"


def test_forecast_matches_true_price_at_short_horizons():
    """When forecast_short_term covers the full window, the blended forecast
    should equal the true price (α=1 everywhere, deviation = true deviation)."""
    T = 50
    params = SimulationParams(seed=7, T=T, forecast_short_term=T, forecast_long_term=T + 1)
    prices = generate_price_series(params)
    forecast = get_forecast_window(0, T, prices, params)
    np.testing.assert_allclose(forecast, prices, rtol=1e-6,
        err_msg="Forecast should match true prices when short-term horizon covers full window")


def test_forecast_equals_seasonal_at_long_horizons():
    """When forecast_long_term=1, α=0 for all h≥1, so the forecast from h=1
    onward should equal the seasonal average."""
    T = 60
    params = SimulationParams(seed=3, T=T, forecast_short_term=0, forecast_long_term=1)
    prices = generate_price_series(params)
    forecast = get_forecast_window(0, T, prices, params)

    from src.simulation import _seasonal_at
    seasonal = _seasonal_at(np.arange(T), params)
    # h=0: α=1 (short-term), so forecast[0] = true price[0]
    assert forecast[0] == pytest.approx(prices[0], rel=1e-6)
    # h≥1: α=0, so forecast should equal seasonal
    np.testing.assert_allclose(forecast[1:], seasonal[1:], rtol=1e-6,
        err_msg="Forecast should equal seasonal average when long-term horizon is 1")


def test_different_seeds_give_different_series():
    p1 = generate_price_series(SimulationParams(seed=1))
    p2 = generate_price_series(SimulationParams(seed=2))
    assert not np.allclose(p1, p2)


def test_same_seed_gives_identical_series():
    p1 = generate_price_series(SimulationParams(seed=99))
    p2 = generate_price_series(SimulationParams(seed=99))
    np.testing.assert_array_equal(p1, p2)


def test_forecast_window_with_explicit_seasonal_array():
    """get_forecast_window accepts an explicit seasonal_array and uses it instead
    of the synthetic formula — results should differ from the default."""
    T = 60
    params = SimulationParams(seed=10, T=T)
    prices = generate_price_series(params)
    # Flat seasonal array offset from the default formula
    seasonal = np.full(T, params.price_mean + 10.0)
    fc_explicit = get_forecast_window(0, 30, prices, params, seasonal_array=seasonal)
    fc_default = get_forecast_window(0, 30, prices, params)
    assert not np.allclose(fc_explicit, fc_default), \
        "Explicit seasonal array should produce a different forecast from the synthetic one"
    assert np.all(fc_explicit >= 0)
