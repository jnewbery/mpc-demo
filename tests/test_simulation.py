import pathlib
import numpy as np
import pytest

from src.simulation import (
    SimulationParams,
    get_forecast_window,
)

_DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


def _prices(scenario: int = 1, rows: int | None = None) -> np.ndarray:
    arr = np.genfromtxt(
        _DATA_DIR / "price_scenarios" / f"scenario_{scenario}.csv",
        delimiter=",", skip_header=1, usecols=(2,),
    )
    return arr if rows is None else arr[:rows]

def _seasonal_prices(scenario: int = 1, rows: int | None = None) -> np.ndarray:
    arr = np.genfromtxt(
        _DATA_DIR / "price_scenarios" / f"scenario_{scenario}.csv",
        delimiter=",", skip_header=1, usecols=(1,),  # 'seasonal' column
    )
    return arr if rows is None else arr[:rows]


def test_prices_non_negative():
    assert np.all(_prices() >= 0), "Prices contain negative values"


def test_prices_plausible_range():
    prices = _prices()
    assert prices.mean() > 0, "Mean price should be positive"
    assert prices.max() < 1000, "Price spike seems unrealistically large"


def test_prices_shape():
    assert _prices().shape == (365,)


def test_seasonal_winter_peak():
    """Prices and demand should be higher in winter (days 0–30) than summer (days 180–210)."""
    prices = np.genfromtxt(
        _DATA_DIR / "price_scenarios" / "scenario_1.csv",
        delimiter=",", skip_header=1, usecols=(1,),  # 'seasonal' column
    )
    demand = np.genfromtxt(
        _DATA_DIR / "heat_demand_scenarios" / "scenario_1.csv",
        delimiter=",", skip_header=1, usecols=(3,),  # 'seasonal_heat_demand' column
    )
    assert prices[:31].mean() > prices[180:211].mean(), \
        "Seasonal prices should peak in winter"
    assert demand[:31].mean() > demand[180:211].mean(), \
        "Seasonal demand should peak in winter"


def test_forecast_matches_true_price_at_short_horizons():
    """When forecast_short_term covers the full window, the blended forecast
    should equal the true price (α=1 everywhere, deviation = true deviation)."""
    T = 50
    params = SimulationParams(seed=7, T=T, forecast_short_term=T, forecast_long_term=T + 1)
    prices = _prices(rows=T)
    seasonal_prices = _seasonal_prices(rows=T)
    forecast = get_forecast_window(0, T, prices, seasonal_prices, params)
    np.testing.assert_allclose(forecast, prices, rtol=1e-6,
        err_msg="Forecast should match true prices when short-term horizon covers full window")


def test_forecast_equals_seasonal_at_long_horizons():
    """When forecast_long_term=1, α=0 for all h≥1, so the forecast from h=1
    onward should equal the seasonal average."""
    T = 60
    params = SimulationParams(seed=3, T=T, forecast_short_term=0, forecast_long_term=1)
    prices = _prices(rows=T)
    seasonal_prices = _seasonal_prices(rows=T)
    forecast = get_forecast_window(0, T, prices, seasonal_prices, params)

    # h=0: α=1 (short-term covers h=0), so forecast[0] = true price[0]
    assert forecast[0] == pytest.approx(prices[0], rel=1e-6)
    # h≥1: α=0, so forecast should equal seasonal
    np.testing.assert_allclose(forecast[1:], seasonal_prices[1:], rtol=1e-6,
        err_msg="Forecast should equal seasonal average when long-term horizon is 1")


def test_different_scenarios_differ():
    assert not np.allclose(_prices(scenario=1), _prices(scenario=2))


def test_loading_same_scenario_twice_is_identical():
    np.testing.assert_array_equal(_prices(scenario=1), _prices(scenario=1))
