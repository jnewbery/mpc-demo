import numpy as np
import pytest

from src.heat_demand import calibrate, compute_demand


def test_compute_demand_above_limit():
    """Above t_limit, demand equals p_base (hot water baseload only)."""
    temps = np.array([16.0, 20.0, 25.0])
    result = compute_demand(temps, k=1.0, p_base=5.0)
    np.testing.assert_allclose(result, 5.0)


def test_compute_demand_below_limit():
    """Below t_limit, demand = k*(t_base - T) + p_base."""
    temps = np.array([10.0])
    k, p_base, t_base = 1.0, 5.0, 20.0
    expected = k * (t_base - 10.0) + p_base  # = 15.0
    assert compute_demand(temps, k=k, p_base=p_base).item() == pytest.approx(expected)


def test_compute_demand_at_limit():
    """At exactly t_limit, demand should equal p_base (no space heating contribution)."""
    temps = np.array([15.0])  # default t_limit
    assert compute_demand(temps, k=1.0, p_base=5.0).item() == pytest.approx(5.0)


def test_compute_demand_non_negative():
    """Demand must be non-negative across the full temperature range."""
    temps = np.linspace(-20, 40, 200)
    demand = compute_demand(temps, k=0.5, p_base=2.0)
    assert np.all(demand >= 0)


def test_calibrate_round_trip():
    """calibrate followed by compute_demand should reproduce the annual demand."""
    seasonal_temp = np.array([5.0] * 100 + [20.0] * 265)
    annual_demand = 10_000.0
    k, p_base = calibrate(seasonal_temp, annual_demand)
    demand = compute_demand(seasonal_temp, k, p_base)
    assert demand.sum() == pytest.approx(annual_demand, rel=1e-6)


def test_calibrate_hot_water_baseload():
    """p_base should equal hot_water_fraction * annual_demand / 365."""
    _, p_base = calibrate(np.array([0.0] * 365), annual_demand_mwh=3650.0, hot_water_fraction=0.3)
    assert p_base == pytest.approx(0.3 * 3650.0 / 365)


def test_calibrate_varying_hot_water_fraction():
    """Higher hot_water_fraction should raise p_base and lower k."""
    temps = np.array([0.0] * 200 + [20.0] * 165)
    k_lo, pb_lo = calibrate(temps, 5000.0, hot_water_fraction=0.1)
    k_hi, pb_hi = calibrate(temps, 5000.0, hot_water_fraction=0.5)
    assert pb_hi > pb_lo
    assert k_hi < k_lo
