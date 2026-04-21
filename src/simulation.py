"""
simulation.py — Synthetic price and heat demand time series generator.

Generates 365 daily steps (one calendar year) with:
- Seasonal energy prices peaking in winter (AR(1) noise)
- Heat demand (MWh/day) with seasonal and weekly (weekend dip) patterns
- Noisy price forecasts with uncertainty growing with horizon
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationParams:
    T: int = 365                        # number of time steps (days)
    # Price parameters
    price_mean: float = 80.0            # £/MWh annual mean
    price_seasonal_amp: float = 30.0    # winter premium amplitude (£/MWh)
    price_ar1_phi: float = 0.85         # AR(1) autocorrelation coefficient
    price_ar1_sigma: float = 8.0        # AR(1) innovation std dev (£/MWh)
    # Demand parameters
    demand_mean: float = 50.0           # MWh/day annual mean heat demand
    demand_seasonal_amp: float = 25.0   # winter peak amplitude (MWh/day)
    demand_weekend_factor: float = 0.80 # weekend demand relative to weekday
    demand_noise_sigma: float = 5.0     # day-to-day Gaussian noise std dev (MWh/day)
    # Forecast parameters
    forecast_sigma_base: float = 5.0    # base forecast error at horizon h=1 (£/MWh)
    forecast_sigma_growth: float = 2.0  # additional sigma per step of horizon (£/MWh)
    seed: int = 42


def generate_price_series(params: SimulationParams) -> np.ndarray:
    """Generate a daily energy price series with seasonal pattern and AR(1) noise.

    t=0 is 1 Jan, so cos() peaks at t=0 → highest prices in winter.

    Returns
    -------
    np.ndarray of shape (T,), prices in £/MWh, clipped to a minimum of 1.0.
    """
    rng = np.random.default_rng(params.seed)
    t = np.arange(params.T)

    seasonal = params.price_mean + params.price_seasonal_amp * np.cos(
        2 * np.pi * t / 365
    )

    # AR(1) noise: ε[t] = φ·ε[t-1] + σ·z[t]
    innovations = rng.standard_normal(params.T) * params.price_ar1_sigma
    noise = np.zeros(params.T)
    for i in range(1, params.T):
        noise[i] = params.price_ar1_phi * noise[i - 1] + innovations[i]

    prices = np.maximum(1.0, seasonal + noise)
    return prices


def generate_demand_series(params: SimulationParams) -> np.ndarray:
    """Generate a daily heat demand series with seasonal and weekly patterns.

    Seasonal component peaks in winter (t=0 = 1 Jan).
    Weekend days (t % 7 in {5, 6}, treating t=0 as Monday) have reduced demand.

    Returns
    -------
    np.ndarray of shape (T,), demand in MWh/day, clipped to a minimum of 0.0.
    """
    rng = np.random.default_rng(params.seed + 1)
    t = np.arange(params.T)

    seasonal = params.demand_mean + params.demand_seasonal_amp * np.cos(
        2 * np.pi * t / 365
    )

    # Weekend multiplier: day-of-week 5=Saturday, 6=Sunday (Mon=0)
    day_of_week = t % 7
    weekend_mask = (day_of_week == 5) | (day_of_week == 6)
    multiplier = np.where(weekend_mask, params.demand_weekend_factor, 1.0)

    noise = rng.standard_normal(params.T) * params.demand_noise_sigma

    demand = np.maximum(0.0, seasonal * multiplier + noise)
    return demand


def generate_price_forecast(
    true_prices: np.ndarray,
    params: SimulationParams,
) -> np.ndarray:
    """Generate a noisy forecast matrix for all time steps and horizons.

    Uncertainty grows linearly with forecast horizon.

    Parameters
    ----------
    true_prices : np.ndarray of shape (T,)
    params : SimulationParams

    Returns
    -------
    forecast : np.ndarray of shape (T, T)
        forecast[t, h] is the price forecast for day t+h, made at day t.
        Values beyond T are filled by repeating the last known price.
    """
    rng = np.random.default_rng(params.seed + 2)
    T = len(true_prices)
    forecast = np.zeros((T, T))

    for t in range(T):
        for h in range(T):
            target = t + h
            sigma = params.forecast_sigma_base + params.forecast_sigma_growth * h
            if target < T:
                true_val = true_prices[target]
            else:
                true_val = true_prices[-1]  # hold last value beyond horizon
            forecast[t, h] = max(1.0, true_val + rng.standard_normal() * sigma)

    return forecast


def get_forecast_window(
    t: int,
    H: int,
    true_prices: np.ndarray,
    params: SimulationParams,
) -> np.ndarray:
    """Return a 1-D forecast window of length H, starting at time t.

    Convenience wrapper around generate_price_forecast for use in the MPC loop.

    Parameters
    ----------
    t : int — current time step
    H : int — MPC horizon length
    true_prices : np.ndarray of shape (T,)
    params : SimulationParams

    Returns
    -------
    np.ndarray of shape (H,)
    """
    rng = np.random.default_rng(params.seed + 2 + t)
    T = len(true_prices)
    window = np.zeros(H)
    for h in range(H):
        target = t + h
        sigma = params.forecast_sigma_base + params.forecast_sigma_growth * h
        true_val = true_prices[target] if target < T else true_prices[-1]
        window[h] = max(1.0, true_val + rng.standard_normal() * sigma)
    return window


if __name__ == "__main__":
    params = SimulationParams()

    prices = generate_price_series(params)
    demand = generate_demand_series(params)

    print("Price series summary:")
    print(f"  min={prices.min():.1f}, max={prices.max():.1f}, mean={prices.mean():.1f} £/MWh")

    print("Demand series summary:")
    print(f"  min={demand.min():.1f}, max={demand.max():.1f}, mean={demand.mean():.1f} MWh")

    # Spot-check: winter (Jan, days 0-30) vs summer (Jul, days 180-210)
    print(f"\nWinter avg price (days 0-30):   {prices[:31].mean():.1f} £/MWh")
    print(f"Summer avg price (days 180-210): {prices[180:211].mean():.1f} £/MWh")
    print(f"Winter avg demand (days 0-30):   {demand[:31].mean():.1f} MWh")
    print(f"Summer avg demand (days 180-210): {demand[180:211].mean():.1f} MWh")

    # Spot-check weekend effect (first two weeks)
    weekday_demand = demand[[t for t in range(14) if t % 7 not in (5, 6)]]
    weekend_demand = demand[[t for t in range(14) if t % 7 in (5, 6)]]
    print(f"\nFirst 2-week weekday avg demand: {weekday_demand.mean():.1f} MWh")
    print(f"First 2-week weekend avg demand: {weekend_demand.mean():.1f} MWh")

    forecast_window = get_forecast_window(0, 30, prices, params)
    print(f"\n30-day price forecast from day 0:")
    print(f"  min={forecast_window.min():.1f}, max={forecast_window.max():.1f}, mean={forecast_window.mean():.1f} £/MWh")
