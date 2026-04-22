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
    demand_ar1_phi: float = 0.90        # AR(1) autocorrelation (weather persistence)
    demand_noise_sigma: float = 5.0     # AR(1) innovation std dev (MWh/day)
    # Forecast parameters
    forecast_short_term: int = 7        # days: forecast matches true price exactly
    forecast_long_term: int = 30        # days: forecast equals seasonal average
    seed: int = 42


def generate_seasonal_prices(params: SimulationParams) -> np.ndarray:
    """Generate the smooth seasonal price baseline (no noise).

    Returns
    -------
    np.ndarray of shape (T,), seasonal prices in £/MWh.
    """
    t = np.arange(params.T)
    return params.price_mean + params.price_seasonal_amp * np.cos(
        2 * np.pi * t / 365
    )


def generate_seasonal_demand(params: SimulationParams) -> np.ndarray:
    """Generate the seasonal demand baseline including the weekend multiplier (no noise).

    Returns
    -------
    np.ndarray of shape (T,), seasonal demand in MWh/day.
    """
    t = np.arange(params.T)
    seasonal = params.demand_mean + params.demand_seasonal_amp * np.cos(
        2 * np.pi * t / 365
    )
    day_of_week = t % 7
    weekend_mask = (day_of_week == 5) | (day_of_week == 6)
    multiplier = np.where(weekend_mask, params.demand_weekend_factor, 1.0)
    return seasonal * multiplier


def generate_price_series(params: SimulationParams) -> np.ndarray:
    """Generate a daily energy price series: seasonal baseline + AR(1) noise.

    t=0 is 1 Jan, so cos() peaks at t=0 → highest prices in winter.

    Returns
    -------
    np.ndarray of shape (T,), prices in £/MWh, clipped to a minimum of 0.0.
    """
    rng = np.random.default_rng(params.seed)
    seasonal = generate_seasonal_prices(params)

    # AR(1) noise: ε[t] = φ·ε[t-1] + σ·z[t]
    innovations = rng.standard_normal(params.T) * params.price_ar1_sigma
    noise = np.zeros(params.T)
    for i in range(1, params.T):
        noise[i] = params.price_ar1_phi * noise[i - 1] + innovations[i]

    prices = np.maximum(0.0, seasonal + noise)
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

    # AR(1) noise: ε[t] = φ·ε[t-1] + σ·z[t]
    innovations = rng.standard_normal(params.T) * params.demand_noise_sigma
    noise = np.zeros(params.T)
    for i in range(1, params.T):
        noise[i] = params.demand_ar1_phi * noise[i - 1] + innovations[i]

    demand = np.maximum(0.0, seasonal * multiplier + noise)
    return demand


def _seasonal_at(day_indices: np.ndarray, params: SimulationParams) -> np.ndarray:
    """Seasonal baseline for arbitrary day indices (may extend beyond T)."""
    return params.price_mean + params.price_seasonal_amp * np.cos(
        2 * np.pi * day_indices / 365
    )


def _forecast_alpha(horizons: np.ndarray, params: SimulationParams) -> np.ndarray:
    """Blending weight α(h): 1 = perfect forecast, 0 = seasonal only.

    - h ≤ short_term:  α = 1  (forecast matches true price)
    - h ≥ long_term:   α = 0  (forecast equals seasonal average)
    - in between:       smooth cosine ramp from 1 → 0
    """
    s = params.forecast_short_term
    l = params.forecast_long_term
    h = np.asarray(horizons, dtype=float)
    alpha = np.ones_like(h)
    mid = (h > s) & (h < l)
    alpha[mid] = 0.5 * (1 + np.cos(np.pi * (h[mid] - s) / (l - s)))
    alpha[h >= l] = 0.0
    return alpha


def forecast_sigma(horizons: np.ndarray, params: SimulationParams) -> np.ndarray:
    """Expected spread (1σ) of forecast predictions around the seasonal baseline.

    σ(h) = σ_stationary × α(h)

    At short horizons (α=1) the forecast can deviate as widely as the true
    price does (±σ_stationary). At long horizons (α→0) it converges to seasonal.
    """
    phi = params.price_ar1_phi
    sigma_stationary = params.price_ar1_sigma / np.sqrt(1 - phi ** 2)
    return sigma_stationary * _forecast_alpha(horizons, params)


def _forecast_deviation(
    current_deviation: float,
    T: int,
    params: SimulationParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a forward forecast deviation series (random walk from current state).

    Starts at current_deviation (h=0, exact knowledge) and random-walks with
    per-step sigma calibrated so std reaches σ_stationary by forecast_long_term.
    This allows the forecast to drift further from seasonal than the true price.
    """
    phi = params.price_ar1_phi
    sigma_stationary = params.price_ar1_sigma / np.sqrt(1 - phi ** 2)
    sigma_step = sigma_stationary / np.sqrt(max(1, params.forecast_long_term))

    dev = np.empty(T)
    dev[0] = current_deviation
    for h in range(1, T):
        dev[h] = dev[h - 1] + rng.standard_normal() * sigma_step
    return dev


def generate_price_forecast(
    true_prices: np.ndarray,
    params: SimulationParams,
) -> np.ndarray:
    """Generate a forecast matrix using a blended random-walk model.

    At each time t, a forecast deviation series is generated:
      - Starts at true_prices[t] − seasonal[t]  (exact knowledge at h=0)
      - Performs an independent random walk forward (can drift past true price)

    The prediction at horizon h is then:
      prediction[t, h] = seasonal[t+h] + α(h) × forecast_dev[h]

    where α(h) = 1 at short horizons (forecast weighted heavily) and 0 at long
    horizons (prediction equals seasonal average).

    Parameters
    ----------
    true_prices : np.ndarray of shape (T,)
    params : SimulationParams

    Returns
    -------
    forecast : np.ndarray of shape (T, T)
        forecast[t, h] is the price prediction for day t+h, made at day t.
    """
    T = len(true_prices)
    seasonal = generate_seasonal_prices(params)
    horizons = np.arange(T)
    alpha = _forecast_alpha(horizons, params)
    rng = np.random.default_rng(params.seed + 2)

    forecast = np.zeros((T, T))
    for t in range(T):
        current_dev = true_prices[t] - seasonal[t]
        dev = _forecast_deviation(current_dev, T, params, rng)
        s = _seasonal_at(t + horizons, params)
        forecast[t, :] = s + alpha * dev

    return forecast


def get_forecast_window(
    t: int,
    H: int,
    true_prices: np.ndarray,
    params: SimulationParams,
) -> np.ndarray:
    """Return a 1-D forecast window of length H starting at time t.

    Uses the same blended random-walk model as generate_price_forecast.
    Seeded per-timestep so each MPC window gets a consistent forecast.

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
    seasonal_t = _seasonal_at(np.array([t]), params)[0]
    current_dev = true_prices[t] - seasonal_t
    rng = np.random.default_rng(params.seed + 2 + t)
    dev = _forecast_deviation(current_dev, H, params, rng)
    horizons = np.arange(H)
    alpha = _forecast_alpha(horizons, params)
    s = _seasonal_at(t + horizons, params)
    return s + alpha * dev


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
