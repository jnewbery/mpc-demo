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
    t: int,
    true_prices: np.ndarray,
    H: int,
    params: SimulationParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a forward forecast deviation series from time t over H steps.

    For h ≤ H_short: exact knowledge — deviation equals true_prices[t+h] − seasonal[t+h].
    For h > H_short: random walk from the last known deviation, with per-step sigma
    calibrated so the std reaches σ_stationary by forecast_long_term.

    This shared deviation series is used by both generate_price_forecast (which applies
    α(h) blending) and generate_raw_price_forecast (α=1 everywhere), guaranteeing that
    the blended forecast always lies between the raw forecast and the seasonal average.
    """
    T_full = len(true_prices)
    H_short = min(params.forecast_short_term, H - 1)
    phi = params.price_ar1_phi
    sigma_stationary = params.price_ar1_sigma / np.sqrt(1 - phi ** 2)
    sigma_step = sigma_stationary / np.sqrt(max(1, params.forecast_long_term))

    dev = np.empty(H)
    for h in range(H_short + 1):
        idx = min(t + h, T_full - 1)
        dev[h] = true_prices[idx] - _seasonal_at(np.array([t + h]), params)[0]
    for h in range(H_short + 1, H):
        dev[h] = dev[h - 1] + rng.standard_normal() * sigma_step
    return dev


def generate_raw_price_forecast(
    true_prices: np.ndarray,
    params: SimulationParams,
) -> np.ndarray:
    """Return the unblended (α=1) forecast from t=0 for all horizons.

    Tracks the true price exactly up to H_short, then random-walks. Because α=1
    everywhere, this is the upper bound: blended forecast ≤ raw forecast when the
    deviation is positive, and ≥ when negative. Seasonal average is the lower/upper
    bound in the opposite direction.

    Returns
    -------
    np.ndarray of shape (T,) — raw forecast from t=0 for all horizons.
    """
    T = len(true_prices)
    rng = np.random.default_rng(params.seed + 2)
    dev = _forecast_deviation(0, true_prices, T, params, rng)
    s = _seasonal_at(np.arange(T), params)
    return s + dev


def generate_price_forecast(
    true_prices: np.ndarray,
    params: SimulationParams,
) -> np.ndarray:
    """Generate a forecast matrix using a blended random-walk model.

    At each time t a shared deviation series is generated (exact for h ≤ H_short,
    random walk beyond). The prediction applies the blending weight α(h):

      prediction[t, h] = seasonal[t+h] + α(h) × dev[h]

    Since α(h) ∈ [0, 1], the blended forecast is guaranteed to lie between the
    raw forecast (α=1) and the seasonal average (α=0) at every horizon.

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
    horizons = np.arange(T)
    alpha = _forecast_alpha(horizons, params)
    rng = np.random.default_rng(params.seed + 2)

    forecast = np.zeros((T, T))
    for t in range(T):
        dev = _forecast_deviation(t, true_prices, T, params, rng)
        s = _seasonal_at(t + horizons, params)
        forecast[t, :] = s + alpha * dev

    return forecast


def get_forecast_window(
    t: int,
    H: int,
    true_prices: np.ndarray,
    params: SimulationParams,
) -> np.ndarray:
    """Return a 1-D blended forecast window of length H starting at time t.

    Uses the same deviation model as generate_price_forecast.
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
    rng = np.random.default_rng(params.seed + 2 + t)
    dev = _forecast_deviation(t, true_prices, H, params, rng)
    horizons = np.arange(H)
    alpha = _forecast_alpha(horizons, params)
    s = _seasonal_at(t + horizons, params)
    return s + alpha * dev


def ar1_fit(year_residuals: list[np.ndarray]) -> tuple[float, float, float]:
    """Estimate AR(1) parameters from per-year residual sequences.

    Consecutive lags are computed within each year only, so year boundaries
    are never crossed in the lag pairs.

    Parameters
    ----------
    year_residuals : list of 1-D arrays, one per year

    Returns
    -------
    (phi, sigma, sigma_stationary)
        phi              — AR(1) autocorrelation (OLS estimate)
        sigma            — innovation std dev
        sigma_stationary — empirical stationary std dev (std of all residuals)
    """
    r0 = np.concatenate([r[:-1] for r in year_residuals])
    r1 = np.concatenate([r[1:] for r in year_residuals])
    phi = float(np.dot(r0, r1) / np.dot(r0, r0))
    sigma = float(np.std(r1 - phi * r0, ddof=1))
    sigma_stationary = float(np.std(np.concatenate(year_residuals), ddof=1))
    return phi, sigma, sigma_stationary


def ar1_simulate(
    phi: float,
    sigma: float,
    n: int = 365,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw a single AR(1) residual path of length n.

    The initial state is drawn from the parametric stationary distribution
    N(0, sigma^2 / (1 - phi^2)).

    Parameters
    ----------
    phi   : AR(1) autocorrelation
    sigma : innovation std dev
    n     : path length (default 365)
    rng   : random generator; a fresh default_rng() is used if None

    Returns
    -------
    np.ndarray of shape (n,)
    """
    if rng is None:
        rng = np.random.default_rng()
    sigma_stat = sigma / np.sqrt(1 - phi ** 2)
    r = np.empty(n)
    r[0] = rng.normal(0, sigma_stat)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + rng.normal(0, sigma)
    return r


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
