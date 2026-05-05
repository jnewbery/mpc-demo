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
    # Forecast parameters
    forecast_short_term: int = 7        # days: forecast matches true price exactly
    forecast_long_term: int = 30        # days: forecast equals seasonal average
    seed: int = 42


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


def _forecast_deviation(
    t: int,
    true_prices: np.ndarray,
    H: int,
    params: SimulationParams,
    rng: np.random.Generator,
    seasonal_array: np.ndarray,
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
        dev[h] = true_prices[idx] - float(seasonal_array[(t + h) % len(seasonal_array)])
    for h in range(H_short + 1, H):
        dev[h] = dev[h - 1] + rng.standard_normal() * sigma_step
    return dev


def generate_raw_price_forecast(
    true_prices: np.ndarray,
    params: SimulationParams,
    seasonal_array: np.ndarray,
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
    dev = _forecast_deviation(0, true_prices, T, params, rng, seasonal_array)
    s = seasonal_array[np.arange(T) % len(seasonal_array)]
    return s + dev


def generate_price_forecast(
    true_prices: np.ndarray,
    params: SimulationParams,
    seasonal_array: np.ndarray,
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
    seasonal_array : shape-(N,) seasonal reference; indexed modulo N for horizons beyond N

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
        dev = _forecast_deviation(t, true_prices, T, params, rng, seasonal_array)
        s = seasonal_array[(t + horizons) % len(seasonal_array)]
        forecast[t, :] = s + alpha * dev

    return forecast


def get_forecast_window(
    t: int,
    H: int,
    true_prices: np.ndarray,
    params: SimulationParams,
    seasonal_array: np.ndarray | None = None,
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
    seasonal_array : optional pre-computed seasonal baseline of shape (T,);
        if None, the synthetic formula from SimulationParams is used.

    Returns
    -------
    np.ndarray of shape (H,)
    """
    rng = np.random.default_rng(params.seed + 2 + t)
    s_arr = seasonal_array if seasonal_array is not None else _seasonal_at(np.arange(len(true_prices)), params)
    dev = _forecast_deviation(t, true_prices, H, params, rng, s_arr)
    horizons = np.arange(H)
    alpha = _forecast_alpha(horizons, params)
    s = s_arr[(t + horizons) % len(s_arr)]
    return s + alpha * dev
