"""
forecast.py — Simulate a controller's forecasts of future prices and heat demand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationParams:
    T: int = 365                    # number of time steps (days)
    price_ar1_phi: float = 0.85     # AR(1) autocorrelation coefficient ρ
    noise_scale: float = 8.0        # AR(1) innovation std dev σ (€/MWh)
    blend_horizon: int = 30         # steps over which forecast blends to seasonal mean
    seed: int = 42


def _ar1_errors(H: int, rho: float, noise_scale: float, rng: np.random.Generator) -> np.ndarray:
    """AR(1) error chain initialised at zero.

    E[0] = 0 (no error at the current step — controller knows current price).
    E[h] = rho * E[h-1] + N(0, noise_scale²)  for h ≥ 1.
    """
    E = np.zeros(H)
    for h in range(1, H):
        E[h] = rho * E[h - 1] + rng.normal(0.0, noise_scale)
    return E


def _linear_weights(H: int, blend_horizon: int) -> np.ndarray:
    """w[h] = max(0, 1 − h/blend_horizon): 1 at h=0, 0 at h≥blend_horizon."""
    if blend_horizon <= 0:
        return np.zeros(H)
    return np.clip(1.0 - np.arange(H, dtype=float) / blend_horizon, 0.0, 1.0)


def generate_raw_price_forecast(
    true_prices: np.ndarray,
    params: SimulationParams,
    seasonal_prices: np.ndarray,
) -> np.ndarray:
    """Return the unblended AR(1) forecast from t=0 over the full horizon.

    F_raw[h] = true_prices[h] + E[h], where E is an AR(1) error chain.
    No mean-reversion blending is applied; this shows the raw noisy forecast.

    Returns
    -------
    np.ndarray of shape (T,)
    """
    T = len(true_prices)
    rng = np.random.default_rng(params.seed + 2)
    E = _ar1_errors(T, params.price_ar1_phi, params.noise_scale, rng)
    return true_prices + E


def generate_price_forecast(
    true_prices: np.ndarray,
    params: SimulationParams,
    seasonal_prices: np.ndarray,
) -> np.ndarray:
    """Generate a blended forecast matrix using AR(1) errors + linear mean reversion.

    At each time t:
      F_raw[h]   = true_prices[t+h] + E[h]           (AR(1) errors from h=0)
      F_final[h] = w[h] * F_raw[h] + (1-w[h]) * S[t+h]   (blend toward seasonal)

    where w[h] decays linearly from 1 at h=0 to 0 at h=blend_horizon.

    Parameters
    ----------
    true_prices    : np.ndarray of shape (T,)
    params         : SimulationParams
    seasonal_prices: shape-(N,) seasonal reference; indexed modulo N

    Returns
    -------
    forecast : np.ndarray of shape (T, T)
        forecast[t, h] is the blended price prediction for day t+h, made at day t.
    """
    T = len(true_prices)
    w = _linear_weights(T, params.blend_horizon)
    horizons = np.arange(T)

    forecast = np.zeros((T, T))
    for t in range(T):
        rng = np.random.default_rng(params.seed + 2 + t)
        E = _ar1_errors(T, params.price_ar1_phi, params.noise_scale, rng)
        P_nominal = true_prices[np.minimum(t + horizons, T - 1)]
        P_mean    = seasonal_prices[(t + horizons) % len(seasonal_prices)]
        forecast[t, :] = w * (P_nominal + E) + (1 - w) * P_mean

    return forecast


def get_forecast_window(
    t: int,
    H: int,
    true_prices: np.ndarray,
    seasonal_prices: np.ndarray,
    params: SimulationParams,
) -> np.ndarray:
    """Return a blended forecast window of length H starting at time t.

    Steps:
      1. Generate AR(1) errors E[0..H-1] seeded per timestep (E[0]=0).
      2. Raw forecast: F_raw[h] = true_prices[t+h] + E[h].
      3. Blend: F_final[h] = w[h]*F_raw[h] + (1-w[h])*seasonal[t+h],
         where w[h] decays linearly from 1 to 0 over blend_horizon steps.

    Parameters
    ----------
    t               : current time step
    H               : forecast window length
    true_prices     : np.ndarray of shape (T,)
    seasonal_prices : seasonal baseline of shape (N,); indexed modulo N
    params          : SimulationParams

    Returns
    -------
    np.ndarray of shape (H,)
    """
    rng = np.random.default_rng(params.seed + 2 + t)
    E = _ar1_errors(H, params.price_ar1_phi, params.noise_scale, rng)
    w = _linear_weights(H, params.blend_horizon)
    horizons = np.arange(H)
    T_full    = len(true_prices)
    P_nominal = true_prices[np.minimum(t + horizons, T_full - 1)]
    P_mean    = seasonal_prices[(t + horizons) % len(seasonal_prices)]
    return w * (P_nominal + E) + (1 - w) * P_mean
