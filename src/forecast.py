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


def get_raw_forecast_window(
    t: int,
    true_prices: np.ndarray,
    params: SimulationParams,
) -> np.ndarray:
    """Return the unblended AR(1) forecast from time t to the end of true_prices.

    Uses the same RNG seed as get_forecast_window so both share the same error
    realisation — the raw and blended traces are directly comparable.

    Returns
    -------
    np.ndarray of shape (len(true_prices) - t,)
    """
    H = len(true_prices) - t
    rng = np.random.default_rng(params.seed + 2 + t)
    E = _ar1_errors(H, params.price_ar1_phi, params.noise_scale, rng)
    return true_prices[t:] + E


def get_forecast_window(
    t: int,
    true_prices: np.ndarray,
    seasonal_prices: np.ndarray,
    params: SimulationParams,
) -> np.ndarray:
    """Return the blended forecast from time t to the end of true_prices.

    Steps:
      1. Generate AR(1) errors E[0..H-1] seeded per timestep (E[0]=0).
      2. Raw forecast: F_raw[h] = true_prices[t+h] + E[h].
      3. Blend: F_final[h] = w[h]*F_raw[h] + (1-w[h])*seasonal[t+h],
         where w[h] decays linearly from 1 to 0 over blend_horizon steps.

    Parameters
    ----------
    t               : current time step
    true_prices     : np.ndarray of shape (T,)
    seasonal_prices : seasonal baseline of shape (N,); indexed modulo N
    params          : SimulationParams

    Returns
    -------
    np.ndarray of shape (len(true_prices) - t,)
    """
    H = len(true_prices) - t
    rng = np.random.default_rng(params.seed + 2 + t)
    E = _ar1_errors(H, params.price_ar1_phi, params.noise_scale, rng)
    w = _linear_weights(H, params.blend_horizon)
    horizons = np.arange(H)
    P_mean = seasonal_prices[(t + horizons) % len(seasonal_prices)]
    return w * (true_prices[t:] + E) + (1 - w) * P_mean
