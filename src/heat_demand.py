from __future__ import annotations

import numpy as np


def calibrate(
    seasonal_temp: np.ndarray,
    annual_demand_mwh: float,
    hot_water_fraction: float = 0.2,
    t_base: float = 20.0,
    t_limit: float = 15.0,
) -> tuple[float, float]:
    """Calibrate degree-day model parameters from annual DH demand.

    Returns (k, p_base) where k is in MW/K and p_base is in MWh/day.
    hot_water_fraction of annual demand is assigned to the flat baseload;
    the remainder is attributed to space heating and used to back-calculate k.
    """
    p_base = hot_water_fraction * annual_demand_mwh / 365
    hdd = float(np.where(seasonal_temp < t_limit, t_base - seasonal_temp, 0.0).sum())
    k = (1.0 - hot_water_fraction) * annual_demand_mwh / hdd
    return k, p_base


def compute_demand(
    temps: np.ndarray,
    k: float,
    p_base: float,
    t_base: float = 20.0,
    t_limit: float = 15.0,
) -> np.ndarray:
    """Piecewise-linear degree-day heat demand in MWh/day.

    Q(T) = k * (t_base - T) + p_base  if T < t_limit
    Q(T) = p_base                      otherwise
    """
    space_heating = np.where(temps < t_limit, k * (t_base - temps), 0.0)
    return space_heating + p_base
