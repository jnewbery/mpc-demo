"""
storage_lp.py — LP-based thermal storage optimiser.

Models a heat pump connected to thermal storage. At every time step the heat
pump output must satisfy demand plus net storage change:

    hp_output[t] = demand[t] + u_plus[t] - u_minus[t]

The heat pump converts electricity to heat with coefficient of performance COP,
so electricity cost per day is  prices[t] * hp_output[t] / COP.

Two solvers are provided:
  - solve_perfect_foresight: full T-step LP with true prices and demand
  - solve_single_window:     H-step LP from current SoC (used in MPC loop)
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np


@dataclass
class StorageParams:
    s_max: float = 100.0       # max storage capacity (MWh)
    s_min: float = 0.0         # min SoC (MWh)
    s0: float = 50.0           # initial SoC (MWh)
    u_plus_max: float = 50.0   # max charge rate (MWh/day)
    u_minus_max: float = 50.0  # max discharge rate (MWh/day)
    eta: float = 0.90          # round-trip discharge efficiency
    cop: float = 3.0           # heat pump coefficient of performance
    h_max: float = 100.0       # max heat pump output (MWh/day)


def _build_lp(
    prices: np.ndarray,
    demand: np.ndarray,
    s0: float,
    params: StorageParams,
) -> tuple[cp.Problem, dict[str, cp.Variable]]:
    """Build the LP for an arbitrary horizon length.

    Parameters
    ----------
    prices : (H,) electricity price forecast (£/MWh)
    demand : (H,) heat demand forecast (MWh/day)
    s0 : starting SoC (MWh)
    params : StorageParams

    Returns
    -------
    problem : cp.Problem (not yet solved)
    model_vars    : dict with keys 'u_plus', 'u_minus', 's'
    """
    H = len(prices)

    u_plus = cp.Variable(H, nonneg=True, name="u_plus")
    u_minus = cp.Variable(H, nonneg=True, name="u_minus")
    # H+1 SoC states: s[0] = initial, s[H] = final after all H steps.
    # This ensures every discharge is bounded by available storage, including
    # the last timestep (which a T-variable formulation leaves unconstrained).
    s = cp.Variable(H + 1, nonneg=True, name="s")

    # hp_output[t] = demand[t] + u_plus[t] - u_minus[t]
    hp_output = demand + u_plus - u_minus

    # Objective: minimise total electricity cost
    objective = cp.Minimize(cp.sum(prices @ hp_output) / params.cop)

    constraints = [
        # Initial SoC
        s[0] == s0,
        # SoC dynamics for all H steps (always H constraints, never empty)
        s[1:] == s[:-1] + u_plus - u_minus / params.eta,
        # SoC bounds on all H+1 states
        s >= params.s_min,
        s <= params.s_max,
        # Charge / discharge bounds
        u_plus <= params.u_plus_max,
        u_minus <= params.u_minus_max,
        # Heat pump: non-negative output and grid cap
        hp_output >= 0,
        hp_output <= params.h_max,
    ]

    return cp.Problem(objective, constraints), {"u_plus": u_plus, "u_minus": u_minus, "s": s}



def solve_perfect_foresight(
    prices: np.ndarray,
    demand: np.ndarray,
    storage_params: StorageParams,
) -> dict:
    """Solve the full-horizon LP with perfect knowledge of prices and demand.

    Parameters
    ----------
    prices : np.ndarray of shape (T,) — true electricity prices (£/MWh)
    demand : np.ndarray of shape (T,) — true heat demand (MWh/day)
    storage_params : StorageParams

    Returns
    -------
    dict with keys:
        soc        : np.ndarray(T) — state of charge at the start of each day (MWh)
        charge     : np.ndarray(T) — heat charged into storage (MWh/day)
        discharge  : np.ndarray(T) — heat discharged from storage (MWh/day)
        hp_output  : np.ndarray(T) — heat pump thermal output (MWh/day)
        cost       : float — total electricity spend (£)
        status     : str — cvxpy solver status

    Raises
    ------
    ValueError if the solver does not find an optimal solution.
    """
    prices = np.asarray(prices, dtype=float)
    demand = np.asarray(demand, dtype=float)

    problem, model_vars = _build_lp(prices, demand, storage_params.s0, storage_params)
    problem.solve(solver=cp.CLARABEL)

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise ValueError(f"LP solver returned non-optimal status: {problem.status!r}")

    u_plus = np.maximum(0.0, np.array(model_vars["u_plus"].value))
    u_minus = np.maximum(0.0, np.array(model_vars["u_minus"].value))

    # Interior-point solvers can produce degenerate solutions where both u_plus
    # and u_minus are positive on the same day. This is never strictly optimal
    # (it wastes round-trip efficiency), so collapse to the net direction.
    overlap = np.minimum(u_plus, u_minus)
    u_plus = u_plus - overlap
    u_minus = u_minus - overlap

    # Recompute SoC from the cleaned flows (slightly better than LP SoC due to
    # eliminated round-trip losses).
    s = np.empty(len(demand))
    s[0] = storage_params.s0
    for t in range(len(demand) - 1):
        s[t + 1] = s[t] + u_plus[t] - u_minus[t] / storage_params.eta

    hp_output = np.maximum(0.0, demand + u_plus - u_minus)
    cost = float(np.dot(prices, hp_output) / storage_params.cop)

    return {
        "soc": s,
        "charge": u_plus,
        "discharge": u_minus,
        "hp_output": hp_output,
        "cost": cost,
        "status": problem.status,
    }


def solve_single_window(
    prices_forecast: np.ndarray,
    demand_forecast: np.ndarray,
    s_current: float,
    storage_params: StorageParams,
) -> dict:
    """Solve the LP over a single MPC window of length H.

    The caller should apply only the first action (charge[0], discharge[0]) and
    advance the true SoC before calling again at the next time step.

    Parameters
    ----------
    prices_forecast  : np.ndarray of shape (H,)
    demand_forecast  : np.ndarray of shape (H,)
    s_current        : current SoC (MWh)
    storage_params   : StorageParams

    Returns
    -------
    Same dict structure as solve_perfect_foresight, but of length H.

    Raises
    ------
    ValueError if the solver does not find an optimal solution.
    """
    prices_forecast = np.asarray(prices_forecast, dtype=float)
    demand_forecast = np.asarray(demand_forecast, dtype=float)

    problem, model_vars = _build_lp(prices_forecast, demand_forecast, s_current, storage_params)
    problem.solve(solver=cp.CLARABEL)

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise ValueError(f"LP solver returned non-optimal status: {problem.status!r}")

    u_plus = np.maximum(0.0, np.array(model_vars["u_plus"].value))
    u_minus = np.maximum(0.0, np.array(model_vars["u_minus"].value))

    overlap = np.minimum(u_plus, u_minus)
    u_plus = u_plus - overlap
    u_minus = u_minus - overlap

    s = np.empty(len(demand_forecast))
    s[0] = s_current
    for t in range(len(demand_forecast) - 1):
        s[t + 1] = s[t] + u_plus[t] - u_minus[t] / storage_params.eta

    hp_output = np.maximum(0.0, demand_forecast + u_plus - u_minus)
    cost = float(np.dot(prices_forecast, hp_output) / storage_params.cop)

    return {
        "soc": s,
        "charge": u_plus,
        "discharge": u_minus,
        "hp_output": hp_output,
        "cost": cost,
        "status": problem.status,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.simulation import SimulationParams, generate_price_series, generate_demand_series

    sim = SimulationParams(seed=42, T=365)
    prices = generate_price_series(sim)
    demand = generate_demand_series(sim)
    sp = StorageParams()

    # Baseline: no storage, buy exactly enough electricity to meet demand every day
    baseline_cost = float(np.dot(prices, demand) / sp.cop)

    result = solve_perfect_foresight(prices, demand, sp)
    print(f"Solver status:     {result['status']}")
    print(f"Baseline cost:     £{baseline_cost:,.0f}")
    print(f"Perfect foresight: £{result['cost']:,.0f}  ({100 * (result['cost'] - baseline_cost) / baseline_cost:+.1f}%)")

    soc = result["soc"]
    print(f"SoC range:         [{soc.min():.1f}, {soc.max():.1f}] MWh  (bounds: [{sp.s_min}, {sp.s_max}])")

    charge = result["charge"]
    discharge = result["discharge"]
    hp = result["hp_output"]
    print(f"Charge range:      [{charge.min():.1f}, {charge.max():.1f}] MWh/day  (max: {sp.u_plus_max})")
    print(f"Discharge range:   [{discharge.min():.1f}, {discharge.max():.1f}] MWh/day  (max: {sp.u_minus_max})")
    print(f"HP output range:   [{hp.min():.1f}, {hp.max():.1f}] MWh/day  (max: {sp.h_max})")

    # Verify constraints
    assert np.all(soc >= sp.s_min - 1e-6), "SoC below s_min!"
    assert np.all(soc <= sp.s_max + 1e-6), "SoC above s_max!"
    assert np.all(hp >= -1e-6), "Negative HP output!"
    assert np.all(hp <= sp.h_max + 1e-6), "HP output exceeds h_max!"
    assert result["cost"] <= baseline_cost + 1e-3, "PF cost exceeds baseline!"
    print("\nAll constraints satisfied. ✓")
