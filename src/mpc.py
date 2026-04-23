"""
mpc.py — Rolling-horizon MPC controller for thermal storage.

At each time step the controller:
  1. Generates a price forecast for the next H days using get_forecast_window
  2. Solves an LP over that window with solve_single_window
  3. Applies only the first action (charge[0], discharge[0])
  4. Advances the true SoC using the true efficiency equations
  5. Records the action and true cost

Only price uncertainty is modelled — demand within the window uses true values,
which is a standard simplification for thermal systems where demand is predictable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .simulation import SimulationParams, get_forecast_window
from .storage_lp import StorageParams, solve_single_window


@dataclass
class MPCParams:
    H: int = 30    # rolling horizon length (days)


def run_mpc(
    true_prices: np.ndarray,
    true_demand: np.ndarray,
    storage_params: StorageParams,
    mpc_params: MPCParams,
    sim_params: SimulationParams,
) -> dict:
    """Run the rolling-horizon MPC controller.

    At each time step a short-horizon LP is solved using a noisy price forecast
    and true demand. Only the first action from each solution is applied; the SoC
    is then advanced using the true physical equations.

    Parameters
    ----------
    true_prices : np.ndarray of shape (T,) — true electricity prices (£/MWh);
        used only for cost accounting, not passed to the LP solver
    true_demand : np.ndarray of shape (T,) — true heat demand (MWh/day);
        used both as the within-window demand forecast and for cost accounting
    storage_params : StorageParams
    mpc_params : MPCParams
    sim_params : SimulationParams — needed by get_forecast_window for the seed
        and forecast blending parameters

    Returns
    -------
    dict with keys:
        soc         : np.ndarray(T) — SoC at start of each day (MWh)
        charge      : np.ndarray(T) — applied heat charge (MWh/day)
        discharge   : np.ndarray(T) — applied heat discharge (MWh/day)
        hp_output   : np.ndarray(T) — true heat pump output (MWh/day)
        cost        : float — total electricity cost at true prices (£)
        n_fallbacks : int — number of steps where the solver failed and hold was applied
    """
    true_prices = np.asarray(true_prices, dtype=float)
    true_demand = np.asarray(true_demand, dtype=float)
    T = len(true_prices)

    soc = np.empty(T)
    charge = np.empty(T)
    discharge = np.empty(T)
    hp_output = np.empty(T)
    total_cost = 0.0
    n_fallbacks = 0

    s = storage_params.s0

    for t in range(T):
        H_window = min(mpc_params.H, T - t)

        price_fc = get_forecast_window(t, H_window, true_prices, sim_params)
        demand_fc = true_demand[t : t + H_window]

        try:
            window = solve_single_window(price_fc, demand_fc, s, storage_params)
            u_plus_t = float(window["charge"][0])
            u_minus_t = float(window["discharge"][0])
        except ValueError:
            u_plus_t = 0.0
            u_minus_t = 0.0
            n_fallbacks += 1

        # True heat pump output at this step
        hp_t = np.clip(
            true_demand[t] + u_plus_t - u_minus_t,
            0.0,
            storage_params.h_max,
        )

        soc[t] = s
        charge[t] = u_plus_t
        discharge[t] = u_minus_t
        hp_output[t] = hp_t
        total_cost += true_prices[t] * hp_t / storage_params.cop

        # Advance true SoC with efficiency applied on discharge
        s = s + u_plus_t - u_minus_t / storage_params.eta
        s = float(np.clip(s, storage_params.s_min, storage_params.s_max))

    return {
        "soc": soc,
        "charge": charge,
        "discharge": discharge,
        "hp_output": hp_output,
        "cost": total_cost,
        "n_fallbacks": n_fallbacks,
    }


if __name__ == "__main__":
    from src.simulation import generate_price_series, generate_demand_series
    from src.storage_lp import solve_perfect_foresight

    sim = SimulationParams(seed=42, T=365)
    prices = generate_price_series(sim)
    demand = generate_demand_series(sim)
    sp = StorageParams()
    mp = MPCParams(H=30)

    baseline_cost = float(np.dot(prices, demand) / sp.cop)
    pf = solve_perfect_foresight(prices, demand, sp)
    mpc = run_mpc(prices, demand, sp, mp, sim)

    print(f"Baseline cost:     £{baseline_cost:,.0f}")
    print(f"Perfect foresight: £{pf['cost']:,.0f}  ({100 * (pf['cost'] - baseline_cost) / baseline_cost:+.1f}%)")
    print(f"MPC (H={mp.H}):        £{mpc['cost']:,.0f}  ({100 * (mpc['cost'] - baseline_cost) / baseline_cost:+.1f}%)")
    print(f"MPC fallbacks:     {mpc['n_fallbacks']}")

    assert mpc["cost"] >= pf["cost"] - 1e-3, "MPC cost is below perfect foresight — something is wrong!"
    assert np.all(mpc["soc"] >= sp.s_min - 1e-6), "SoC below s_min!"
    assert np.all(mpc["soc"] <= sp.s_max + 1e-6), "SoC above s_max!"
    assert np.all(mpc["hp_output"] >= -1e-6), "Negative HP output!"
    assert np.all(mpc["hp_output"] <= sp.h_max + 1e-6), "HP output exceeds h_max!"
    print("\nAll constraints satisfied. ✓")
