import pathlib
import numpy as np
import pytest

from src.forecast import SimulationParams
from src.storage_lp import StorageParams, solve_perfect_foresight, solve_single_window
from src.mpc import MPCParams, run_mpc

_DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


def _prices(rows: int | None = None) -> np.ndarray:
    arr = np.genfromtxt(
        _DATA_DIR / "price_scenarios" / "scenario_1.csv",
        delimiter=",", skip_header=1, usecols=(2,),
    )
    return arr if rows is None else arr[:rows]

def _seasonal_prices(scenario: int = 1, rows: int | None = None) -> np.ndarray:
    arr = np.genfromtxt(
        _DATA_DIR / "price_scenarios" / f"scenario_{scenario}.csv",
        delimiter=",", skip_header=1, usecols=(1,),  # 'seasonal' column
    )
    return arr if rows is None else arr[:rows]

def _demand(rows: int | None = None) -> np.ndarray:
    arr = np.genfromtxt(
        _DATA_DIR / "heat_demand_scenarios" / "scenario_1.csv",
        delimiter=",", skip_header=1, usecols=(2,),
    )
    return arr if rows is None else arr[:rows]

def test_mpc_cost_at_least_pf():
    """MPC cost must be >= perfect foresight cost (PF is optimal with full information)."""
    sim = SimulationParams(T=365)
    prices = _prices()
    seasonal_prices = _seasonal_prices()
    demand = _demand()
    sp = StorageParams()
    mp = MPCParams(H=30)

    pf = solve_perfect_foresight(prices, demand, sp)
    mpc = run_mpc(prices, demand, sp, mp, sim, seasonal_prices=seasonal_prices)

    assert mpc["cost"] >= pf["cost"] - 1e-3


def test_mpc_soc_bounds():
    sim = SimulationParams(T=365)
    prices = _prices()
    seasonal_prices = _seasonal_prices()
    demand = _demand()
    sp = StorageParams()
    mpc = run_mpc(prices, demand, sp, MPCParams(H=30), sim, seasonal_prices=seasonal_prices)

    assert np.all(mpc["soc"] >= sp.s_min - 1e-5)
    assert np.all(mpc["soc"] <= sp.s_max + 1e-5)


def test_mpc_hp_output_bounds():
    sim = SimulationParams(T=365)
    prices = _prices()
    seasonal_prices = _seasonal_prices()
    demand = _demand()
    sp = StorageParams()
    mpc = run_mpc(prices, demand, sp, MPCParams(H=30), sim, seasonal_prices=seasonal_prices)

    assert np.all(mpc["hp_output"] >= -1e-5)
    assert np.all(mpc["hp_output"] <= sp.h_max + 1e-5)


def test_mpc_short_horizon_still_feasible():
    """A horizon of H=1 (myopic) should still complete without fallbacks."""
    sim = SimulationParams(seed=5, T=60)
    prices = _prices(rows=60)
    seasonal_prices = _seasonal_prices(rows=60)
    demand = _demand(rows=60)
    sp = StorageParams()
    mpc = run_mpc(prices, demand, sp, MPCParams(H=1), sim, seasonal_prices=seasonal_prices)
    assert mpc["n_fallbacks"] == 0


# ---------------------------------------------------------------------------
# solve_single_window
# ---------------------------------------------------------------------------

def test_single_window_matches_pf_on_full_horizon():
    """solve_single_window over the full horizon from s0 should give the same
    cost as solve_perfect_foresight."""
    prices = _prices(rows=30)
    demand = _demand(rows=30)
    sp = StorageParams()

    pf = solve_perfect_foresight(prices, demand, sp)
    sw = solve_single_window(prices, demand, sp.s0, sp)

    assert sw["cost"] == pytest.approx(pf["cost"], rel=1e-4)
