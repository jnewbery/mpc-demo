import numpy as np
import pytest

from src.forecast import SimulationParams
from src.storage_lp import StorageParams, solve_perfect_foresight, solve_single_window
from src.mpc import run_mpc


def test_mpc_cost_at_least_pf(test_prices: np.ndarray,
                              test_seasonal_prices: np.ndarray,
                              test_demand: np.ndarray,
                              test_sp: StorageParams) -> None:
    """MPC cost must be >= perfect foresight cost (PF is optimal with full information)."""
    sim = SimulationParams(T=365)

    pf = solve_perfect_foresight(test_prices, test_demand, test_sp)
    mpc = run_mpc(test_prices, test_demand, test_sp, sim, seasonal_prices=test_seasonal_prices)

    assert mpc["cost"] >= pf["cost"] - 1e-3


def test_mpc_soc_bounds(test_prices: np.ndarray,
                        test_seasonal_prices: np.ndarray,
                        test_demand: np.ndarray,
                        test_sp: StorageParams) -> None:
    sim = SimulationParams(T=365)
    mpc = run_mpc(test_prices, test_demand, test_sp, sim, seasonal_prices=test_seasonal_prices)

    assert np.all(mpc["soc"] >= test_sp.s_min - 1e-5)
    assert np.all(mpc["soc"] <= test_sp.s_max + 1e-5)


def test_mpc_hp_output_bounds(test_prices: np.ndarray,
                              test_seasonal_prices: np.ndarray,
                              test_demand: np.ndarray,
                              test_sp: StorageParams) -> None:
    sim = SimulationParams(T=365)
    mpc = run_mpc(test_prices, test_demand, test_sp, sim, seasonal_prices=test_seasonal_prices)

    assert np.all(mpc["hp_output"] >= -1e-5)
    assert np.all(mpc["hp_output"] <= test_sp.h_max + 1e-5)


# ---------------------------------------------------------------------------
# solve_single_window
# ---------------------------------------------------------------------------

def test_single_window_matches_pf_on_full_horizon(test_prices: np.ndarray,
                                                  test_demand: np.ndarray,
                                                  test_sp: StorageParams) -> None:
    """solve_single_window over the full horizon from s0 should give the same
    cost as solve_perfect_foresight."""
    prices = test_prices[:30]
    demand = test_demand[:30]

    pf = solve_perfect_foresight(prices, demand, test_sp)
    sw = solve_single_window(prices, demand, test_sp.s0, test_sp)

    assert sw["cost"] == pytest.approx(pf["cost"], rel=1e-4)
