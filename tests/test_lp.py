import pathlib
import numpy as np
import pytest

from src.simulation import SimulationParams
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_pf():
    prices = _prices()
    demand = _demand()
    sp = StorageParams()
    result = solve_perfect_foresight(prices, demand, sp)
    return result, prices, demand, sp


# ---------------------------------------------------------------------------
# Perfect foresight
# ---------------------------------------------------------------------------

def test_pf_status_optimal():
    result, *_ = _default_pf()
    assert result["status"] in ("optimal", "optimal_inaccurate")


def test_pf_soc_bounds():
    result, _, _, sp = _default_pf()
    assert np.all(result["soc"] >= sp.s_min - 1e-5)
    assert np.all(result["soc"] <= sp.s_max + 1e-5)


def test_pf_charge_bounds():
    result, _, _, sp = _default_pf()
    assert np.all(result["charge"] >= -1e-6)
    assert np.all(result["charge"] <= sp.u_plus_max + 1e-5)


def test_pf_discharge_bounds():
    result, _, _, sp = _default_pf()
    assert np.all(result["discharge"] >= -1e-6)
    assert np.all(result["discharge"] <= sp.u_minus_max + 1e-5)


def test_pf_hp_output_bounds():
    result, _, _, sp = _default_pf()
    assert np.all(result["hp_output"] >= -1e-5)
    assert np.all(result["hp_output"] <= sp.h_max + 1e-5)


def test_pf_cost_below_baseline():
    """Perfect foresight cost must not exceed the no-storage baseline."""
    result, prices, demand, sp = _default_pf()
    baseline = float(np.dot(prices, demand) / sp.cop)
    assert result["cost"] <= baseline + 1e-3


# ---------------------------------------------------------------------------
# Trivial 2-step case with known optimal solution
#
# Setup: s_max=10, s_min=0, s0=0, eta=1, cop=1, h_max=100,
#        u_plus_max=u_minus_max=10
#        prices=[2, 1], demand=[5, 5]
#
# Day 0 is expensive (£2), day 1 is cheap (£1).
# Optimal: charge on day 1 (cheap), but there's no day 2 to discharge into.
# With only 2 steps the solver cannot benefit from storage at all — charging
# on day 0 to discharge on day 1 saves money (buy less at £2, buy more at £1).
#
# Specifically: charge x MWh on day 0 costs 2x extra; discharging x on day 1
# saves 1x. Net saving = -x (costs more). So optimal is no storage.
# BUT: prices=[1, 2] (cheap day 0, expensive day 1) — charge on day 0, discharge day 1.
# Charge x on day 0: extra cost x*1. Discharge x on day 1: save x*2. Net = +x. Charge max.
# ---------------------------------------------------------------------------

def test_pf_2step_known_optimum():
    """Cheap day 0, expensive day 1: optimal strategy runs the heat pump hard on day 0
    (pre-charging storage) and idles it on day 1 (meeting demand from storage).

    hp[0] = 10 MWh @ £1 = £10; hp[1] = 0 MWh @ £2 = £0. Total = £10 vs baseline £15.
    The LP has degenerate solutions (any charge/discharge split with the same net flow
    is equally optimal), so we assert on hp_output and cost rather than raw charge values.
    """
    prices = np.array([1.0, 2.0])
    demand = np.array([5.0, 5.0])
    sp = StorageParams(s_max=10, s_min=0, s0=0, u_plus_max=10, u_minus_max=10,
                       eta=1.0, cop=1.0, h_max=100)
    result = solve_perfect_foresight(prices, demand, sp)

    assert result["hp_output"][0] == pytest.approx(10.0, abs=1e-3)
    assert result["hp_output"][1] == pytest.approx(0.0, abs=1e-3)
    assert result["cost"] == pytest.approx(10.0, abs=1e-3)

    baseline = float(np.dot(prices, demand) / sp.cop)
    assert result["cost"] < baseline - 0.5


def test_pf_2step_no_benefit():
    """Expensive day 0, cheap day 1: storage cannot reduce cost (nothing is stored yet
    to discharge on day 0, and pre-charging on day 0 at £2 to save £1 on day 1 is a loss).
    Cost should equal the no-storage baseline."""
    prices = np.array([2.0, 1.0])
    demand = np.array([5.0, 5.0])
    sp = StorageParams(s_max=10, s_min=0, s0=0, u_plus_max=10, u_minus_max=10,
                       eta=1.0, cop=1.0, h_max=100)
    result = solve_perfect_foresight(prices, demand, sp)

    baseline = float(np.dot(prices, demand) / sp.cop)
    assert result["cost"] == pytest.approx(baseline, abs=1e-3)
    # Net charge/discharge at each step should be zero (storage offers no benefit)
    assert (result["charge"] - result["discharge"]).sum() == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# MPC
# ---------------------------------------------------------------------------

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


def test_pf_no_simultaneous_charge_discharge():
    """After post-processing, no day should have both charge > 0 and discharge > 0."""
    result, *_ = _default_pf()
    overlap = np.minimum(result["charge"], result["discharge"])
    assert np.all(overlap < 1e-6), "Simultaneous charge/discharge found in LP solution"
