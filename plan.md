# MPC Energy Storage Demo — Project Plan

A marimo notebook demonstrating Model Predictive Control (MPC) for an energy storage system
operating under price and demand uncertainty, benchmarked against a perfect-foresight oracle.
The optimisation is formulated as a Linear Programme (LP) solved with `cvxpy`.

---

## Stage 1: Project Setup

- [ ] Initialise a `uv` project (`uv init mpc-storage-demo`)
- [ ] Add core dependencies:
  - `marimo` — reactive notebook runtime
  - `cvxpy` — LP formulation and solving
  - `numpy` — numerical operations and scenario generation
  - `pandas` — time-indexed results storage
  - `plotly` — interactive visualisations
- [ ] Create the top-level marimo notebook file: `notebook.py`
- [ ] Add a `README.md` with a brief description of the demo and instructions for running it (`uv run marimo run notebook.py`)

---

## Stage 2: Price and Demand Simulation

Implement a module `src/simulation.py` that generates synthetic price and demand time series.

- [ ] Define a `SimulationParams` dataclass holding:
  - Number of time steps `T` (e.g. 48 half-hourly steps = 24 hours)
  - Diurnal price shape parameters (peak hours, amplitude)
  - AR(1) noise parameters (`phi`, `sigma`) for price uncertainty
  - Mean demand and demand noise standard deviation
  - Random seed for reproducibility
- [ ] Implement `generate_price_series(params) -> np.ndarray` using an AR(1) process
  superimposed on a smooth diurnal curve (e.g. double Gaussian for morning/evening peaks)
- [ ] Implement `generate_demand_series(params) -> np.ndarray` with independent Gaussian noise
- [ ] Implement `generate_price_forecast(true_prices, params) -> np.ndarray` that returns a
  noisy forecast (used by MPC in place of true future prices); noise should grow with
  forecast horizon to reflect realistic uncertainty
- [ ] Write a standalone test (or marimo cell) that plots the true vs forecast price series
  to verify the simulation visually

---

## Stage 3: LP Formulation

Implement a module `src/storage_lp.py` containing the core optimisation model.

- [ ] Define a `StorageParams` dataclass holding:
  - Capacity `s_max` (MWh)
  - Minimum SoC `s_min` (MWh, e.g. 10% of capacity)
  - Maximum charge rate `u_plus_max` (MW)
  - Maximum discharge rate `u_minus_max` (MW)
  - Charge efficiency `eta_plus`
  - Discharge efficiency `eta_minus`
  - Initial SoC `s0`
- [ ] Implement `solve_perfect_foresight(prices, storage_params) -> dict`:
  - Variables: `u_plus[t]`, `u_minus[t]` (charge/discharge rates), `s[t]` (SoC)
  - Objective: minimise total cost `sum(prices[t] * (u_plus[t] - u_minus[t]))`
    (negative cost = revenue from discharging at high prices)
  - Constraints:
    - SoC dynamics: `s[t+1] == s[t] + eta_plus * u_plus[t] - u_minus[t] / eta_minus`
    - `s_min <= s[t] <= s_max` for all `t`
    - `0 <= u_plus[t] <= u_plus_max` for all `t`
    - `0 <= u_minus[t] <= u_minus_max` for all `t`
    - `s[0] == s0`
  - Return a dict with keys: `soc`, `charge`, `discharge`, `cost`, `status`
- [ ] Implement `solve_single_window(prices_forecast, s_current, storage_params) -> dict`
  for a single MPC window solve (same LP over a horizon of length `H`); return the
  full planned trajectory but only the first action will be applied
- [ ] Handle infeasible/solver-error cases gracefully with clear error messages

---

## Stage 4: MPC Loop

Implement `src/mpc.py` containing the rolling-horizon controller.

- [ ] Define an `MPCParams` dataclass holding:
  - Horizon length `H` (number of steps to optimise over)
  - Whether to use a noisy forecast or true prices within the window
- [ ] Implement `run_mpc(true_prices, storage_params, mpc_params, sim_params) -> dict`:
  - At each time step `t`:
    1. Generate a forecast for `prices[t : t+H]` using `generate_price_forecast`
    2. Solve the LP over the horizon using `solve_single_window`
    3. Apply only `u_plus[0]`, `u_minus[0]` from the solution
    4. Advance the true SoC using the true efficiency equations
    5. Record the applied action and resulting SoC
  - Return a dict with keys: `soc`, `charge`, `discharge`, `cost` (computed using
    true prices, not forecast prices)
- [ ] Include a fallback if the solver returns infeasible for a given window (e.g. hold,
  or apply zero charge/discharge)

---

## Stage 5: Marimo Notebook

Build the interactive demo in `notebook.py`.

- [ ] **Parameter panel** — use `mo.ui` controls for:
  - Storage capacity, min SoC, max charge/discharge rate, efficiencies
  - MPC horizon length `H`
  - Simulation seed and number of time steps
  - Forecast noise level
- [ ] **Simulation cell** — call `generate_price_series` and `generate_demand_series`;
  display a price series preview plot
- [ ] **Perfect foresight cell** — call `solve_perfect_foresight`; display result or solver error
- [ ] **MPC cell** — call `run_mpc`; display result or solver error
- [ ] **Visualisation cells** — see Stage 6 below
- [ ] **Summary stats cell** — compute and display a `mo.stat` or table showing:
  - Total cost / revenue for each strategy
  - Number of MPC windows that were infeasible
  - Average SoC utilisation

---

## Stage 6: Plotly Visualisations

Implement a module `src/plots.py` with functions that return `plotly.graph_objects.Figure` objects.

- [ ] `plot_price_series(true_prices, forecast_prices) -> Figure`:
  - True prices as a solid line
  - Forecast as a dashed line with a shaded uncertainty band (±1 std, estimated from
    noise params)
- [ ] `plot_soc_comparison(soc_pf, soc_mpc, time_index) -> Figure`:
  - Two SoC traces on the same axes (perfect foresight vs MPC)
  - Shaded band between `s_min` and `s_max` to show feasible region
- [ ] `plot_dispatch_comparison(charge_pf, discharge_pf, charge_mpc, discharge_mpc, prices, time_index) -> Figure`:
  - Grouped bar chart of charge/discharge for each strategy
  - Price overlaid as a secondary y-axis line
- [ ] `plot_cost_breakdown(cost_pf, cost_mpc) -> Figure`:
  - Cumulative cost over time for each strategy
  - Final bar comparing total cost/revenue
- [ ] All figures should use a consistent colour scheme and be sized appropriately for
  embedding in a marimo notebook (set `height` explicitly)

---

## Stage 7: Testing and Validation

- [ ] Add a `tests/` directory
- [ ] Write `tests/test_lp.py`:
  - Test that the perfect foresight solution is always at least as good as MPC (by cost)
  - Test SoC constraint satisfaction for both solvers
  - Test that the solver returns a valid result for a trivial 2-step case with known
    optimal solution
- [ ] Write `tests/test_simulation.py`:
  - Test that generated prices are within a plausible range
  - Test that the forecast converges to the true price when noise sigma is 0
- [ ] Run tests with `uv run pytest`

---

## Stage 8: Polish and Documentation

- [ ] Add docstrings to all public functions in `src/`
- [ ] Add a `NOTES.md` explaining:
  - The mathematical formulation of the LP
  - The MPC rolling-horizon algorithm
  - Key modelling simplifications (no simultaneous charge/discharge enforcement,
    linearised efficiency, no degradation model)
  - Ideas for extending the demo (stochastic MPC, battery degradation, multi-asset)
- [ ] Verify the notebook runs cleanly end-to-end with `uv run marimo run notebook.py`
- [ ] Optional: export a static HTML snapshot with `uv run marimo export html notebook.py`

---

## File Structure (target)

```
mpc-storage-demo/
├── pyproject.toml
├── README.md
├── NOTES.md
├── PLAN.md
├── notebook.py
├── src/
│   ├── __init__.py
│   ├── simulation.py
│   ├── storage_lp.py
│   ├── mpc.py
│   └── plots.py
└── tests/
    ├── test_lp.py
    └── test_simulation.py
```
