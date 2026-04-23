# MPC Energy Storage Demo — Project Plan

A marimo notebook demonstrating Model Predictive Control (MPC) for an energy storage system
operating under price and demand uncertainty, benchmarked against a perfect-foresight oracle.
The optimisation is formulated as a Linear Programme (LP) solved with `cvxpy`.

---

## Stage 1: Project Setup

- [x] Initialise a `uv` project (`uv init mpc-storage-demo`)
- [x] Add core dependencies:
  - `marimo` — reactive notebook runtime
  - `cvxpy` — LP formulation and solving
  - `numpy` — numerical operations and scenario generation
  - `polars` — time-indexed results storage
  - `plotly` — interactive visualisations
- [x] Create the marimo notebook file: `notebooks/notebook.py`
- [x] Add a `README.md` with a brief description of the demo and instructions for running it (`poe run`)

---

## Stage 2: Price and Demand Simulation

Implement a module `src/simulation.py` that generates synthetic price and demand time series.

Uses **365 daily time steps** (one per day of the year) to model a full calendar year.
Heat demand and energy prices both follow seasonal patterns peaking in winter.

- [x] Define a `SimulationParams` dataclass holding:
  - Number of time steps `T` (default 365 — one day per step)
  - Price parameters: annual mean, seasonal amplitude, AR(1) coefficients (`phi`, `sigma`)
  - Demand parameters: annual mean, seasonal amplitude, weekend reduction factor, noise std dev
  - Forecast parameters: base error and per-step growth in forecast uncertainty
  - Random seed for reproducibility
- [x] Implement `generate_price_series(params) -> np.ndarray` using a seasonal cosine
  (peaking on day 0 = 1 Jan, i.e. highest in winter) superimposed with AR(1) correlated noise;
  prices clipped to a minimum of £1/MWh
- [x] Implement `generate_demand_series(params) -> np.ndarray` with a seasonal cosine
  (winter peak) multiplied by a weekend reduction factor (`t % 7` proxy for day-of-week),
  plus independent Gaussian noise; demand clipped to a minimum of 0 MWh/day
- [x] Implement `generate_price_forecast(true_prices, params) -> np.ndarray` returning a
  `(T, T)` matrix where `forecast[t, h]` is the price forecast for day `t+h` made at day `t`;
  noise grows linearly with horizon `h`
- [x] Implement `get_forecast_window(t, H, true_prices, params) -> np.ndarray` convenience
  wrapper returning a 1-D array of length `H` for use in the MPC loop
- [x] Verified via `__main__` block: winter prices ~108 £/MWh vs summer ~57 £/MWh;
  winter demand ~69 MWh/day vs summer ~25 MWh/day; weekend demand ~19% lower than weekday

---

## Stage 3: LP Formulation

Implemented in `src/storage_lp.py`.

The system models a heat pump connected to thermal storage. The heat pump is the sole
heat source; its output must satisfy the demand balance at every time step:
`hp[t] = demand[t] + u_plus[t] - u_minus[t]`. Electricity cost is `prices[t] * hp[t] / COP`.

- [x] Define a `StorageParams` dataclass holding:
  - `s_max`, `s_min` (MWh) — storage capacity bounds
  - `s0` (MWh) — initial state of charge
  - `u_plus_max`, `u_minus_max` (MWh/day) — charge/discharge rate limits
  - `eta` — round-trip discharge efficiency (applied on discharge: `u_minus / eta`)
  - `cop` — heat pump coefficient of performance
  - `h_max` (MWh/day) — maximum heat pump output (grid/equipment cap)
- [x] Implement `solve_perfect_foresight(prices, demand, storage_params) -> dict`:
  - Variables: `u_plus[t]`, `u_minus[t]`, `s[t]`
  - Objective: minimise `sum(prices[t] * (demand[t] + u_plus[t] - u_minus[t]) / COP)`
  - Constraints:
    - SoC dynamics: `s[t+1] == s[t] + u_plus[t] - u_minus[t] / eta`
    - `s_min <= s[t] <= s_max` for all `t`; `s[0] == s0`
    - `0 <= u_plus[t] <= u_plus_max` for all `t`
    - `0 <= u_minus[t] <= u_minus_max` for all `t`
    - Demand balance / HP bounds: `0 <= demand[t] + u_plus[t] - u_minus[t] <= h_max`
  - Returns dict with keys: `soc`, `charge`, `discharge`, `hp_output`, `cost`, `status`
- [x] Implement `solve_single_window(prices_forecast, demand_forecast, s_current, storage_params) -> dict`
  for a single MPC window solve (same LP over a horizon of length `H`); returns the
  full planned trajectory but only the first action will be applied
- [x] Raises `ValueError` with solver status if cvxpy returns a non-optimal result
- [x] Verified via `__main__` block: perfect foresight saves ~3.5% vs no-storage baseline;
  all SoC, charge/discharge, and HP output constraints satisfied
- [x] Created `notebooks/lp.py` demonstrating the LP with interactive parameter sliders
  and three charts: dispatch schedule, state of charge, and cumulative cost vs baseline

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
