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

Implemented in `src/mpc.py` and `notebooks/mpc.py`.

Demand within the window uses true values. The horizon shrinks at the end of the
year so the window never extends beyond the simulation.

- [x] Define `MPCParams(H: int = 30)` — horizon length is the only MPC-specific parameter
- [x] Implement `run_mpc(true_prices, true_demand, storage_params, mpc_params, sim_params) -> dict`:
  - At each time step `t`:
    1. Compute `H_window = min(H, T - t)` to handle end-of-year boundary
    2. Generate a price forecast via `get_forecast_window(t, H_window, true_prices, sim_params)`
    3. Solve LP with `solve_single_window(price_fc, demand_fc, s, storage_params)`
    4. Apply `charge[0]`, `discharge[0]` from the solution
    5. Advance true SoC: `s = s + u_plus - u_minus / eta`, clipped to `[s_min, s_max]`
    6. Record true cost: `prices[t] * hp_output[t] / cop`
  - Returns dict with keys: `soc`, `charge`, `discharge`, `hp_output`, `cost`, `n_fallbacks`
- [x] Fallback on solver failure: hold (zero charge/discharge)
- [x] Verified via `python -m src.mpc`: MPC saves 3.1% vs baseline, PF saves 3.5%;
  MPC cost ≥ PF cost; all constraints satisfied
- [x] Created `notebooks/mpc.py` with parameter panel (storage + HP + horizon slider),
  stacked dispatch charts with secondary price axis, SoC comparison, cumulative cost,
  and summary stats including "cost of price uncertainty" (MPC − PF gap)

---

## Stage 5: Marimo Notebooks

Rather than a single `notebook.py`, the demo was split into three focused notebooks,
each with its own parameter panel, plots, and summary stats:

- [x] `notebooks/simulation.py` — price and demand time series, forecast visualisation,
  regenerate buttons, forecast horizon range slider
- [x] `notebooks/lp.py` — perfect foresight LP demo with full parameter panel
  (storage, HP), dispatch/SoC/cumulative-cost charts, summary stats
- [x] `notebooks/mpc.py` — MPC vs perfect foresight comparison with horizon slider,
  stacked dispatch charts, SoC comparison, cumulative cost, summary stats including
  "cost of price uncertainty"
- [x] Solver errors surfaced via `mo.callout` / `mo.stop` in both `lp.py` and `mpc.py`

Note: a "forecast noise level" slider was not added to `mpc.py`; the forecast horizon
sliders in `simulation.py` serve a similar exploratory purpose.

---

## Stage 6: Plotly Visualisations

All visualisations were implemented inline within the notebooks rather than as a
separate `src/plots.py` module. This avoided an extra layer of abstraction for a demo
project. The following charts are present across the notebooks:

- [x] Price series with seasonal average, raw forecast, and blended forecast (`simulation.py`)
- [x] Heat demand with seasonal baseline (`simulation.py`)
- [x] Dispatch schedule (charge/discharge bars + price overlay) (`lp.py`, `mpc.py`)
- [x] State of charge over time (`lp.py`, `mpc.py`)
- [x] Cumulative electricity cost vs baseline (`lp.py`, `mpc.py`)
- [x] PF vs MPC SoC comparison on shared axes (`mpc.py`)
- [x] Consistent colour scheme and explicit `height` on all figures

Note: `src/plots.py` was not created; a shaded s_min/s_max feasibility band and a
final summary bar chart were not implemented.

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
