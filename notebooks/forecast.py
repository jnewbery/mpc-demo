"""
Title: Forecast
Description: Price and heat demand forecasts against pre-generated nominal scenarios
"""

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import dataclasses
    import pathlib
    import marimo as mo
    import numpy as np
    import polars as pl
    import plotly.graph_objects as go

    from src.simulation import (
        SimulationParams,
        generate_seasonal_demand,
        generate_demand_series,
        generate_price_forecast,
        generate_raw_price_forecast,
    )

    return (
        SimulationParams,
        dataclasses,
        generate_demand_series,
        generate_price_forecast,
        generate_raw_price_forecast,
        generate_seasonal_demand,
        go,
        mo,
        np,
        pathlib,
        pl,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Simulation

    ---

    ## Price scenario

    The price time series and its seasonal baseline are imported from a pre-generated
    scenario file (select below).  Each scenario is produced by fitting a Fourier
    regression and AR(1) residual model to historical Germany/Luxembourg day-ahead
    prices — see the [prices notebook](../prices) for details.

    The CSV contains both the scenario price path $P_t$ and the fitted seasonal curve
    $S_t$, which is used as the long-run convergence target for the forecast below.

    ---

    ## Heat demand

    Heat demand is generated synthetically. The seasonal baseline follows a cosine with a winter peak, scaled by a weekend multiplier $m_t$:

    $$S^D_t = \left(\mu_D + A_D \cos\!\left(\frac{2\pi t}{365}\right)\right) \cdot m_t, \qquad m_t = \begin{cases} 0.8 & t \bmod 7 \in \{5, 6\} \\ 1.0 & \text{otherwise} \end{cases}$$

    Day-to-day variability is an AR(1) process with $\phi_D = 0.90$, reflecting the stronger persistence of weather systems:

    $$\delta_t = \phi_D\,\delta_{t-1} + \sigma_D z_t, \qquad D_t = \max\!\left(0,\; S^D_t + \delta_t\right)$$

    ---

    ## Price forecast

    A forecast made at time $t$ for horizon $h$ is constructed in two steps.

    **Step 1 — forecast deviation random walk.**
    The forecaster knows the current deviation from the scenario seasonal exactly, then projects it forward with growing uncertainty:

    $$\hat{d}_0 = P_t - S_t, \qquad \hat{d}_h = \hat{d}_{h-1} + \sigma_\text{step}\, z_h$$

    where $\sigma_\text{step} = \sigma_\infty / \sqrt{H_\text{long}}$ is calibrated so that the forecast uncertainty reaches $\sigma_\infty$ by the long-term horizon $H_\text{long}$.

    **Step 2 — blend toward the seasonal average.**
    The prediction weights the forecast deviation against the scenario seasonal $S_t$ via a blending weight $\alpha(h)$:

    $$\hat{P}_{t+h} = S_{t+h} + \alpha(h)\,\hat{d}_h$$

    $$\alpha(h) = \begin{cases} 1 & h \le H_\text{short} \\ \dfrac{1}{2}\!\left(1 + \cos\!\left(\pi\,\dfrac{h - H_\text{short}}{H_\text{long} - H_\text{short}}\right)\right) & H_\text{short} < h < H_\text{long} \\ 0 & h \ge H_\text{long} \end{cases}$$

    For $h \le H_\text{short}$ the forecast closely tracks the true price. For $h \ge H_\text{long}$ the weight is zero and the forecast converges to $S_t$. The **forecast horizon** slider controls $H_\text{short}$ and $H_\text{long}$.
    """)
    return


@app.cell
def _(mo, pathlib):
    _scenario_dir = pathlib.Path(__file__).parent.parent / "data" / "price_scenarios"
    _files = sorted(_scenario_dir.glob("*.csv"))
    scenario_selector = mo.ui.dropdown(
        options={f.stem.replace("_", " ").title(): str(f) for f in _files},
        value=_files[0].stem.replace("_", " ").title() if _files else None,
        label="Price scenario",
    )
    return (scenario_selector,)


@app.cell
def _(mo, np):
    get_demand_seed, set_demand_seed = mo.state(42)
    regen_demand = mo.ui.run_button(
        label="Regenerate demand",
        on_change=lambda _: set_demand_seed(int(np.random.randint(0, 100_000))),
    )
    return get_demand_seed, regen_demand


@app.cell
def _(mo):
    forecast_horizon = mo.ui.range_slider(
        start=0, stop=90, value=[7, 30], step=1,
        label="Forecast horizon: short-term / long-term (days)",
        show_value=True,
    )
    forecast_horizon
    return (forecast_horizon,)


@app.cell
def _(pl, scenario_selector):
    _df = pl.read_csv(scenario_selector.value)
    prices = _df.get_column("price").to_numpy()
    seasonal = _df.get_column("seasonal").to_numpy()
    return prices, seasonal


@app.cell
def _(
    SimulationParams,
    dataclasses,
    forecast_horizon,
    generate_price_forecast,
    generate_raw_price_forecast,
    prices,
    seasonal,
):
    _short, _long = forecast_horizon.value
    _fparams = dataclasses.replace(
        SimulationParams(T=len(prices)),
        forecast_short_term=_short,
        forecast_long_term=_long,
    )
    _forecast_matrix = generate_price_forecast(prices, _fparams, seasonal_array=seasonal)
    price_forecast = _forecast_matrix[0, :]
    price_forecast_raw = generate_raw_price_forecast(prices, _fparams, seasonal_array=seasonal)
    return price_forecast, price_forecast_raw


@app.cell
def _(
    SimulationParams,
    generate_demand_series,
    generate_seasonal_demand,
    get_demand_seed,
):
    _params = SimulationParams(seed=get_demand_seed())
    seasonal_demand = generate_seasonal_demand(_params)
    demand = generate_demand_series(_params)
    return demand, seasonal_demand


@app.cell
def _(
    go,
    mo,
    np,
    price_forecast,
    price_forecast_raw,
    prices,
    scenario_selector,
    seasonal,
):
    _days = np.arange(1, len(prices) + 1)

    _fig = go.Figure()

    # Seasonal baseline
    _fig.add_trace(go.Scatter(
        x=_days,
        y=seasonal,
        mode="lines",
        name="Seasonal average",
        line=dict(color="#4a90d9", width=1.5, dash="dash"),
    ))

    # Raw deviation (unblended forecast)
    _fig.add_trace(go.Scatter(
        x=_days,
        y=price_forecast_raw,
        mode="lines",
        name="Raw forecast (unblended)",
        line=dict(color="grey", width=1, dash="dot"),
    ))

    # Blended forecast line
    _fig.add_trace(go.Scatter(
        x=_days,
        y=price_forecast,
        mode="lines",
        name="Forecast",
        line=dict(color="black", width=1),
    ))

    # True price line
    _fig.add_trace(go.Scatter(
        x=_days,
        y=prices,
        mode="lines",
        name="True price",
        line=dict(color="#e07b39", width=1.5),
    ))

    _fig.update_layout(
        title="Energy Price",
        xaxis_title="Day of year",
        yaxis_title="€/MWh",
        height=400,
        margin=dict(t=50, b=40, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    mo.vstack([scenario_selector, _fig])
    return


@app.cell
def _(demand, go, mo, np, regen_demand, seasonal_demand):
    _days = np.arange(1, len(demand) + 1)
    _fig = go.Figure()

    # Seasonal baseline
    _fig.add_trace(go.Scatter(
        x=_days,
        y=seasonal_demand,
        mode="lines",
        name="Seasonal average",
        line=dict(color="#4a90d9", width=1),
    ))

    # True demand
    _fig.add_trace(go.Scatter(
        x=_days,
        y=demand,
        mode="lines",
        name="True demand",
        line=dict(color="#e07b39", width=1.5),
    ))

    _fig.update_layout(
        title="Heat Demand",
        xaxis_title="Day of year",
        yaxis_title="MWh/day",
        height=400,
        margin=dict(t=50, b=40, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    mo.vstack([regen_demand, _fig])
    return


if __name__ == "__main__":
    app.run()
