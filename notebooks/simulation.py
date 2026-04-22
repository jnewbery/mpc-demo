import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import dataclasses
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import sys
    sys.path.insert(0, ".")

    from src.simulation import (
        SimulationParams,
        generate_seasonal_prices,
        generate_seasonal_demand,
        generate_price_series,
        generate_demand_series,
        generate_price_forecast,
        forecast_sigma,
    )

    return (
        SimulationParams,
        dataclasses,
        forecast_sigma,
        generate_demand_series,
        generate_price_forecast,
        generate_price_series,
        generate_seasonal_demand,
        generate_seasonal_prices,
        go,
        mo,
        np,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Simulation

    Three time series are generated over a 365-day year ($t = 0$ $\implies$ 1 January).

    ---

    ## Seasonal baseline

    The smooth annual price cycle is a cosine with a winter peak:

    $$S_t = \mu + A \cos\!\left(\frac{2\pi t}{365}\right)$$

    where $\mu$ is the annual mean price and $A$ is the seasonal amplitude.

    ---

    ## True price

    Day-to-day price variability is modelled as an AR(1) process layered on top of the seasonal baseline:

    $$\varepsilon_t = \phi\,\varepsilon_{t-1} + \sigma z_t, \qquad z_t \overset{\text{iid}}{\sim} \mathcal{N}(0,1)$$

    $$P_t = \max\!\left(1,\; S_t + \varepsilon_t\right)$$

    The unconditional standard deviation of the noise is $\sigma_\infty = \sigma / \sqrt{1 - \phi^2}$.

    **Heat demand** follows the same structure — a seasonal cosine plus Gaussian noise — with an additional weekend multiplier $m_t$ (0.8 on Saturdays and Sundays, 1.0 otherwise):

    $$D_t = \max\!\left(0,\; \bigl(\mu_D + A_D \cos\tfrac{2\pi t}{365}\bigr)\cdot m_t + \eta_t\right), \qquad \eta_t \sim \mathcal{N}(0, \sigma_D^2)$$

    ---

    ## Price forecast

    A forecast made at time $t$ for horizon $h$ is constructed in two steps.

    **Step 1 — forecast deviation random walk.**
    The forecaster knows the current deviation from seasonal exactly, then projects it forward with growing uncertainty:

    $$\hat{d}_0 = P_t - S_t, \qquad \hat{d}_h = \hat{d}_{h-1} + \sigma_\text{step}\, z_h$$

    where $\sigma_\text{step} = \sigma_\infty / \sqrt{H_\text{long}}$ is calibrated so that the forecast uncertainty reaches $\sigma_\infty$ by the long-term horizon $H_\text{long}$. Because the walk is independent of the true future prices, the forecast can drift further from the seasonal average than the true price does.

    **Step 2 — blend toward the seasonal average.**
    The prediction weights the forecast deviation against the seasonal baseline via a blending weight $\alpha(h)$:

    $$\hat{P}_{t+h} = S_{t+h} + \alpha(h)\,\hat{d}_h$$

    $$\alpha(h) = \begin{cases} 1 & h \le H_\text{short} \\ \dfrac{1}{2}\!\left(1 + \cos\!\left(\pi\,\dfrac{h - H_\text{short}}{H_\text{long} - H_\text{short}}\right)\right) & H_\text{short} < h < H_\text{long} \\ 0 & h \ge H_\text{long} \end{cases}$$

    For $h \le H_\text{short}$ the forecast deviation is weighted fully, so the prediction closely tracks the true price. For $h \ge H_\text{long}$ the weight is zero, so the prediction equals the seasonal average. The **forecast accuracy** slider controls $H_\text{short}$, with $H_\text{long} = H_\text{short} + 23$.
    """)
    return


@app.cell
def _(mo, np):
    get_price_seed, set_price_seed = mo.state(42)
    regen_price = mo.ui.run_button(
        label="Regenerate price",
        on_change=lambda _: set_price_seed(int(np.random.randint(0, 100_000))),
    )
    return get_price_seed, regen_price


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
    forecast_quality = mo.ui.slider(
        start=0, stop=60, value=7, step=1,
        label="Forecast accuracy (days of exact forecast)",
        show_value=True,
    )
    forecast_quality
    return (forecast_quality,)


@app.cell
def _(
    SimulationParams,
    generate_price_series,
    generate_seasonal_prices,
    get_price_seed,
):
    _params = SimulationParams(seed=get_price_seed())
    seasonal = generate_seasonal_prices(_params)
    prices = generate_price_series(_params)
    price_params = _params
    return price_params, prices, seasonal


@app.cell
def _(
    dataclasses,
    forecast_quality,
    forecast_sigma,
    generate_price_forecast,
    np,
    price_params,
    prices,
):
    _short = forecast_quality.value
    _long = _short + 23
    _fparams = dataclasses.replace(
        price_params,
        forecast_short_term=_short,
        forecast_long_term=_long,
    )
    _forecast_matrix = generate_price_forecast(prices, _fparams)
    price_forecast = _forecast_matrix[0, :]
    price_forecast_sigma = forecast_sigma(np.arange(len(prices)), _fparams)
    return price_forecast, price_forecast_sigma


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
    price_forecast_sigma,
    prices,
    regen_price,
    seasonal,
):
    _days = np.arange(1, len(prices) + 1)

    _fig = go.Figure()

    # Uncertainty band (±1σ around seasonal)
    _fig.add_trace(go.Scatter(
        x=np.concatenate([_days, _days[::-1]]),
        y=np.concatenate([
            seasonal + price_forecast_sigma,
            np.maximum(0, seasonal - price_forecast_sigma)[::-1],
        ]),
        fill="toself",
        fillcolor="rgba(100, 160, 220, 0.15)",
        line=dict(width=0),
        name="Seasonal ±1σ",
        hoverinfo="skip",
    ))

    # Seasonal baseline
    _fig.add_trace(go.Scatter(
        x=_days,
        y=seasonal,
        mode="lines",
        name="Seasonal average",
        line=dict(color="#4a90d9", width=1.5, dash="dash"),
    ))

    # Forecast line
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
        yaxis_title="£/MWh",
        height=400,
        margin=dict(t=50, b=40, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    mo.vstack([regen_price, _fig])
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
