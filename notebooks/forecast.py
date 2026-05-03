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
    import datetime
    import pathlib
    import marimo as mo
    import numpy as np
    import polars as pl
    import plotly.graph_objects as go

    from src.simulation import (
        SimulationParams,
        generate_price_forecast,
        generate_raw_price_forecast,
    )

    return (
        SimulationParams,
        dataclasses,
        datetime,
        generate_price_forecast,
        generate_raw_price_forecast,
        go,
        mo,
        np,
        pathlib,
        pl,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Forecasts

    This notebook loads pre-generated **nominal** price and heat demand scenarios
    produced by `scripts/generate_prices.py` and `scripts/generate_heat_demand.py`.

    A nominal scenario is a plausible full-year realisation of prices and heat
    demand, drawn by fitting a Fourier seasonal model and AR(1) residuals to
    historical data.  These scenarios represent the *true* year that unfolds —
    the controller does not observe future values directly and must instead rely
    on forecasts.  How well those forecasts track the nominal scenario determines
    how much value the controller can extract.

    Select a price scenario and a heat demand scenario below.
    """)
    return


@app.cell
def _(mo, pathlib):
    _price_dir = pathlib.Path(__file__).parent.parent / "data" / "price_scenarios"
    _heat_dir = pathlib.Path(__file__).parent.parent / "data" / "heat_demand_scenarios"

    _price_files = sorted(_price_dir.glob("*.csv"), key=lambda f: int("".join(filter(str.isdigit, f.stem)) or 0))
    _heat_files = sorted(_heat_dir.glob("*.csv"), key=lambda f: int("".join(filter(str.isdigit, f.stem)) or 0))

    price_scenario_selector = mo.ui.dropdown(
        options={f.stem.replace("_", " ").title(): str(f) for f in _price_files},
        value=_price_files[0].stem.replace("_", " ").title() if _price_files else None,
        label="Price scenario",
    )
    heat_scenario_selector = mo.ui.dropdown(
        options={f.stem.replace("_", " ").title(): str(f) for f in _heat_files},
        value=_heat_files[0].stem.replace("_", " ").title() if _heat_files else None,
        label="Heat demand scenario",
    )
    mo.hstack([price_scenario_selector, heat_scenario_selector], justify="start")
    return heat_scenario_selector, price_scenario_selector


@app.cell
def _(pl, price_scenario_selector):
    _df = pl.read_csv(price_scenario_selector.value)
    prices = _df.get_column("price").to_numpy()
    seasonal_price = _df.get_column("seasonal").to_numpy()
    return prices, seasonal_price


@app.cell
def _(heat_scenario_selector, pl):
    _df = pl.read_csv(heat_scenario_selector.value)
    heat_demand = _df.get_column("heat_demand").to_numpy()
    return (heat_demand,)


@app.cell
def _(datetime, go, heat_demand, mo, prices, seasonal_price):
    _tick_months = [datetime.date(2001, m, 1) for m in range(1, 13)]
    _tick_doys = [d.timetuple().tm_yday for d in _tick_months]
    _tick_labels = [d.strftime("%b") for d in _tick_months]

    _x = list(range(1, 366))
    _blue = "#1f77b4"
    _red = "#d62728"

    _fig = go.Figure()

    _fig.add_trace(go.Scatter(
        x=_x, y=seasonal_price.tolist(),
        mode="lines", name="Seasonal price",
        line=dict(color=_blue, width=1.5, dash="dash"),
        yaxis="y1",
        hovertemplate="Day %{x}<br>%{y:.1f} €/MWh<extra></extra>",
    ))
    _fig.add_trace(go.Scatter(
        x=_x, y=prices.tolist(),
        mode="lines", name="Price",
        line=dict(color=_blue, width=1.5),
        yaxis="y1",
        hovertemplate="Day %{x}<br>%{y:.1f} €/MWh<extra></extra>",
    ))
    _fig.add_trace(go.Scatter(
        x=_x, y=heat_demand.tolist(),
        mode="lines", name="Heat demand",
        line=dict(color=_red, width=1.5),
        yaxis="y2",
        hovertemplate="Day %{x}<br>%{y:.1f} MWh<extra></extra>",
    ))

    _fig.update_layout(
        title="Nominal price and heat demand scenarios",
        xaxis=dict(
            title="",
            tickvals=_tick_doys, ticktext=_tick_labels,
            showgrid=True, gridcolor="#e5e5e5",
        ),
        yaxis=dict(
            title=dict(text="€/MWh", font=dict(color=_blue)),
            showgrid=True, gridcolor="#e5e5e5",
        ),
        yaxis2=dict(
            title=dict(text="MWh", font=dict(color=_red)),
            overlaying="y", side="right", showgrid=False,
        ),
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60, b=40, l=60, r=60),
        height=500,
    )
    mo.ui.plotly(_fig)


@app.cell
def _(mo):
    mo.md(r"""
    ## Price forecast

    A forecast made at time $t$ for horizon $h$ is constructed in two steps.

    **Step 1 — forecast deviation random walk.**
    The forecaster knows the current deviation from the scenario seasonal exactly,
    then projects it forward with growing uncertainty:

    $$\hat{d}_0 = P_t - S_t, \qquad \hat{d}_h = \hat{d}_{h-1} + \sigma_\text{step}\, z_h$$

    where $\sigma_\text{step} = \sigma_\infty / \sqrt{H_\text{long}}$ is calibrated so
    that forecast uncertainty reaches $\sigma_\infty$ by the long-term horizon $H_\text{long}$.

    **Step 2 — blend toward the seasonal average.**

    $$\hat{P}_{t+h} = S_{t+h} + \alpha(h)\,\hat{d}_h$$

    $$\alpha(h) = \begin{cases} 1 & h \le H_\text{short} \\ \tfrac{1}{2}\!\left(1 + \cos\!\left(\pi\,\tfrac{h - H_\text{short}}{H_\text{long} - H_\text{short}}\right)\right) & H_\text{short} < h < H_\text{long} \\ 0 & h \ge H_\text{long} \end{cases}$$

    For $h \le H_\text{short}$ the forecast closely tracks the true price. For
    $h \ge H_\text{long}$ the weight is zero and the forecast converges to $S_t$.
    The slider below controls $H_\text{short}$ and $H_\text{long}$.
    """)
    return


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
def _(
    SimulationParams,
    dataclasses,
    forecast_horizon,
    generate_price_forecast,
    generate_raw_price_forecast,
    prices,
    seasonal_price,
):
    _short, _long = forecast_horizon.value
    _fparams = dataclasses.replace(
        SimulationParams(T=len(prices)),
        forecast_short_term=_short,
        forecast_long_term=_long,
    )
    _forecast_matrix = generate_price_forecast(prices, _fparams, seasonal_array=seasonal_price)
    price_forecast = _forecast_matrix[0, :]
    price_forecast_raw = generate_raw_price_forecast(prices, _fparams, seasonal_array=seasonal_price)
    return price_forecast, price_forecast_raw


@app.cell
def _(datetime, go, mo, np, price_forecast, price_forecast_raw, prices, seasonal_price):
    _tick_months = [datetime.date(2001, m, 1) for m in range(1, 13)]
    _tick_doys = [d.timetuple().tm_yday for d in _tick_months]
    _tick_labels = [d.strftime("%b") for d in _tick_months]

    _days = np.arange(1, len(prices) + 1)
    _fig2 = go.Figure()

    _fig2.add_trace(go.Scatter(
        x=_days, y=seasonal_price,
        mode="lines", name="Seasonal average",
        line=dict(color="#4a90d9", width=1.5, dash="dash"),
        hovertemplate="Day %{x}<br>%{y:.1f} €/MWh<extra></extra>",
    ))
    _fig2.add_trace(go.Scatter(
        x=_days, y=price_forecast_raw,
        mode="lines", name="Raw forecast (unblended)",
        line=dict(color="grey", width=1, dash="dot"),
        hovertemplate="Day %{x}<br>%{y:.1f} €/MWh<extra></extra>",
    ))
    _fig2.add_trace(go.Scatter(
        x=_days, y=price_forecast,
        mode="lines", name="Forecast",
        line=dict(color="black", width=1.5),
        hovertemplate="Day %{x}<br>%{y:.1f} €/MWh<extra></extra>",
    ))
    _fig2.add_trace(go.Scatter(
        x=_days, y=prices,
        mode="lines", name="True price",
        line=dict(color="#e07b39", width=1.5),
        hovertemplate="Day %{x}<br>%{y:.1f} €/MWh<extra></extra>",
    ))

    _fig2.update_layout(
        title="Price forecast vs nominal scenario",
        xaxis=dict(
            title="",
            tickvals=_tick_doys, ticktext=_tick_labels,
            showgrid=True, gridcolor="#e5e5e5",
        ),
        yaxis=dict(title="€/MWh", showgrid=True, gridcolor="#e5e5e5"),
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60, b=40, l=60, r=20),
        height=450,
    )
    mo.ui.plotly(_fig2)


if __name__ == "__main__":
    app.run()
