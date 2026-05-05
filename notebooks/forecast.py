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

    from src.forecast import (
        SimulationParams,
        get_forecast_window,
        get_raw_forecast_window,
    )

    return (
        SimulationParams,
        dataclasses,
        datetime,
        get_forecast_window,
        get_raw_forecast_window,
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

    This notebook loads pre-generated nominal price and heat demand scenarios
    produced by `scripts/generate_prices.py` and `scripts/generate_heat_demand.py`,
    and then constructs forecasts from the perspective of the controller.

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
    seasonal_heat_demand = _df.get_column("seasonal_heat_demand").to_numpy()
    return heat_demand, seasonal_heat_demand


@app.cell
def _(
    datetime,
    go,
    heat_demand,
    mo,
    prices,
    seasonal_heat_demand,
    seasonal_price,
):
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
        x=_x, y=seasonal_heat_demand.tolist(),
        mode="lines", name="Seasonal heat demand",
        line=dict(color=_red, width=1.5, dash="dash"),
        yaxis="y2",
        hovertemplate="Day %{x}<br>%{y:.1f} MWh<extra></extra>",
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
            domain=[0.05, 0.95],
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
        legend=dict(orientation="h", yanchor="bottom", y=0.98, xanchor="left", x=0),
        margin=dict(t=60, b=40, l=60, r=60),
        height=500,
    )
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Price forecast

    A forecast made at time $t$ for horizon $h$ is constructed in two steps.

    **Step 1 — AR(1) error generation.**
    Starting from zero error at $h=0$ (the current price is known exactly), errors
    evolve as an AR(1) process:

    $$E_0 = 0, \qquad E_h = \rho\, E_{h-1} + \varepsilon_h, \quad \varepsilon_h \sim \mathcal{N}(0,\,\sigma^2)$$

    The raw forecast is $F_\text{raw}(h) = P_\text{nominal}(t+h) + E_h$.

    **Step 2 — linear blend toward the seasonal mean.**

    $$w(h) = \max\!\left(0,\; 1 - \tfrac{h}{H_\text{blend}}\right)$$

    $$F_\text{final}(h) = w(h)\cdot F_\text{raw}(h) + \bigl(1 - w(h)\bigr)\cdot S_{t+h}$$

    At $h=0$ the forecast equals the true price; by $h = H_\text{blend}$ it has
    fully reverted to the seasonal mean $S$.
    The slider below controls $H_\text{blend}$.
    """)
    return


@app.cell
def _(mo):
    blend_horizon = mo.ui.slider(
        start=1, stop=90, value=30, step=1,
        label="Blend horizon (days)",
        show_value=True,
    )
    controller_day = mo.ui.slider(
        start=1, stop=365, value=1, step=1,
        label="Controller day of year",
        show_value=True,
    )
    mo.hstack([controller_day, blend_horizon], justify="start")
    return blend_horizon, controller_day


@app.cell
def _(
    SimulationParams,
    blend_horizon,
    controller_day,
    dataclasses,
    get_forecast_window,
    get_raw_forecast_window,
    prices,
    seasonal_price,
):
    _t = controller_day.value - 1
    _H = blend_horizon.value
    _fparams = dataclasses.replace(
        SimulationParams(T=len(prices)),
        blend_horizon=_H,
    )
    price_forecast = get_forecast_window(_t, prices, seasonal_price, _fparams)
    price_forecast_raw = get_raw_forecast_window(_t, prices, _fparams)
    return price_forecast, price_forecast_raw


@app.cell
def _(
    blend_horizon,
    controller_day,
    datetime,
    go,
    np,
    price_forecast,
    price_forecast_raw,
    prices,
    seasonal_price,
):
    _t = controller_day.value - 1
    _H = blend_horizon.value

    _tick_months = [datetime.date(2001, m, 1) for m in range(1, 13)]
    _tick_doys = [d.timetuple().tm_yday for d in _tick_months]
    _tick_labels = [d.strftime("%b") for d in _tick_months]

    # Clip forecast so it doesn't extend past day 365
    _H_clip  = min(_H, len(prices) - _t)
    _raw_fc_days = np.arange(_t + 1, _t + _H_clip + 1)  # controller_day … min(controller_day+H-1, 365)
    _blended_fc_days = np.arange(_t + 1, len(prices) + 1)  # controller_day … 365 (blended forecast extends to end of year)

    _fig2 = go.Figure()

    _fig2.add_trace(go.Scatter(
        x=np.arange(1, len(prices) + 1), y=seasonal_price,
        mode="lines", name="Seasonal average",
        line=dict(color="#4a90d9", width=1.5, dash="dash"),
        hovertemplate="Day %{x}<br>%{y:.1f} €/MWh<extra></extra>",
    ))
    _fig2.add_trace(go.Scatter(
        x=np.arange(_t + 1, len(prices) + 1), y=prices[_t:],
        mode="lines", name="True price",
        line=dict(color="#e07b39", width=1.5),
        hovertemplate="Day %{x}<br>%{y:.1f} €/MWh<extra></extra>",
    ))
    _fig2.add_trace(go.Scatter(
        x=_raw_fc_days, y=price_forecast_raw[:_H_clip],
        mode="lines", name="Raw forecast (unblended)",
        line=dict(color="grey", width=1, dash="dot"),
        hovertemplate="Day %{x}<br>%{y:.1f} €/MWh<extra></extra>",
    ))
    _fig2.add_trace(go.Scatter(
        x=_blended_fc_days, y=price_forecast,
        mode="lines", name="Forecast",
        line=dict(color="black", width=1.5),
        hovertemplate="Day %{x}<br>%{y:.1f} €/MWh<extra></extra>",
    ))
    _fig2.add_trace(go.Scatter(
        x=np.arange(1, _t + 1), y=prices,
        mode="lines", name="Historic",
        line=dict(color="black", width=1.5),
        hovertemplate="Day %{x}<br>%{y:.1f} €/MWh<extra></extra>",
    ))

    _fig2.update_layout(
        title="Price forecast vs nominal scenario",
        xaxis=dict(
            title="",
            tickvals=_tick_doys, ticktext=_tick_labels,
            showgrid=True, gridcolor="#e5e5e5",
            domain=[0.05, 0.95],
        ),
        yaxis=dict(title="€/MWh", showgrid=True, gridcolor="#e5e5e5", range=[0, max(prices) * 1.1]),
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=0.98, xanchor="left", x=0),
        margin=dict(t=60, b=40, l=60, r=20),
        height=450,
    )
    mo.ui.plotly(_fig2)
    return


if __name__ == "__main__":
    app.run()
