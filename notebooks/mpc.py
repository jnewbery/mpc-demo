"""
Title: MPC controller
Description: A rolling-horizon Model Predictive Control (MPC) strategy for the thermal energy storage problem
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
    from plotly.subplots import make_subplots

    from src.simulation import SimulationParams
    from src.storage_lp import StorageParams, solve_perfect_foresight
    from src.mpc import MPCParams, run_mpc

    return (
        MPCParams,
        SimulationParams,
        StorageParams,
        dataclasses,
        datetime,
        go,
        make_subplots,
        mo,
        np,
        pathlib,
        pl,
        run_mpc,
        solve_perfect_foresight,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # MPC vs Perfect Foresight

    A rolling-horizon **Model Predictive Control (MPC)** controller is compared against
    the **perfect foresight** benchmark.

    Both manage the same thermal storage system. The perfect foresight solver
    knows all future prices, while the MPC controller only sees a noisy price
    forecast over a short window.

    At each day $t$ the MPC controller:

    1. Generates a price forecast for days $t, \ldots, t+H-1$ using the blended
       random-walk model from the simulation module
    2. Solves the LP over that $H$-day window, treating demand as known
    3. Applies only the first action $(u^+_t, u^-_t)$ from the solution
    4. Observes the true price and advances the true state of charge

    The gap between MPC cost and perfect foresight cost measures the **value of
    price information**: how much the forecast uncertainty costs relative to knowing
    the future perfectly.
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
def _(mo):
    s_max = mo.ui.number(start=10, stop=10_000, step=10, value=500, label="Storage capacity (MWh)")
    s_min = mo.ui.number(start=0, stop=500, step=10, value=0, label="Min SoC (MWh)")
    s0 = mo.ui.number(start=0, stop=10_000, step=10, value=250, label="Initial SoC (MWh)")
    u_plus_max = mo.ui.number(start=5, stop=10_000, step=5, value=300, label="Max charge rate (MWh/day)")
    u_minus_max = mo.ui.number(start=5, stop=10_000, step=5, value=300, label="Max discharge rate (MWh/day)")
    eta = mo.ui.slider(start=0.5, stop=1.0, step=0.05, value=0.9, label="Discharge efficiency η", show_value=True)
    cop = mo.ui.slider(start=1.0, stop=5.0, step=0.25, value=3.0, label="Heat pump COP", show_value=True)
    h_max = mo.ui.number(start=10, stop=10_000, step=10, value=600, label="Max HP output (MWh/day)")
    mpc_horizon = mo.ui.slider(start=1, stop=90, step=1, value=30, label="MPC horizon H (days)", show_value=True)
    forecast_horizon = mo.ui.range_slider(
        start=0, stop=90, value=[7, 30], step=1,
        label="Forecast horizon: short-term / long-term (days)",
        show_value=True,
    )

    mo.vstack([
        mo.md("### Storage"),
        mo.hstack([s_max, s_min, s0, u_plus_max, u_minus_max], justify="start"),
        mo.md("### Heat pump"),
        mo.hstack([eta, cop, h_max], justify="start"),
        mo.md("### MPC"),
        mo.hstack([mpc_horizon, forecast_horizon], justify="start"),
    ])
    return cop, eta, forecast_horizon, h_max, mpc_horizon, s0, s_max, s_min, u_minus_max, u_plus_max


@app.cell
def _(
    MPCParams,
    SimulationParams,
    StorageParams,
    cop,
    dataclasses,
    eta,
    forecast_horizon,
    h_max,
    heat_scenario_selector,
    mo,
    mpc_horizon,
    pl,
    price_scenario_selector,
    run_mpc,
    s0,
    s_max,
    s_min,
    solve_perfect_foresight,
    u_minus_max,
    u_plus_max,
):
    _price_df = pl.read_csv(price_scenario_selector.value)
    prices = _price_df.get_column("price").to_numpy()
    seasonal_prices = _price_df.get_column("seasonal").to_numpy()
    demand = pl.read_csv(heat_scenario_selector.value).get_column("heat_demand").to_numpy()

    _sp = StorageParams(
        s_max=s_max.value,
        s_min=s_min.value,
        s0=s0.value,
        u_plus_max=u_plus_max.value,
        u_minus_max=u_minus_max.value,
        eta=eta.value,
        cop=cop.value,
        h_max=h_max.value,
    )
    _short, _long = forecast_horizon.value
    _sim = dataclasses.replace(
        SimulationParams(T=len(prices)),
        forecast_short_term=_short,
        forecast_long_term=_long,
    )
    _mp = MPCParams(H=mpc_horizon.value)

    try:
        pf = solve_perfect_foresight(prices, demand, _sp)
        pf_error = None
    except ValueError as e:
        pf = None
        pf_error = str(e)

    if pf_error:
        mo.stop(True, mo.callout(mo.md(f"**Perfect foresight solver error:** {pf_error}"), kind="danger"))

    try:
        mpc = run_mpc(prices, demand, _sp, _mp, _sim, seasonal_prices)
        mpc_error = None
    except Exception as e:
        mpc = None
        mpc_error = str(e)

    if mpc_error:
        mo.stop(True, mo.callout(mo.md(f"**MPC error:** {mpc_error}"), kind="danger"))

    cop_value = cop.value
    return cop_value, demand, mpc, pf, prices


@app.cell
def _(cop_value, datetime, demand, go, make_subplots, mpc, np, pf, prices):
    _tick_months = [datetime.date(2001, m, 1) for m in range(1, 13)]
    _tick_doys = [d.timetuple().tm_yday for d in _tick_months]
    _tick_labels = [d.strftime("%b") for d in _tick_months]

    _days = np.arange(1, len(prices) + 1)
    _colours = {"pf": "#4a90d9", "mpc": "#e07b39", "price": "black", "baseline": "grey"}

    _fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=("Perfect Foresight Dispatch", "MPC Dispatch", "State of Charge", "Cumulative Electricity Cost"),
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
        row_heights=[0.27, 0.27, 0.23, 0.23],
        vertical_spacing=0.07,
    )

    # Rows 1 & 2: Dispatch (PF and MPC)
    for _row, _res, _col in [(1, pf, _colours["pf"]), (2, mpc, _colours["mpc"])]:
        _fig.add_trace(go.Bar(x=_days, y=_res["charge"], name="Charge", marker_color=_col,
                              opacity=0.7, showlegend=(_row == 1)),
                       row=_row, col=1, secondary_y=False)
        _fig.add_trace(go.Bar(x=_days, y=-_res["discharge"], name="Discharge",
                              marker_color=_col, opacity=0.4, showlegend=(_row == 1)),
                       row=_row, col=1, secondary_y=False)
        _fig.add_trace(go.Scatter(x=_days, y=prices, name="Price", mode="lines",
                                  line=dict(color=_colours["price"], width=1),
                                  showlegend=(_row == 1),
                                  hovertemplate="Day %{x}<br>%{y:.1f} €/MWh<extra></extra>"),
                       row=_row, col=1, secondary_y=True)

    # Row 3: State of charge (legend2)
    _fig.add_trace(go.Scatter(x=_days, y=pf["soc"], name="Perfect foresight",
                              legend="legend2", mode="lines",
                              line=dict(color=_colours["pf"], width=1.5),
                              hovertemplate="Day %{x}<br>%{y:.1f} MWh<extra></extra>"),
                   row=3, col=1)
    _fig.add_trace(go.Scatter(x=_days, y=mpc["soc"], name="MPC",
                              legend="legend2", mode="lines",
                              line=dict(color=_colours["mpc"], width=1.5),
                              hovertemplate="Day %{x}<br>%{y:.1f} MWh<extra></extra>"),
                   row=3, col=1)

    # Row 4: Cumulative cost (legend3)
    _baseline_daily = prices * demand / cop_value
    _pf_daily = prices * pf["hp_output"] / cop_value
    _mpc_daily = prices * mpc["hp_output"] / cop_value
    _fig.add_trace(go.Scatter(x=_days, y=np.cumsum(_baseline_daily), name="Baseline (no storage)",
                              legend="legend3", mode="lines",
                              line=dict(color=_colours["baseline"], width=1.5, dash="dash"),
                              hovertemplate="Day %{x}<br>%{y:,.0f} €<extra></extra>"),
                   row=4, col=1)
    _fig.add_trace(go.Scatter(x=_days, y=np.cumsum(_pf_daily), name="Perfect foresight",
                              legend="legend3", mode="lines",
                              line=dict(color=_colours["pf"], width=1.5),
                              hovertemplate="Day %{x}<br>%{y:,.0f} €<extra></extra>"),
                   row=4, col=1)
    _fig.add_trace(go.Scatter(x=_days, y=np.cumsum(_mpc_daily), name="MPC",
                              legend="legend3", mode="lines",
                              line=dict(color=_colours["mpc"], width=1.5),
                              hovertemplate="Day %{x}<br>%{y:,.0f} €<extra></extra>"),
                   row=4, col=1)

    # row_heights=[0.27,0.27,0.23,0.23], vspacing=0.07 → row tops: 1.0, 0.717, 0.433, 0.182
    _legend_style = dict(orientation="h", xanchor="left", x=0.01,
                         bgcolor="rgba(255,255,255,0.8)", bordercolor="#e5e5e5", borderwidth=1)
    _fig.update_xaxes(tickvals=_tick_doys, ticktext=_tick_labels, showgrid=True, gridcolor="#e5e5e5")
    _fig.update_yaxes(showgrid=True, gridcolor="#e5e5e5")
    _fig.update_yaxes(title_text="MWh/day", secondary_y=False)
    _fig.update_yaxes(title_text="€/MWh", secondary_y=True, showgrid=False)
    _fig.update_yaxes(title_text="MWh", row=3, col=1)
    _fig.update_yaxes(title_text="€", row=4, col=1)
    _fig.update_layout(
        barmode="relative",
        plot_bgcolor="white",
        height=1300,
        margin=dict(t=60, b=40, l=60, r=60),
        legend=dict(**_legend_style, yanchor="top", y=0.98),
        legend2=dict(**_legend_style, yanchor="top", y=0.42),
        legend3=dict(**_legend_style, yanchor="top", y=0.17),
    )
    _fig


@app.cell
def _(cop_value, demand, mo, mpc, np, pf, prices):
    _baseline = float(np.dot(prices, demand) / cop_value)
    _pf_saving = _baseline - pf["cost"]
    _mpc_saving = _baseline - mpc["cost"]
    _info_gap = mpc["cost"] - pf["cost"]

    mo.hstack([
        mo.stat(value=f"€{_baseline:,.0f}", label="Baseline (no storage)"),
        mo.stat(value=f"€{pf['cost']:,.0f}  ({100 * _pf_saving / _baseline:.1f}% saved)", label="Perfect foresight"),
        mo.stat(value=f"€{mpc['cost']:,.0f}  ({100 * _mpc_saving / _baseline:.1f}% saved)", label="MPC"),
        mo.stat(value=f"€{_info_gap:,.0f}", label="Cost of price uncertainty"),
        mo.stat(value=str(mpc["n_fallbacks"]), label="MPC fallbacks"),
    ], justify="start")
    return


if __name__ == "__main__":
    app.run()
