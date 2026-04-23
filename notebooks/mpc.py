import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import sys
    sys.path.insert(0, ".")

    from src.simulation import SimulationParams, generate_price_series, generate_demand_series
    from src.storage_lp import StorageParams, solve_perfect_foresight
    from src.mpc import MPCParams, run_mpc

    return (
        MPCParams,
        SimulationParams,
        StorageParams,
        generate_demand_series,
        generate_price_series,
        go,
        make_subplots,
        mo,
        np,
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
def _(mo):
    sim_seed = mo.ui.number(start=0, stop=99999, step=1, value=42, label="Seed")
    s_max = mo.ui.number(start=10, stop=10000, step=10, value=100, label="Storage capacity (MWh)")
    s_min = mo.ui.number(start=0, stop=50, step=5, value=0, label="Min SoC (MWh)")
    s0 = mo.ui.number(start=0, stop=200, step=5, value=50, label="Initial SoC (MWh)")
    u_plus_max = mo.ui.number(start=5, stop=10000, step=5, value=50, label="Max charge rate (MWh/day)")
    u_minus_max = mo.ui.number(start=5, stop=10000, step=5, value=50, label="Max discharge rate (MWh/day)")
    eta = mo.ui.slider(start=0.5, stop=1.0, step=0.05, value=0.9, label="Discharge efficiency η", show_value=True)
    cop = mo.ui.slider(start=1.0, stop=5.0, step=0.25, value=3.0, label="Heat pump COP", show_value=True)
    h_max = mo.ui.number(start=10, stop=300, step=10, value=100, label="Max HP output (MWh/day)")
    mpc_horizon = mo.ui.slider(start=1, stop=90, step=1, value=30, label="MPC horizon H (days)", show_value=True)

    mo.vstack([
        mo.md("### Simulation"),
        mo.hstack([sim_seed], justify="start"),
        mo.md("### Storage"),
        mo.hstack([s_max, s_min, s0, u_plus_max, u_minus_max], justify="start"),
        mo.md("### Heat pump"),
        mo.hstack([eta, cop, h_max], justify="start"),
        mo.md("### MPC"),
        mo.hstack([mpc_horizon], justify="start"),
    ])
    return (
        cop,
        eta,
        h_max,
        mpc_horizon,
        s0,
        s_max,
        s_min,
        sim_seed,
        u_minus_max,
        u_plus_max,
    )


@app.cell
def _(
    MPCParams,
    SimulationParams,
    StorageParams,
    cop,
    eta,
    generate_demand_series,
    generate_price_series,
    h_max,
    mo,
    mpc_horizon,
    run_mpc,
    s0,
    s_max,
    s_min,
    sim_seed,
    solve_perfect_foresight,
    u_minus_max,
    u_plus_max,
):
    _sim = SimulationParams(seed=sim_seed.value)
    prices = generate_price_series(_sim)
    demand = generate_demand_series(_sim)

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
        mpc = run_mpc(prices, demand, _sp, _mp, _sim)
        mpc_error = None
    except Exception as e:
        mpc = None
        mpc_error = str(e)

    if mpc_error:
        mo.stop(True, mo.callout(mo.md(f"**MPC error:** {mpc_error}"), kind="danger"))

    cop_value = cop.value
    return cop_value, demand, mpc, pf, prices


@app.cell
def _(cop_value, demand, go, make_subplots, mo, mpc, np, pf, prices):
    _days = np.arange(1, len(prices) + 1)
    _colours = {"pf": "#4a90d9", "mpc": "#e07b39", "price": "black", "baseline": "grey"}

    # --- Dispatch comparison (two subplots stacked, each with secondary y for price) ---
    _fig1 = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Perfect Foresight Dispatch", "MPC Dispatch"),
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
        vertical_spacing=0.12,
    )
    for _row, _res, _col in [(1, pf, _colours["pf"]), (2, mpc, _colours["mpc"])]:
        _fig1.add_trace(go.Bar(x=_days, y=_res["charge"], name="Charge", marker_color=_col,
                               opacity=0.7, showlegend=(_row == 1)),
                        row=_row, col=1, secondary_y=False)
        _fig1.add_trace(go.Bar(x=_days, y=-_res["discharge"], name="Discharge",
                               marker_color=_col, opacity=0.4, showlegend=(_row == 1)),
                        row=_row, col=1, secondary_y=False)
        _fig1.add_trace(go.Scatter(x=_days, y=prices, name="Price", mode="lines",
                                   line=dict(color=_colours["price"], width=1),
                                   showlegend=(_row == 1)),
                        row=_row, col=1, secondary_y=True)
    _fig1.update_yaxes(title_text="MWh/day", secondary_y=False)
    _fig1.update_yaxes(title_text="£/MWh", secondary_y=True, showgrid=False)
    _fig1.update_layout(
        height=550, barmode="relative",
        margin=dict(t=60, b=40, l=60, r=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
    )

    # --- SoC comparison ---
    _fig2 = go.Figure()
    _fig2.add_trace(go.Scatter(x=_days, y=pf["soc"], name="Perfect foresight",
                               mode="lines", line=dict(color=_colours["pf"], width=1.5)))
    _fig2.add_trace(go.Scatter(x=_days, y=mpc["soc"], name="MPC",
                               mode="lines", line=dict(color=_colours["mpc"], width=1.5)))
    _fig2.update_layout(
        title="State of Charge",
        xaxis_title="Day of year", yaxis_title="MWh",
        height=300, margin=dict(t=50, b=40, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    # --- Cumulative cost ---
    _baseline_daily = prices * demand / cop_value
    _pf_daily = prices * pf["hp_output"] / cop_value
    _mpc_daily = prices * mpc["hp_output"] / cop_value
    _fig3 = go.Figure()
    _fig3.add_trace(go.Scatter(x=_days, y=np.cumsum(_baseline_daily), name="Baseline (no storage)",
                               mode="lines", line=dict(color=_colours["baseline"], width=1.5, dash="dash")))
    _fig3.add_trace(go.Scatter(x=_days, y=np.cumsum(_pf_daily), name="Perfect foresight",
                               mode="lines", line=dict(color=_colours["pf"], width=1.5)))
    _fig3.add_trace(go.Scatter(x=_days, y=np.cumsum(_mpc_daily), name="MPC",
                               mode="lines", line=dict(color=_colours["mpc"], width=1.5)))
    _fig3.update_layout(
        title="Cumulative Electricity Cost",
        xaxis_title="Day of year", yaxis_title="£",
        height=300, margin=dict(t=50, b=40, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    mo.vstack([_fig1, _fig2, _fig3])
    return


@app.cell
def _(cop_value, demand, mo, mpc, np, pf, prices):
    _baseline = float(np.dot(prices, demand) / cop_value)
    _pf_saving = _baseline - pf["cost"]
    _mpc_saving = _baseline - mpc["cost"]
    _info_gap = mpc["cost"] - pf["cost"]

    mo.hstack([
        mo.stat(value=f"£{_baseline:,.0f}", label="Baseline (no storage)"),
        mo.stat(value=f"£{pf['cost']:,.0f}  ({100 * _pf_saving / _baseline:.1f}% saved)", label="Perfect foresight"),
        mo.stat(value=f"£{mpc['cost']:,.0f}  ({100 * _mpc_saving / _baseline:.1f}% saved)", label="MPC"),
        mo.stat(value=f"£{_info_gap:,.0f}", label="Cost of price uncertainty"),
        mo.stat(value=str(mpc["n_fallbacks"]), label="MPC fallbacks"),
    ], justify="start")
    return


if __name__ == "__main__":
    app.run()
