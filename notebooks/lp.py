"""
Title: LP model
Description: A perfect foresight benchmark for the thermal energy storage problem, solved with a Linear Programme (LP)
"""

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import datetime
    import pathlib
    import marimo as mo
    import numpy as np
    import polars as pl
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from src.storage_lp import StorageParams, solve_perfect_foresight

    return StorageParams, datetime, go, make_subplots, mo, np, pathlib, pl, solve_perfect_foresight


@app.cell
def _(mo):
    mo.md(r"""
    # Perfect Foresight LP

    A **thermal storage system** connected to a **heat pump**
    is optimised over a full year using a Linear Programme (LP) with perfect knowledge
    of future prices and heat demand.  The LP decides each day how much heat to store or
    retrieve to minimise the amount spent on electricity (ie maximising heat production
    on low-price days so storage covers demand on expensive days).

    ---

    ## Notation

    | Symbol | Unit | Meaning |
    |--------|------|---------|
    | $t$ | day | Time index, $t = 0, \ldots, T-1$ |
    | $p_t$ | €/MWh | Spot electricity price on day $t$ |
    | $D_t$ | MWh/day | Heat demand on day $t$ |
    | $\text{hp}_t$ | MWh/day | Heat pump thermal output on day $t$ |
    | $u^+_t$ | MWh/day | Heat charged into storage on day $t$ |
    | $u^-_t$ | MWh/day | Heat discharged from storage on day $t$ |
    | $s_t$ | MWh | Stored heat (state of charge) at start of day $t$ |
    | $\text{COP}$ | — | Heat pump coefficient of performance: MWh heat per MWh electricity |
    | $\eta$ | — | Round-trip discharge efficiency (heat recovered per MWh withdrawn) |
    | $h_{\max}$ | MWh/day | Maximum heat pump output (grid / equipment cap) |

    ---

    ## Physical model

    The heat pump is the sole heat source. Each day its output must exactly cover
    demand plus any net change in storage:

    \begin{equation}
        \text{hp}_t \;=\; D_t + u^+_t - u^-_t \tag{1}
    \end{equation}

    Because the heat pump converts electricity to heat with efficiency $\text{COP}$,
    the electricity consumed on day $t$ is $\text{hp}_t / \text{COP}$, and the cost is

    \begin{equation}
        c_t \;=\; p_t \cdot \frac{\text{hp}_t}{\text{COP}} \tag{2}
    \end{equation}

    There are round-trip losses when storing and retrieving heat. For simplicity we
    treat these as constant, disregarding self-discharge when heat is stored for multiple
    days.

    ---

    ## Optimisation problem

    Choose $u^+_t$ and $u^-_t$ for every day to minimise total annual electricity spend:

    \begin{equation}
        \min_{u^+, u^-} \;\sum_{t=0}^{T-1} \frac{p_t}{\text{COP}}
        \!\left( D_t + u^+_t - u^-_t \right) \tag{3}
    \end{equation}

    Since $D_t$ is fixed, this is equivalent to shifting electricity consumption away
    from high-price days (charge storage cheaply, discharge on expensive days).

    ### Constraints

    **Storage dynamics** — heat stored tomorrow equals heat stored today plus net flow,
    with a discharge efficiency penalty $\eta < 1$ (some heat is lost when retrieving):

    \begin{equation}
        s_{t+1} \;=\; s_t + u^+_t - \frac{u^-_t}{\eta}
        \qquad \forall\, t = 0, \ldots, T-2 \tag{4}
    \end{equation}

    **Capacity condition** — storage must stay within physical bounds:

    \begin{equation}
        s_{\min} \;\le\; s_t \;\le\; s_{\max} \tag{5}
    \end{equation}

    $s_0$ is given as an initial condition, and $s_T$ is unconstrained (end-of-year storage can be anything).

    **Charge and discharge rate limits** — the pipework and pump impose daily flow limits:

    \begin{equation}
        0 \;\le\; u^+_t \;\le\; u^+_{\max}, \qquad
        0 \;\le\; u^-_t \;\le\; u^-_{\max} \tag{6}
    \end{equation}

    **Heat demand condition** — combining (1) with the non-negativity and grid cap requirements:

    \begin{equation}
        0 \;\le\; D_t + u^+_t - u^-_t \;\le\; h_{\max} \tag{7}
    \end{equation}

    Together, equations (3)–(7) form a Linear Programme (all variables appear linearly),
    which is solved to global optimality by an LP solver.
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

    mo.vstack([
        mo.md("### Storage"),
        mo.hstack([s_max, s_min, s0], justify="start"),
        mo.hstack([u_plus_max, u_minus_max, eta], justify="start"),
        mo.md("### Heat pump"),
        mo.hstack([cop, h_max], justify="start"),
    ])
    return cop, eta, h_max, s0, s_max, s_min, u_minus_max, u_plus_max


@app.cell
def _(
    StorageParams,
    cop,
    eta,
    h_max,
    heat_scenario_selector,
    mo,
    pl,
    price_scenario_selector,
    s0,
    s_max,
    s_min,
    solve_perfect_foresight,
    u_minus_max,
    u_plus_max,
):
    prices = pl.read_csv(price_scenario_selector.value).get_column("price").to_numpy()
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

    try:
        result = solve_perfect_foresight(prices, demand, _sp)
        solve_error = None
    except ValueError as e:
        result = None
        solve_error = str(e)

    if solve_error:
        mo.stop(True, mo.callout(mo.md(f"**Solver error:** {solve_error}"), kind="danger"))

    cop_value = cop.value
    return cop_value, demand, prices, result


@app.cell
def _(cop_value, datetime, demand, go, make_subplots, np, prices, result):
    _tick_months = [datetime.date(2001, m, 1) for m in range(1, 13)]
    _tick_doys = [d.timetuple().tm_yday for d in _tick_months]
    _tick_labels = [d.strftime("%b") for d in _tick_months]

    _days = np.arange(1, len(prices) + 1)

    _fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=("Dispatch Schedule", "Heat Demand", "State of Charge", "Cumulative Electricity Cost"),
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
        row_heights=[0.31, 0.23, 0.23, 0.23],
        vertical_spacing=0.06,
    )

    # Row 1: Dispatch
    _fig.add_trace(go.Bar(
        x=_days, y=result["charge"],
        name="Charge", marker_color="#4a90d9", opacity=0.7,
    ), row=1, col=1, secondary_y=False)
    _fig.add_trace(go.Bar(
        x=_days, y=-result["discharge"],
        name="Discharge", marker_color="#e07b39", opacity=0.7,
    ), row=1, col=1, secondary_y=False)
    _fig.add_trace(go.Scatter(
        x=_days, y=prices, name="Price", mode="lines",
        line=dict(color="black", width=1.5),
        hovertemplate="Day %{x}<br>%{y:.1f} €/MWh<extra></extra>",
    ), row=1, col=1, secondary_y=True)

    # Row 2: Heat demand
    _fig.add_trace(go.Scatter(
        x=_days, y=demand, name="Heat demand", mode="lines",
        line=dict(color="#d62728", width=1.5),
        fill="tozeroy", fillcolor="rgba(214,39,40,0.1)",
        showlegend=False,
        hovertemplate="Day %{x}<br>%{y:.1f} MWh/day<extra></extra>",
    ), row=2, col=1)

    # Row 3: State of charge
    _fig.add_trace(go.Scatter(
        x=_days, y=result["soc"], name="State of charge", mode="lines",
        line=dict(color="#4a90d9", width=1.5),
        fill="tozeroy", fillcolor="rgba(74,144,217,0.15)",
        showlegend=False,
        hovertemplate="Day %{x}<br>%{y:.1f} MWh<extra></extra>",
    ), row=3, col=1)

    # Row 4: Cumulative cost (assigned to legend2 so it sits inside that subplot)
    _baseline_daily = prices * demand / cop_value
    _pf_daily = prices * result["hp_output"] / cop_value
    _fig.add_trace(go.Scatter(
        x=_days, y=np.cumsum(_baseline_daily),
        name="Baseline (no storage)", mode="lines",
        legend="legend2",
        line=dict(color="grey", width=1.5, dash="dash"),
        hovertemplate="Day %{x}<br>%{y:,.0f} €<extra></extra>",
    ), row=4, col=1)
    _fig.add_trace(go.Scatter(
        x=_days, y=np.cumsum(_pf_daily),
        name="Perfect foresight", mode="lines",
        legend="legend2",
        line=dict(color="#2a9d5c", width=1.5),
        hovertemplate="Day %{x}<br>%{y:,.0f} €<extra></extra>",
    ), row=4, col=1)

    # row_heights=[0.31,0.23,0.23,0.23], vspacing=0.06 → row tops: 1.0, 0.686, 0.437, 0.189
    _legend_style = dict(orientation="h", xanchor="left", x=0.01,
                         bgcolor="rgba(255,255,255,0.8)", bordercolor="#e5e5e5", borderwidth=1)
    _fig.update_xaxes(tickvals=_tick_doys, ticktext=_tick_labels, showgrid=True, gridcolor="#e5e5e5")
    _fig.update_yaxes(showgrid=True, gridcolor="#e5e5e5")
    _fig.update_yaxes(title_text="MWh/day", zeroline=True, row=1, col=1, secondary_y=False)
    _fig.update_yaxes(title_text="€/MWh", showgrid=False, row=1, col=1, secondary_y=True)
    _fig.update_yaxes(title_text="MWh/day", row=2, col=1)
    _fig.update_yaxes(title_text="MWh", row=3, col=1)
    _fig.update_yaxes(title_text="€", row=4, col=1)
    _fig.update_layout(
        barmode="relative",
        plot_bgcolor="white",
        height=1250,
        margin=dict(t=60, b=40, l=60, r=60),
        legend=dict(**_legend_style, yanchor="top", y=0.98),
        legend2=dict(**_legend_style, yanchor="top", y=0.18),
    )
    _fig


@app.cell
def _(cop_value, demand, mo, np, prices, result):
    _baseline_cost = float(np.dot(prices, demand) / cop_value)
    _saving = _baseline_cost - result["cost"]
    _pct = 100 * _saving / _baseline_cost

    mo.hstack([
        mo.stat(value=f"€{_baseline_cost:,.0f}", label="Baseline cost"),
        mo.stat(value=f"€{result['cost']:,.0f}", label="Perfect foresight cost"),
        mo.stat(value=f"€{_saving:,.0f} ({_pct:.1f}%)", label="Saving"),
        mo.stat(value=result["status"], label="Solver status"),
    ], justify="start")


if __name__ == "__main__":
    app.run()
