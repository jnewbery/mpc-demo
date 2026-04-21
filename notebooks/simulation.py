import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import sys
    sys.path.insert(0, ".")

    from src.simulation import SimulationParams, generate_price_series, generate_demand_series

    return (
        SimulationParams,
        generate_demand_series,
        generate_price_series,
        go,
        mo,
        np,
    )


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
def _(SimulationParams, generate_price_series, get_price_seed):
    _params = SimulationParams(seed=get_price_seed())
    prices = generate_price_series(_params)
    return (prices,)


@app.cell
def _(SimulationParams, generate_demand_series, get_demand_seed):
    _params = SimulationParams(seed=get_demand_seed())
    demand = generate_demand_series(_params)
    return (demand,)


@app.cell
def _(go, mo, np, prices, regen_price):
    _days = np.arange(1, len(prices) + 1)
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=_days,
        y=prices,
        mode="lines",
        line=dict(color="#e07b39", width=1.5),
    ))
    _fig.update_layout(
        title="Energy Price",
        xaxis_title="Day of year",
        yaxis_title="£/MWh",
        height=350,
        margin=dict(t=50, b=40, l=60, r=20),
    )
    mo.vstack([regen_price, _fig])
    return


@app.cell
def _(demand, go, mo, np, regen_demand):
    _days = np.arange(1, len(demand) + 1)
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=_days,
        y=demand,
        mode="lines",
        line=dict(color="#4a90d9", width=1.5),
    ))
    _fig.update_layout(
        title="Heat Demand",
        xaxis_title="Day of year",
        yaxis_title="MWh/day",
        height=350,
        margin=dict(t=50, b=40, l=60, r=20),
    )
    mo.vstack([regen_demand, _fig])
    return


if __name__ == "__main__":
    app.run()
