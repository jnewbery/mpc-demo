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

    return (
        SimulationParams,
        generate_demand_series,
        generate_price_series,
        go,
        make_subplots,
        mo,
        np,
    )


@app.cell
def _(mo, np):
    get_seed, set_seed = mo.state(42)
    regenerate = mo.ui.run_button(
        label="Regenerate",
        on_change=lambda _: set_seed(int(np.random.randint(0, 100_000))),
    )
    return get_seed, regenerate


@app.cell
def _(regenerate):
    regenerate
    return


@app.cell
def _(
    SimulationParams,
    generate_demand_series,
    generate_price_series,
    get_seed,
    mo,
):
    _seed = get_seed()
    _params = SimulationParams(seed=_seed)
    _prices = generate_price_series(_params)
    _demand = generate_demand_series(_params)
    mo.output.replace(mo.md(f"Seed: **{_seed}**"))
    prices = _prices
    demand = _demand
    return demand, prices


@app.cell
def _(demand, go, make_subplots, np, prices):
    _days = np.arange(1, len(prices) + 1)

    _fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Energy Price", "Heat Demand"),
        vertical_spacing=0.10,
    )

    _fig.add_trace(
        go.Scatter(
            x=_days,
            y=prices,
            mode="lines",
            name="Price",
            line=dict(color="#e07b39", width=1.5),
        ),
        row=1,
        col=1,
    )

    _fig.add_trace(
        go.Scatter(
            x=_days,
            y=demand,
            mode="lines",
            name="Heat demand",
            line=dict(color="#4a90d9", width=1.5),
        ),
        row=2,
        col=1,
    )

    _fig.update_yaxes(title_text="£/MWh", row=1, col=1)
    _fig.update_yaxes(title_text="MWh/day", row=2, col=1)
    _fig.update_xaxes(title_text="Day of year", row=2, col=1)

    _fig.update_layout(
        height=550,
        showlegend=False,
        margin=dict(t=50, b=40, l=60, r=20),
    )

    _fig
    return


if __name__ == "__main__":
    app.run()
