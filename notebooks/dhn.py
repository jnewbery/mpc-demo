"""
Title: 0. District Heating Networks
Description: District heating networks (DHNs) in Saxony, sourced from RWTH Aachen University
"""

import marimo

app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import plotly.graph_objects as go
    return go, mo, np, pl


@app.cell
def _(mo):
    mo.md(r"""
    # District Heating Networks — Saxony

    District heating networks (DHNs) Saxony, adapted from
    building heat demand data by RWTH-EBC as part of the
    [AixDHN](https://github.com/RWTH-EBC/AixDHN) project.
    The dataset covers 434 network areas across 251 municipalities.
    """)
    return


@app.cell
def _(pl):
    import pathlib
    _data_file = pathlib.Path(__file__).parent.parent / "data" / "dhn" / "heatgrids_Sachsen.csv"
    df = pl.read_csv(_data_file)
    return (df,)


@app.cell
def _(df, mo):
    mo.hstack([
        mo.stat(str(len(df)), label="DHNs"),
        mo.stat(str(df["GEN"].n_unique()), label="municipalities"),
        mo.stat(f"{df['DH_demand'].sum() / 1e6:.2f} TWh/year", label="total DH demand"),
        mo.stat(f"{int(df['DH_supplied_households'].sum()):,}", label="households supplied"),
    ])


@app.cell
def _(mo):
    mo.md(r"""
    ## DH demand vs households supplied

    Each point is one DHN.  Both axes use a log scale to accommodate the
    wide range — from tiny village clusters (< 10 households) to large city
    networks (> 100,000 households).
    """)
    return


@app.cell
def _(df, go, mo):
    _x = df["DH_supplied_households"].to_list()
    _y = df["DH_demand"].to_list()
    _hover = [
        f"{row['GEN']} — {row['ID']}<br>"
        f"DH demand: {row['DH_demand']:,.0f} MWh/yr<br>"
        f"Households: {int(row['DH_supplied_households']):,}<br>"
        f"DH share: {row['share_DH']:.1f}%<br>"
        f"Heat density: {row['heat_density_DH']:.2f} GWh/yr/km²"
        for row in df.iter_rows(named=True)
    ]

    _fig1 = go.Figure(go.Scatter(
        x=_x, y=_y,
        mode="markers",
        marker=dict(
            colorscale="Viridis",
            size=6,
            opacity=0.7,
            line=dict(width=0),
        ),
        text=_hover,
        hovertemplate="%{text}<extra></extra>",
    ))

    _fig1.update_layout(
        title="DH annual demand vs households supplied",
        xaxis=dict(
            title="Households supplied",
            type="log", showgrid=True, gridcolor="#e5e5e5",
        ),
        yaxis=dict(
            title="DH demand (MWh/year)",
            type="log", showgrid=True, gridcolor="#e5e5e5",
        ),
        plot_bgcolor="white",
        margin=dict(t=60, b=50, l=70, r=20),
        height=450,
    )

    mo.ui.plotly(_fig1)


@app.cell
def _(mo):
    mo.md(r"""
    ## Heat density vs DHN market share

    **Heat density** (DHN demand per km²) measures how concentrated the load is.
    **DH share** is the fraction of total area heat demand met by district heating.
    Networks in the top-right corner are both dense and dominant.
    Bubble size is proportional to DH demand.
    """)
    return


@app.cell
def _(df, go, mo, np):
    _x = df["share_DH"].to_list()
    _y = df["heat_density_DH"].to_list()

    _raw = np.sqrt(df["DH_demand"].to_numpy())
    _sizes = (4 + 36 * (_raw - _raw.min()) / (_raw.max() - _raw.min())).tolist()

    _hover2 = [
        f"{row['GEN']} — {row['ID']}<br>"
        f"DH share: {row['share_DH']:.1f}%<br>"
        f"Heat density: {row['heat_density_DH']:.2f} GWh/yr/km²<br>"
        f"DH demand: {row['DH_demand']:,.0f} MWh/yr<br>"
        f"Area: {row['area']:.3f} km²"
        for row in df.iter_rows(named=True)
    ]

    _fig2 = go.Figure(go.Scatter(
        x=_x, y=_y,
        mode="markers",
        marker=dict(
            size=_sizes,
            color="rgba(31,119,180,0.45)",
            line=dict(color="rgba(31,119,180,0.8)", width=0.8),
        ),
        text=_hover2,
        hovertemplate="%{text}<extra></extra>",
    ))

    _fig2.update_layout(
        title="Heat density vs DH market share  (bubble size ∝ DH demand)",
        xaxis=dict(title="DH share (%)", showgrid=True, gridcolor="#e5e5e5"),
        yaxis=dict(title="Heat density (GWh/yr/km²)", showgrid=True, gridcolor="#e5e5e5"),
        plot_bgcolor="white",
        margin=dict(t=60, b=50, l=70, r=20),
        height=450,
    )

    mo.ui.plotly(_fig2)


@app.cell
def _(mo):
    mo.md(r"""
    ## Top municipalities by DHN demand

    Aggregated across all DHNs within each municipality.
    """)
    return


@app.cell
def _(df, go, mo, pl):
    TOP_N = 20
    _agg = (
        df.group_by("GEN")
        .agg(
            pl.col("DH_demand").sum().alias("DH_demand_MWh"),
            pl.col("DH_supplied_households").sum().alias("households"),
            pl.len().alias("n_networks"),
        )
        .sort("DH_demand_MWh", descending=True)
        .head(TOP_N)
        .sort("DH_demand_MWh")
    )

    _fig3 = go.Figure()
    _fig3.add_trace(go.Bar(
        y=_agg["GEN"].to_list(),
        x=(_agg["DH_demand_MWh"] / 1000).to_list(),
        orientation="h",
        marker_color="rgba(44,160,44,0.75)",
        customdata=list(zip(
            [int(v) for v in _agg["households"].to_list()],
            _agg["n_networks"].to_list(),
        )),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "DH demand: %{x:,.0f} GWh/yr<br>"
            "Households: %{customdata[0]:,}<br>"
            "Networks: %{customdata[1]}<extra></extra>"
        ),
    ))

    _fig3.update_layout(
        title=f"Top {TOP_N} municipalities by total DH demand",
        xaxis=dict(title="DH demand (GWh/year)", showgrid=True, gridcolor="#e5e5e5"),
        yaxis=dict(showgrid=False),
        plot_bgcolor="white",
        margin=dict(t=60, b=50, l=120, r=20),
        height=max(300, TOP_N * 28),
    )

    mo.ui.plotly(_fig3)


@app.cell
def _(mo):
    mo.md(r"""
    ## Distribution of DH market share

    DH share is the fraction of total heat demand in the network area that is met
    by district heating.
    """)
    return


@app.cell
def _(df, go, mo):
    _fig4 = go.Figure(go.Histogram(
        x=df["share_DH"].to_list(),
        nbinsx=40,
        marker_color="rgba(214,39,40,0.65)",
        marker_line=dict(color="rgba(214,39,40,0.9)", width=0.5),
        hovertemplate="Share: %{x:.0f}–%{x:.0f}%<br>Count: %{y}<extra></extra>",
    ))

    _fig4.update_layout(
        title="Distribution of DH market share across all 434 networks",
        xaxis=dict(title="DH share (%)", showgrid=True, gridcolor="#e5e5e5"),
        yaxis=dict(title="Number of networks", showgrid=True, gridcolor="#e5e5e5"),
        plot_bgcolor="white",
        bargap=0.05,
        margin=dict(t=60, b=50, l=60, r=20),
        height=380,
    )

    mo.ui.plotly(_fig4)
