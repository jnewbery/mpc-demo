"""
Title: 0. Day-ahead prices
Description: Germany/Luxembourg day-ahead electricity prices (2019–2025) with year selector
"""

import marimo

app = marimo.App(width="full")


@app.cell
def _():
    import datetime
    import marimo as mo
    import polars as pl
    import plotly.graph_objects as go
    return datetime, go, mo, pl


@app.cell
def _(pl):
    import pathlib
    _data_file = pathlib.Path(__file__).parent.parent / "data" / "Day-ahead_prices_201601010000_202601010100_Day.csv"
    price_col = "Germany/Luxembourg [€/MWh] Calculated resolutions"
    df = (
        pl.read_csv(
            _data_file,
            separator=";",
            encoding="utf8-lossy",
            infer_schema=False,
        )
        .with_columns(
            pl.col("Start date").str.strptime(pl.Date, "%b %d, %Y").alias("date"),
            pl.col(price_col).str.replace("^-$", "").cast(pl.Float64, strict=False).alias("price"),
        )
        .with_columns(
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.ordinal_day().alias("day_of_year"),
        )
        .select(["date", "year", "day_of_year", "price"])
    )
    available_years = (
        df.filter(pl.col("price").is_not_null())
        .select("year").unique().sort("year")
        .get_column("year").to_list()
    )
    return available_years, df


@app.cell
def _(available_years, mo):
    year_select = mo.ui.dropdown(
        options={str(y): y for y in available_years},
        value=str(available_years[-2]),
        label="Year",
    )
    return (year_select,)


@app.cell
def _(datetime, df, go, mo, pl, year_select):
    year = year_select.value
    year_df = df.filter(pl.col("year") == year)

    tick_months = [datetime.date(2001, m, 1) for m in range(1, 13)]
    tick_doys = [d.timetuple().tm_yday for d in tick_months]
    tick_labels = [d.strftime("%b") for d in tick_months]

    missing = year_df.filter(pl.col("price").is_null()).height
    has_missing = missing > 0

    dates_str = year_df.select(
        pl.col("date").dt.strftime("%d %b %Y")
    ).get_column("date").to_list()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=year_df.get_column("day_of_year").to_list(),
        y=year_df.get_column("price").to_list(),
        mode="lines",
        line=dict(color="#1f77b4", width=1.5),
        name="DE/LU price",
        hovertemplate="%{customdata}<br>%{y:.2f} €/MWh<extra></extra>",
        customdata=dates_str,
    ))
    fig.update_layout(
        title=f"Germany/Luxembourg Day-ahead Prices — {year}",
        xaxis=dict(
            title="",
            tickvals=tick_doys,
            ticktext=tick_labels,
            showgrid=True,
            gridcolor="#e5e5e5",
        ),
        yaxis=dict(title="€/MWh", showgrid=True, gridcolor="#e5e5e5"),
        plot_bgcolor="white",
        hovermode="x unified",
        margin=dict(t=50, b=40, l=60, r=20),
        height=420,
    )

    warning = (
        mo.callout(mo.md(f"**{missing} days missing** for {year}."), kind="warn")
        if has_missing
        else None
    )

    mo.vstack(
        [year_select, mo.ui.plotly(fig)] + ([warning] if warning else [])
    )


@app.cell
def _(available_years, datetime, df, go, mo, pl):
    _tick_months = [datetime.date(2001, m, 1) for m in range(1, 13)]
    _tick_doys = [d.timetuple().tm_yday for d in _tick_months]
    _tick_labels = [d.strftime("%b") for d in _tick_months]

    # Plotly's default qualitative colour sequence
    _colours = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    _fig = go.Figure()
    for _i, _year in enumerate(available_years):
        _ydf = df.filter(pl.col("year") == _year)
        _dates_str = _ydf.select(
            pl.col("date").dt.strftime("%d %b %Y")
        ).get_column("date").to_list()
        _fig.add_trace(go.Scatter(
            x=_ydf.get_column("day_of_year").to_list(),
            y=_ydf.get_column("price").to_list(),
            mode="lines",
            line=dict(color=_colours[_i % len(_colours)], width=1.2),
            name=str(_year),
            hovertemplate="%{customdata}<br>%{y:.2f} €/MWh<extra></extra>",
            customdata=_dates_str,
        ))
    _fig.update_layout(
        title="Germany/Luxembourg Day-ahead Prices — all years",
        xaxis=dict(
            title="",
            tickvals=_tick_doys,
            ticktext=_tick_labels,
            showgrid=True,
            gridcolor="#e5e5e5",
        ),
        yaxis=dict(title="€/MWh", showgrid=True, gridcolor="#e5e5e5"),
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(title="Year"),
        margin=dict(t=50, b=40, l=60, r=20),
        height=450,
    )
    mo.ui.plotly(_fig)
