"""
Title: 2. Temperature
Description: Daily mean temperatures at Görlitz (DWD station 01684), 2015–2024
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
def _(mo):
    mo.md(r"""
    # Daily Mean Temperatures — Görlitz

    Historical daily mean air temperatures (°C) for the Görlitz weather station
    (DWD station ID 01684), sourced from the **Deutscher Wetterdienst (DWD)** open
    data portal:
    [opendata.dwd.de — daily climate observations, historical](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical/)

    The column used is `TMK` (daily mean temperature at 2 m height).
    """)
    return


@app.cell
def _(pl):
    import pathlib
    _data_file = pathlib.Path(__file__).parent.parent / "data" / "goerlitz_daily_weather.txt.gz"
    df_temp = (
        pl.read_csv(
            _data_file,
            separator=";",
            encoding="utf8-lossy",
            infer_schema=False,
        )
        .with_columns(
            pl.col("MESS_DATUM").str.strip_chars().str.strptime(pl.Date, "%Y%m%d").alias("date"),
            pl.col(" TMK").str.strip_chars().cast(pl.Float64, strict=False).alias("tmk"),
        )
        .with_columns(
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.ordinal_day().alias("day_of_year"),
        )
        .with_columns(
            pl.when(pl.col("tmk") <= -990).then(None).otherwise(pl.col("tmk")).alias("tmk"),
        )
        .filter(pl.col("year") >= 2015)
        .select(["date", "year", "day_of_year", "tmk"])
    )
    available_temp_years = sorted(df_temp["year"].unique().to_list())
    return available_temp_years, df_temp


@app.cell
def _(mo):
    mo.md(r"""
    ## Historical temperatures

    Each year is plotted as a separate trace against day-of-year so seasonal
    patterns are directly comparable.  Years can be toggled by clicking the legend.
    """)
    return


@app.cell
def _(available_temp_years, datetime, df_temp, go, mo, pl):
    _tick_months = [datetime.date(2001, m, 1) for m in range(1, 13)]
    _tick_doys = [d.timetuple().tm_yday for d in _tick_months]
    _tick_labels = [d.strftime("%b") for d in _tick_months]

    _colours = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    _fig = go.Figure()
    for _i, _year in enumerate(available_temp_years):
        _ydf = df_temp.filter(pl.col("year") == _year)
        _dates_str = _ydf.select(
            pl.col("date").dt.strftime("%d %b %Y")
        ).get_column("date").to_list()
        _fig.add_trace(go.Scatter(
            x=_ydf.get_column("day_of_year").to_list(),
            y=_ydf.get_column("tmk").to_list(),
            mode="lines",
            line=dict(color=_colours[_i % len(_colours)], width=1.2),
            name=str(_year),
            hovertemplate="%{customdata}<br>%{y:.1f} °C<extra></extra>",
            customdata=_dates_str,
        ))

    # Zero-degree reference line
    _fig.add_hline(y=0, line=dict(color="rgba(0,0,0,0.25)", width=1, dash="dot"))

    _fig.update_layout(
        title="Görlitz Daily Mean Temperature — all years",
        xaxis=dict(
            title="",
            tickvals=_tick_doys,
            ticktext=_tick_labels,
            showgrid=True,
            gridcolor="#e5e5e5",
        ),
        yaxis=dict(title="°C", showgrid=True, gridcolor="#e5e5e5"),
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(title="Year"),
        margin=dict(t=50, b=40, l=60, r=20),
        height=450,
    )
    mo.ui.plotly(_fig)
