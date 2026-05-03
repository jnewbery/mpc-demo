"""
Title: Heat Demand
Description: Görlitz DHN heat demand model: temperature data, seasonal fit, AR(1) residuals, and degree-day calibration
"""

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="full")


@app.cell
def _():
    import datetime
    import pathlib
    import marimo as mo
    import numpy as np
    import polars as pl
    import plotly.graph_objects as go
    from src.fourier import fourier_seasonal_fit
    from src.heat_demand import calibrate, compute_demand
    from src.simulation import ar1_fit, ar1_simulate
    return ar1_fit, ar1_simulate, calibrate, compute_demand, datetime, fourier_seasonal_fit, go, mo, np, pathlib, pl


@app.cell
def _(mo):
    mo.md(r"""
    # Daily Mean Temperatures — Görlitz

    Historical daily mean air temperatures (°C) for the Görlitz weather station
    (DWD station ID 01684), sourced from the **Deutscher Wetterdienst (DWD)** open
    data portal:
    [opendata.dwd.de — daily climate observations, historical](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical/).
    The column used is `TMK` (daily mean temperature at 2 m height).
    """)
    return


@app.cell
def _(pathlib, pl):
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
def _(pathlib, pl):
    _dhn_file = pathlib.Path(__file__).parent.parent / "data" / "dhn" / "heatgrids_Sachsen.csv"
    goerlitz_dh_demand_mwh = float(
        pl.read_csv(_dhn_file)
        .filter(pl.col("GEN") == "Görlitz")
        .select(pl.col("DH_demand").sum())
        .item()
    )
    return (goerlitz_dh_demand_mwh,)


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


@app.cell
def _(mo):
    mo.md(r"""
    ## Seasonal fit

    A Fourier regression is fitted to the selected years — see the
    [Prices](/notebooks/prices) notebook for a full description of the method.
    $K = 3$ harmonics are used by default.
    """)
    return


@app.cell
def _(available_temp_years, mo):
    temp_regression_years = mo.ui.multiselect(
        options={str(y): y for y in available_temp_years},
        value=[str(y) for y in available_temp_years],
        label="Years included in regression",
    )
    temp_k_harmonics = mo.ui.number(start=1, stop=20, step=1, value=3, label="Harmonics K")
    return temp_k_harmonics, temp_regression_years


@app.cell
def _(available_temp_years, datetime, df_temp, fourier_seasonal_fit, go, mo, np, pl, temp_k_harmonics, temp_regression_years):
    K_temp = temp_k_harmonics.value

    _all = df_temp.filter(
        pl.col("tmk").is_not_null() & pl.col("year").is_in(temp_regression_years.value)
    )
    _doys = _all.get_column("day_of_year").to_numpy().astype(float)
    _tmks = _all.get_column("tmk").to_numpy()

    _doy_range = np.arange(1, 366, dtype=float)
    seasonal_temp = fourier_seasonal_fit(_doys, _tmks, K_temp)

    _tick_months = [datetime.date(2001, m, 1) for m in range(1, 13)]
    _tick_doys = [d.timetuple().tm_yday for d in _tick_months]
    _tick_labels = [d.strftime("%b") for d in _tick_months]

    _colours = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    _included = set(temp_regression_years.value)
    _fig2 = go.Figure()
    for _i, _year in enumerate(available_temp_years):
        _ydf = df_temp.filter(pl.col("year") == _year)
        _in = _year in _included
        _fig2.add_trace(go.Scatter(
            x=_ydf.get_column("day_of_year").to_list(),
            y=_ydf.get_column("tmk").to_list(),
            mode="lines",
            line=dict(color=_colours[_i % len(_colours)], width=1, dash="solid" if _in else "dot"),
            name=str(_year),
            opacity=0.45 if _in else 0.2,
            hovertemplate=f"{_year} day %{{x}}<br>%{{y:.1f}} °C<extra></extra>",
        ))
    _fig2.add_trace(go.Scatter(
        x=_doy_range.tolist(),
        y=seasonal_temp.tolist(),
        mode="lines",
        line=dict(color="black", width=2.5),
        name="Seasonal fit",
        hovertemplate="Seasonal day %{x}<br>%{y:.1f} °C<extra></extra>",
    ))
    _fig2.add_hline(y=0, line=dict(color="rgba(0,0,0,0.25)", width=1, dash="dot"))
    _fig2.update_layout(
        title=f"Görlitz — Fourier seasonal fit (K={K_temp} harmonics)",
        xaxis=dict(
            title="",
            tickvals=_tick_doys, ticktext=_tick_labels,
            showgrid=True, gridcolor="#e5e5e5",
        ),
        yaxis=dict(title="°C", showgrid=True, gridcolor="#e5e5e5"),
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(title="Year"),
        margin=dict(t=50, b=40, l=60, r=20),
        height=450,
    )
    mo.vstack([mo.hstack([temp_regression_years, temp_k_harmonics], justify="start"), mo.ui.plotly(_fig2)])


@app.cell
def _(datetime, go, mo, seasonal_temp):
    _tick_months = [datetime.date(2001, m, 1) for m in range(1, 13)]
    _tick_doys = [d.timetuple().tm_yday for d in _tick_months]
    _tick_labels = [d.strftime("%b") for d in _tick_months]

    _fig3 = go.Figure()
    _fig3.add_trace(go.Scatter(
        x=list(range(1, 366)),
        y=seasonal_temp.tolist(),
        mode="lines",
        line=dict(color="#d62728", width=2.5),
        hovertemplate="Day %{x}<br>%{y:.1f} °C<extra></extra>",
        showlegend=False,
    ))
    _fig3.add_hline(y=0, line=dict(color="rgba(0,0,0,0.25)", width=1, dash="dot"))
    _fig3.update_layout(
        title="Fourier seasonal fit — temperature",
        xaxis=dict(
            title="",
            tickvals=_tick_doys, ticktext=_tick_labels,
            showgrid=True, gridcolor="#e5e5e5",
        ),
        yaxis=dict(title="°C", showgrid=True, gridcolor="#e5e5e5"),
        plot_bgcolor="white",
        hovermode="x unified",
        margin=dict(t=50, b=40, l=60, r=20),
        height=300,
    )
    mo.ui.plotly(_fig3)


@app.cell
def _(mo):
    mo.md(r"""
    ## AR(1) parameter estimation

    The detrended residuals are modelled as an AR(1) process — see the
    [Prices](/notebooks/prices) notebook for a full description.
    """)
    return


@app.cell
def _(ar1_fit, df_temp, np, pl, seasonal_temp, temp_regression_years):
    _included = temp_regression_years.value
    _year_residuals_temp = []

    for _year in _included:
        _ydf = (
            df_temp.filter(pl.col("year") == _year)
            .sort("day_of_year")
            .filter(pl.col("tmk").is_not_null())
        )
        _doys = np.clip(_ydf.get_column("day_of_year").to_numpy() - 1, 0, 364)
        _year_residuals_temp.append(_ydf.get_column("tmk").to_numpy() - seasonal_temp[_doys])

    phi_temp, sigma_temp, sigma_stat_temp = ar1_fit(_year_residuals_temp)
    n_years_temp = len(_included)
    n_obs_temp = sum(len(r) for r in _year_residuals_temp)
    return n_obs_temp, n_years_temp, phi_temp, sigma_stat_temp, sigma_temp


@app.cell
def _(mo, n_obs_temp, n_years_temp, phi_temp, sigma_stat_temp, sigma_temp):
    mo.vstack([
        mo.md(
            f"Estimated from **{n_years_temp} years** of detrended daily temperatures "
            f"({n_obs_temp} observations)."
        ),
        mo.hstack([
            mo.stat(f"{phi_temp:.3f}", label="φ  (AR(1) autocorrelation)"),
            mo.stat(f"{sigma_temp:.2f} °C", label="σ  (innovation std dev)"),
            mo.stat(f"{sigma_stat_temp:.2f} °C", label="σ_stationary  (residual std dev)"),
        ]),
    ])


@app.cell
def _(mo):
    mo.md(r"""
    ## Nominal temperature year

    The seasonal curve and AR(1) parameters define a generative model for a
    typical temperature year — see the [Prices](/notebooks/prices) notebook for
    a full description. The shaded bands show ±1σ and ±2σ around the seasonal mean.
    Press **Generate new year** to draw a new scenario.
    """)
    return


@app.cell
def _(mo):
    generate_temp_btn = mo.ui.run_button(label="Generate new year")
    return (generate_temp_btn,)


@app.cell
def _(ar1_simulate, generate_temp_btn, phi_temp, seasonal_temp, sigma_temp):
    generate_temp_btn  # re-run on each button press
    _r = ar1_simulate(phi_temp, sigma_temp)
    sim_temp = (seasonal_temp + _r).tolist()
    return (sim_temp,)


@app.cell
def _(datetime, generate_temp_btn, go, mo, np, seasonal_temp, sigma_stat_temp, sim_temp):
    _tick_months = [datetime.date(2001, m, 1) for m in range(1, 13)]
    _tick_doys = [d.timetuple().tm_yday for d in _tick_months]
    _tick_labels = [d.strftime("%b") for d in _tick_months]

    _x = list(range(1, 366))
    _s = sigma_stat_temp
    _colour = "214,39,40"  # red in RGB

    _N = 30
    _step = 2 * _s / _N
    _band_alpha = 0.04

    _fig4 = go.Figure()

    for _i in range(_N, 0, -1):
        _hw = _i * _step
        _fig4.add_trace(go.Scatter(
            x=_x + _x[::-1],
            y=(seasonal_temp + _hw).tolist() + (seasonal_temp - _hw).tolist()[::-1],
            fill="toself", fillcolor=f"rgba({_colour},{_band_alpha:.3f})",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))

    for _n, _label in [(1, "±1σ"), (2, "±2σ")]:
        for _sign, _show in [(1, True), (-1, False)]:
            _fig4.add_trace(go.Scatter(
                x=_x, y=(seasonal_temp + _sign * _n * _s).tolist(),
                mode="lines",
                line=dict(color=f"rgba({_colour},0.7)", width=1, dash="dash"),
                name=_label, showlegend=_show,
                hovertemplate=f"{_label} %{{y:.1f}} °C<extra></extra>",
            ))

    _fig4.add_trace(go.Scatter(
        x=_x, y=seasonal_temp.tolist(),
        mode="lines", line=dict(color=f"rgb({_colour})", width=2.5),
        name="Seasonal mean",
        hovertemplate="Day %{x}<br>%{y:.1f} °C<extra></extra>",
    ))

    _fig4.add_trace(go.Scatter(
        x=_x, y=sim_temp,
        mode="lines", line=dict(color="rgba(50,50,50,0.7)", width=1.5),
        name="Simulated year",
        hovertemplate="Day %{x}<br>%{y:.1f} °C<extra></extra>",
    ))

    _fig4.add_hline(y=0, line=dict(color="rgba(0,0,0,0.2)", width=1, dash="dot"))

    _fig4.update_layout(
        title="Average seasonal temperature with uncertainty bands",
        xaxis=dict(
            title="",
            tickvals=_tick_doys, ticktext=_tick_labels,
            showgrid=True, gridcolor="#e5e5e5",
        ),
        yaxis=dict(
            title="°C",
            showgrid=True, gridcolor="#e5e5e5",
            range=[
                float(np.min(seasonal_temp - 2 * _s)) - 3,
                float(np.max(seasonal_temp + 2 * _s)) + 3,
            ],
        ),
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=60, r=20),
        height=400,
    )
    mo.vstack([generate_temp_btn, mo.ui.plotly(_fig4)])


@app.cell
def _(calibrate, goerlitz_dh_demand_mwh, seasonal_temp):
    k_goerlitz, p_base_goerlitz = calibrate(seasonal_temp, goerlitz_dh_demand_mwh)
    return k_goerlitz, p_base_goerlitz


@app.cell
def _(mo):
    mo.md(r"""
    ## Heat demand model

    A **piecewise linear heat demand model** maps daily mean temperature to
    aggregate thermal load — the standard *degree-day* formulation used in
    energy planning:

    $$
    Q(T) =
    \begin{cases}
        k\,(T_{\text{base}} - T_{\text{ext}}) + P_{\text{base}} & T_{\text{ext}} < T_{\text{limit}} \\
        P_{\text{base}} & T_{\text{ext}} \ge T_{\text{limit}}
    \end{cases}
    $$

    | Parameter | Value | Meaning |
    |-----------|-------|---------|
    | $T_{\text{base}}$ | 20 °C | Indoor target temperature |
    | $T_{\text{limit}}$ | 15 °C | Heating cut-off (no space heating above this) |
    | $P_{\text{base}}$ | MWh/day | Baseload (hot water, always-on) |
    | $k$ | MW/K | Aggregate fabric heat-loss coefficient |

    Both $P_{\text{base}}$ and $k$ parameters are derived from the Görlitz annual DH demand (Saxony heatgrid dataset,
    [RWTH-EBC](https://data.mendeley.com/datasets/k3ry9p944k)):
    - $P_{\text{base}}$ is set to
    20 % of annual demand spread evenly over 365 days
    - $k$ is back-calculated from the
    remaining 80 % (space heating) and the seasonal temperature curve.
    """)
    return


@app.cell
def _(goerlitz_dh_demand_mwh, k_goerlitz, mo, p_base_goerlitz):
    mo.hstack([
        mo.stat(f"{goerlitz_dh_demand_mwh:,.0f} MWh/yr", label="Görlitz DH demand"),
        mo.stat(f"{k_goerlitz:.2f} MW/K", label="Calibrated k"),
        mo.stat(f"{p_base_goerlitz:.1f} MWh/day", label="Baseload P_base"),
    ])


@app.cell
def _(compute_demand, datetime, go, k_goerlitz, mo, np, p_base_goerlitz, sim_temp):
    _x = list(range(1, 366))
    _q = compute_demand(np.array(sim_temp), k_goerlitz, p_base_goerlitz).tolist()

    _tick_months = [datetime.date(2001, m, 1) for m in range(1, 13)]
    _tick_doys = [d.timetuple().tm_yday for d in _tick_months]
    _tick_labels = [d.strftime("%b") for d in _tick_months]

    _fig_demand = go.Figure()
    _fig_demand.add_trace(go.Scatter(
        x=_x, y=_q,
        mode="lines",
        line=dict(color="rgba(214,39,40,0.8)", width=1.5),
        name="Heat demand",
        hovertemplate="Day %{x}<br>%{y:.1f} MWh<extra></extra>",
    ))

    _fig_demand.update_layout(
        title="Görlitz DHN — nominal year heat demand",
        xaxis=dict(
            title="",
            tickvals=_tick_doys, ticktext=_tick_labels,
            showgrid=True, gridcolor="#e5e5e5",
        ),
        yaxis=dict(title="MWh", showgrid=True, gridcolor="#e5e5e5", rangemode="tozero"),
        plot_bgcolor="white",
        hovermode="x unified",
        margin=dict(t=60, b=40, l=60, r=20),
        height=400,
    )

    mo.ui.plotly(_fig_demand)


if __name__ == "__main__":
    app.run()
