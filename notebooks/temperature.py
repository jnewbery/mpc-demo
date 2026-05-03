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
    import numpy as np
    import polars as pl
    import plotly.graph_objects as go
    from src.fourier import fourier_seasonal_fit
    from src.simulation import ar1_fit, ar1_simulate
    return ar1_fit, ar1_simulate, datetime, fourier_seasonal_fit, go, mo, np, pl


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


@app.cell
def _(mo):
    mo.md(r"""
    ## Seasonal fit

    A **Fourier regression** is fitted to the selected years to extract the average
    seasonal temperature curve.  The model has the form

    $$
    T(d) = a_0 + \sum_{k=1}^{K} \bigl[ a_k \cos\!\tfrac{2\pi k d}{365.25} + b_k \sin\!\tfrac{2\pi k d}{365.25} \bigr]
    $$

    where $d$ is the day-of-year and $K = 3$ harmonics are used by default.
    The regression is fitted by ordinary least squares across all selected years
    simultaneously.  Years can be added or removed using the selector below.
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

    After removing the seasonal trend, the residuals $r_t = T_t - \hat{T}(d_t)$ are
    modelled as an **AR(1)** autoregressive process:

    $$r_t = \varphi \, r_{t-1} + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \sigma^2)$$

    $\varphi$ is estimated by ordinary least squares on consecutive within-year
    residual pairs (year boundaries are never crossed).  $\sigma$ is the standard
    deviation of the one-step innovations $\varepsilon_t = r_t - \hat\varphi r_{t-1}$.

    The stationary standard deviation $\sigma_\text{stat} = \sigma / \sqrt{1 - \varphi^2}$
    gives the long-run spread of temperatures around the seasonal mean.
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

    The seasonal curve and AR(1) parameters estimated above define a generative model
    for a typical temperature year.  A scenario is drawn by:

    1. Sampling the initial residual from the stationary distribution:
       $r_0 \sim \mathcal{N}\!\left(0,\, \sigma^2/(1-\varphi^2)\right)$
    2. Iterating the AR(1) for 365 days:
       $r_t = \varphi \, r_{t-1} + \varepsilon_t$
    3. Adding the seasonal mean:
       $T_t = \hat{T}(t) + r_t$

    The shaded region shows ±1σ and ±2σ around the seasonal mean (darker = higher
    probability density).  Press **Generate new year** to draw a new scenario.
    """)
    return


@app.cell
def _(mo):
    generate_temp_btn = mo.ui.run_button(label="Generate new year")
    return (generate_temp_btn,)


@app.cell
def _(ar1_simulate, datetime, generate_temp_btn, go, mo, np, phi_temp, seasonal_temp, sigma_stat_temp, sigma_temp):
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

    _r = ar1_simulate(phi_temp, sigma_temp)
    _sim_temp = (seasonal_temp + _r).tolist()

    _fig4.add_trace(go.Scatter(
        x=_x, y=_sim_temp,
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
    | $k$ | adjustable (kW/K) | Aggregate fabric heat-loss coefficient |
    | $P_{\text{base}} = 0.1\,k$ | kWh/day | Baseload (hot water, always-on) |

    A typical UK semi-detached house has $k \approx 0.15$–$0.25$ kW/K; a
    neighbourhood of 100 homes is roughly $k = 15$–$25$ kW/K.
    """)
    return


@app.cell
def _(mo):
    heat_k = mo.ui.slider(
        start=1, stop=100, step=1, value=20,
        label="Thermal loss coefficient  k  (kW/K)",
        show_value=True,
    )
    return (heat_k,)


@app.cell
def _(available_temp_years, datetime, df_temp, go, heat_k, mo, pl):
    _T_BASE = 20
    _T_LIMIT = 15
    _k = heat_k.value
    _P_BASE = _k * 0.1

    _df_demand = (
        df_temp
        .with_columns(
            pl.when(pl.col("tmk") < _T_LIMIT)
            .then(_k * (_T_BASE - pl.col("tmk")) + _P_BASE)
            .otherwise(_P_BASE)
            .alias("q_kwh")
        )
        .drop_nulls("q_kwh")
    )

    _colours = [
        "31,119,180", "255,127,14", "44,160,44", "214,39,40",
        "148,103,189", "140,86,75", "227,119,194", "127,127,127",
        "188,189,34", "23,190,207",
    ]

    _tick_months = [datetime.date(2001, m, 1) for m in range(1, 13)]
    _tick_doys = [d.timetuple().tm_yday for d in _tick_months]
    _tick_labels = [d.strftime("%b") for d in _tick_months]

    _fig_demand = go.Figure()
    for _i, _year in enumerate(available_temp_years):
        _c = _colours[_i % len(_colours)]
        _yr = _df_demand.filter(pl.col("year") == _year)
        _fig_demand.add_trace(go.Scatter(
            x=_yr["day_of_year"].to_list(),
            y=_yr["q_kwh"].to_list(),
            mode="lines",
            line=dict(color=f"rgba({_c},0.7)", width=1.2),
            name=str(_year),
            hovertemplate=f"{_year} — day %{{x}}<br>%{{y:.0f}} kWh/day<extra></extra>",
        ))

    _fig_demand.update_layout(
        title=f"Daily heat demand — all years  (k = {_k} kW/K)",
        xaxis=dict(
            title="",
            tickvals=_tick_doys, ticktext=_tick_labels,
            showgrid=True, gridcolor="#e5e5e5",
        ),
        yaxis=dict(title="kWh/day", showgrid=True, gridcolor="#e5e5e5"),
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=60, r=20),
        height=400,
    )

    mo.vstack([heat_k, mo.ui.plotly(_fig_demand)])


@app.cell
def _(available_temp_years, df_temp, go, heat_k, mo, pl):
    _T_BASE = 20
    _T_LIMIT = 15
    _k = heat_k.value
    _P_BASE = _k * 0.1

    _df_sc = (
        df_temp
        .with_columns(
            pl.when(pl.col("tmk") < _T_LIMIT)
            .then(_k * (_T_BASE - pl.col("tmk")) + _P_BASE)
            .otherwise(_P_BASE)
            .alias("q_kwh")
        )
        .drop_nulls("q_kwh")
    )

    _colours = [
        "31,119,180", "255,127,14", "44,160,44", "214,39,40",
        "148,103,189", "140,86,75", "227,119,194", "127,127,127",
        "188,189,34", "23,190,207",
    ]

    _fig_sc = go.Figure()
    for _i, _year in enumerate(available_temp_years):
        _c = _colours[_i % len(_colours)]
        _yr = _df_sc.filter(pl.col("year") == _year)
        _fig_sc.add_trace(go.Scatter(
            x=_yr["tmk"].to_list(),
            y=_yr["q_kwh"].to_list(),
            mode="markers",
            marker=dict(color=f"rgba({_c},0.55)", size=4),
            name=str(_year),
            hovertemplate=f"{_year} — %{{x:.1f}} °C → %{{y:.0f}} kWh/day<extra></extra>",
        ))

    _fig_sc.add_vline(
        x=_T_LIMIT,
        line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dash"),
        annotation_text=f"T_limit = {_T_LIMIT} °C",
        annotation_position="top right",
    )

    _fig_sc.update_layout(
        title=f"Temperature vs heat demand — all years  (k = {_k} kW/K)",
        xaxis=dict(title="Daily mean temperature (°C)", showgrid=True, gridcolor="#e5e5e5"),
        yaxis=dict(title="kWh/day", showgrid=True, gridcolor="#e5e5e5"),
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=60, r=20),
        height=400,
    )

    mo.ui.plotly(_fig_sc)


@app.cell
def _(df_temp, heat_k, mo, pl):
    _T_BASE = 20
    _T_LIMIT = 15
    _k = heat_k.value
    _P_BASE = _k * 0.1

    _df_s = (
        df_temp
        .with_columns(
            pl.when(pl.col("tmk") < _T_LIMIT)
            .then(_k * (_T_BASE - pl.col("tmk")) + _P_BASE)
            .otherwise(_P_BASE)
            .alias("q_kwh")
        )
        .drop_nulls("q_kwh")
    )

    _annual = (
        _df_s
        .group_by("year")
        .agg(pl.col("q_kwh").sum().alias("total_kwh"))
        .sort("year")
    )

    _peak_row = _df_s.sort("q_kwh", descending=True).row(0, named=True)
    _peak_kwh = _peak_row["q_kwh"]
    _peak_date = _peak_row["date"].strftime("%d %b %Y")

    mo.vstack([
        mo.md("**Annual heat demand**"),
        mo.hstack([
            mo.stat(f"{row['total_kwh'] / 1000:,.0f} MWh", label=str(row["year"]))
            for row in _annual.iter_rows(named=True)
        ]),
        mo.stat(f"{_peak_kwh:.0f} kWh/day", label=f"Peak daily demand ({_peak_date})"),
    ])
