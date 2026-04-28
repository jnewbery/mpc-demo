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
    return datetime, go, mo, np, pl


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
def _(available_temp_years, datetime, df_temp, go, mo, np, pl, temp_k_harmonics, temp_regression_years):
    K_temp = temp_k_harmonics.value

    _all = df_temp.filter(
        pl.col("tmk").is_not_null() & pl.col("year").is_in(temp_regression_years.value)
    )
    _doys = _all.get_column("day_of_year").to_numpy().astype(float)
    _tmks = _all.get_column("tmk").to_numpy()

    def _fourier_features(doys, k):
        cols = [np.ones(len(doys))]
        for _k in range(1, k + 1):
            cols.append(np.cos(2 * np.pi * _k * doys / 365.25))
            cols.append(np.sin(2 * np.pi * _k * doys / 365.25))
        return np.column_stack(cols)

    _coeffs, _, _, _ = np.linalg.lstsq(_fourier_features(_doys, K_temp), _tmks, rcond=None)

    _doy_range = np.arange(1, 366, dtype=float)
    seasonal_temp = _fourier_features(_doy_range, K_temp) @ _coeffs

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
def _(df_temp, np, pl, seasonal_temp, temp_regression_years):
    _included = temp_regression_years.value
    _year_residuals_temp = []

    for _year in _included:
        _ydf = (
            df_temp.filter(pl.col("year") == _year)
            .sort("day_of_year")
            .filter(pl.col("tmk").is_not_null())
        )
        _doys = np.clip(_ydf.get_column("day_of_year").to_numpy() - 1, 0, 364)
        _resid = _ydf.get_column("tmk").to_numpy() - seasonal_temp[_doys]
        _year_residuals_temp.append(_resid)

    _r0 = np.concatenate([r[:-1] for r in _year_residuals_temp])
    _r1 = np.concatenate([r[1:] for r in _year_residuals_temp])

    phi_temp = float(np.dot(_r0, _r1) / np.dot(_r0, _r0))
    sigma_temp = float(np.std(_r1 - phi_temp * _r0, ddof=1))
    sigma_stat_temp = float(np.std(np.concatenate(_year_residuals_temp), ddof=1))
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
def _(datetime, generate_temp_btn, go, mo, np, phi_temp, seasonal_temp, sigma_stat_temp, sigma_temp):
    _tick_months = [datetime.date(2001, m, 1) for m in range(1, 13)]
    _tick_doys = [d.timetuple().tm_yday for d in _tick_months]
    _tick_labels = [d.strftime("%b") for d in _tick_months]

    _x = list(range(1, 366))
    _s = sigma_stat_temp
    _colour = "214,39,40"  # red in RGB

    _N = 30
    _step = 2 * _s / _N
    _max_alpha = 0.55

    _fig4 = go.Figure()

    for _i in range(_N, 0, -1):
        _alpha = (_N - _i + 1) / _N * _max_alpha
        _outer = _i * _step
        _inner = (_i - 1) * _step
        _fig4.add_trace(go.Scatter(
            x=_x + _x[::-1],
            y=(seasonal_temp + _outer).tolist() + (seasonal_temp + _inner).tolist()[::-1],
            fill="toself", fillcolor=f"rgba({_colour},{_alpha:.3f})",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        _fig4.add_trace(go.Scatter(
            x=_x + _x[::-1],
            y=(seasonal_temp - _inner).tolist() + (seasonal_temp - _outer).tolist()[::-1],
            fill="toself", fillcolor=f"rgba({_colour},{_alpha:.3f})",
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

    _rng = np.random.default_rng()
    _r = np.empty(365)
    _r[0] = _rng.normal(0, sigma_stat_temp)
    for _t in range(1, 365):
        _r[_t] = phi_temp * _r[_t - 1] + _rng.normal(0, sigma_temp)
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
