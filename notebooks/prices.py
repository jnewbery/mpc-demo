"""
Title: Day-ahead prices
Description: Germany/Luxembourg day-ahead electricity prices (2019–2025) with fourier regression and AR(1) residual model
"""

import marimo

__generated_with = "0.19.4"
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
    # Germany/Luxembourg Day-ahead Electricity Prices

    Historical day-ahead electricity prices for the Germany/Luxembourg bidding zone,
    sourced from **SMARD** (Bundesnetzagentur), the German energy market data platform:
    [smard.de/en/downloadcenter/download-market-data](https://www.smard.de/en/downloadcenter/download-market-data/)

    Prices are daily averages in €/MWh covering October 1st 2019 - December
    31st 2025. Dates before October 2018 are excluded because SMARD did not
    report Germany/Luxembourg as a separate bidding zone for that period.
    """)
    return


@app.cell
def _(pl):
    import pathlib
    _data_file = pathlib.Path(__file__).parent.parent / "data" / "day_ahead_prices.csv"
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
        df.filter(pl.col("price").is_not_null() & (pl.col("year") < 2026))
        .select("year").unique().sort("year")
        .get_column("year").to_list()
    )
    return available_years, df



@app.cell
def _(mo):
    mo.md(r"""
    ## Historical prices

    Each year is plotted as a separate trace against day-of-year so seasonal
    patterns are directly comparable.  Years can be toggled by clicking the legend.
    """)
    return


@app.cell
def _(available_years, datetime, df, go, mo, np, pl):
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


@app.cell
def _(mo):
    mo.md(r"""
    ## Seasonal fit

    A **Fourier regression** is fitted to the selected years to extract the average
    seasonal price curve.  The model has the form

    $$
    p(d) = a_0 + \sum_{k=1}^{K} \bigl[ a_k \cos\!\tfrac{2\pi k d}{365.25} + b_k \sin\!\tfrac{2\pi k d}{365.25} \bigr]
    $$

    where $d$ is the day-of-year.  $K = 3$ harmonics are used by default; this can be adjusted with the
    selector below.  The regression is fitted by ordinary least squares across all
    selected years simultaneously.

    2022 is excluded by default since that year's prices were elevated by the gas supply
    shock following the Russian invasion of Ukraine. Years can be added or removed using the
    selector below.
    """)
    return


@app.cell
def _(available_years, mo):
    regression_years = mo.ui.multiselect(
        options={str(y): y for y in available_years},
        value=[str(y) for y in available_years if y != 2022],
        label="Years included in regression",
    )
    k_harmonics = mo.ui.number(start=1, stop=50, step=1, value=3, label="Harmonics K")
    return k_harmonics, regression_years


@app.cell
def _(available_years, datetime, df, fourier_seasonal_fit, go, k_harmonics, mo, np, pl, regression_years):
    K = k_harmonics.value

    _all = df.filter(
        pl.col("price").is_not_null() & pl.col("year").is_in(regression_years.value)
    )
    _doys = _all.get_column("day_of_year").to_numpy().astype(float)
    _prices = _all.get_column("price").to_numpy()

    _doy_range = np.arange(1, 366, dtype=float)
    seasonal_price = fourier_seasonal_fit(_doys, _prices, K)

    # Build tick labels
    _tick_months = [datetime.date(2001, m, 1) for m in range(1, 13)]
    _tick_doys = [d.timetuple().tm_yday for d in _tick_months]
    _tick_labels = [d.strftime("%b") for d in _tick_months]

    _colours = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    _included = set(regression_years.value)
    _fig2 = go.Figure()
    for _i, _year in enumerate(available_years):
        _ydf = df.filter(pl.col("year") == _year)
        _in = _year in _included
        _fig2.add_trace(go.Scatter(
            x=_ydf.get_column("day_of_year").to_list(),
            y=_ydf.get_column("price").to_list(),
            mode="lines",
            line=dict(color=_colours[_i % len(_colours)], width=1, dash="solid" if _in else "dot"),
            name=str(_year),
            opacity=0.45 if _in else 0.2,
            hovertemplate=f"{_year} day %{{x}}<br>%{{y:.2f}} €/MWh<extra></extra>",
        ))
    _fig2.add_trace(go.Scatter(
        x=_doy_range.tolist(),
        y=seasonal_price.tolist(),
        mode="lines",
        line=dict(color="black", width=2.5),
        name="Seasonal fit",
        hovertemplate="Seasonal day %{x}<br>%{y:.2f} €/MWh<extra></extra>",
    ))
    _fig2.update_layout(
        title=f"Germany/Luxembourg — Fourier seasonal fit (K={K} harmonics)",
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
    mo.vstack([mo.hstack([regression_years, k_harmonics], justify="start"), mo.ui.plotly(_fig2)])


@app.cell
def _(datetime, go, mo, seasonal_price):
    _tick_months = [datetime.date(2001, m, 1) for m in range(1, 13)]
    _tick_doys = [d.timetuple().tm_yday for d in _tick_months]
    _tick_labels = [d.strftime("%b") for d in _tick_months]

    _fig_fit = go.Figure()
    _fig_fit.add_trace(go.Scatter(
        x=list(range(1, 366)),
        y=seasonal_price.tolist(),
        mode="lines",
        line=dict(color="#1f77b4", width=2.5),
        hovertemplate="Day %{x}<br>%{y:.2f} €/MWh<extra></extra>",
        showlegend=False,
    ))
    _fig_fit.update_layout(
        title="Fourier seasonal fit",
        xaxis=dict(
            title="",
            tickvals=_tick_doys, ticktext=_tick_labels,
            showgrid=True, gridcolor="#e5e5e5",
        ),
        yaxis=dict(title="€/MWh", showgrid=True, gridcolor="#e5e5e5"),
        plot_bgcolor="white",
        hovermode="x unified",
        margin=dict(t=50, b=40, l=60, r=20),
        height=300,
    )
    mo.ui.plotly(_fig_fit)


@app.cell
def _(mo):
    mo.md(r"""
    ## AR(1) parameter estimation

    After removing the seasonal trend, the residuals $r_t = p_t - \hat{p}(d_t)$ are
    modelled as an **AR(1)** autoregressive process:

    $$r_t = \varphi \, r_{t-1} + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \sigma^2)$$

    $\varphi$ is estimated by ordinary least squares on consecutive within-year
    residual pairs.  $\sigma$ is the standard deviation of the
    one-step innovations $\varepsilon_t = r_t - \hat\varphi r_{t-1}$.

    The stationary standard deviation $\sigma_\text{stat} = \sigma / \sqrt{1 - \varphi^2}$
    gives the long-run spread of prices around the seasonal mean.
    """)
    return


@app.cell
def _(ar1_fit, df, np, pl, regression_years, seasonal_price):
    _included = regression_years.value
    _year_residuals = []

    for _year in _included:
        _ydf = (
            df.filter(pl.col("year") == _year)
            .sort("day_of_year")
            .filter(pl.col("price").is_not_null())
        )
        _doys = np.clip(_ydf.get_column("day_of_year").to_numpy() - 1, 0, 364)
        _year_residuals.append(_ydf.get_column("price").to_numpy() - seasonal_price[_doys])

    phi_hat, sigma_hat, sigma_stationary = ar1_fit(_year_residuals)
    n_years = len(_included)
    n_obs = sum(len(r) for r in _year_residuals)
    return n_obs, n_years, phi_hat, sigma_hat, sigma_stationary


@app.cell
def _(mo, n_obs, n_years, phi_hat, sigma_hat, sigma_stationary):
    mo.vstack([
        mo.md(
            f"Estimated from **{n_years} years** of detrended daily prices "
            f"({n_obs} observations)."
        ),
        mo.hstack([
            mo.stat(f"{phi_hat:.3f}", label="φ  (AR(1) autocorrelation)"),
            mo.stat(f"{sigma_hat:.2f} €/MWh", label="σ  (innovation std dev)"),
            mo.stat(f"{sigma_stationary:.2f} €/MWh", label="σ_stationary  (residual std dev)"),
        ]),
    ])


@app.cell
def _(mo):
    mo.md(r"""
    ## Nominal price year

    The seasonal curve and AR(1) parameters estimated above define a generative model
    for a typical price year.  A scenario is drawn by:

    1. Sampling the initial residual from the stationary distribution:
       $r_0 \sim \mathcal{N}\!\left(0,\, \sigma^2/(1-\varphi^2)\right)$
    2. Iterating the AR(1) for 365 days:
       $r_t = \varphi \, r_{t-1} + \varepsilon_t$
    3. Adding the seasonal mean and clipping to zero:
       $p_t = \max\!\left(0,\; \hat{p}(t) + r_t\right)$

    The shaded region shows ±1σ and ±2σ around the seasonal mean (darker = higher
    probability density).  Press **Generate new year** to draw a new scenario.
    """)
    return


@app.cell
def _(mo):
    generate_btn = mo.ui.run_button(label="Generate new scenario year")
    return (generate_btn,)


@app.cell
def _(ar1_simulate, datetime, generate_btn, go, mo, np, phi_hat, seasonal_price, sigma_hat, sigma_stationary):
    _tick_months = [datetime.date(2001, m, 1) for m in range(1, 13)]
    _tick_doys = [d.timetuple().tm_yday for d in _tick_months]
    _tick_labels = [d.strftime("%b") for d in _tick_months]

    _x = list(range(1, 366))
    _s = sigma_stationary

    # Gradient bands: N full bands all centred on the seasonal mean, each with the
    # same alpha. Drawn outer-to-inner so bands accumulate — the centre (covered by
    # all N bands) appears darkest, fading smoothly to the outer edge.
    _N = 30
    _step = 2 * _s / _N
    _band_alpha = 0.04  # per-band alpha; centre accumulates all N → ~0.70 effective

    _fig3 = go.Figure()

    for _i in range(_N, 0, -1):
        _hw = _i * _step  # half-width of this band
        _fig3.add_trace(go.Scatter(
            x=_x + _x[::-1],
            y=(seasonal_price + _hw).tolist() + (seasonal_price - _hw).tolist()[::-1],
            fill="toself", fillcolor=f"rgba(31,119,180,{_band_alpha:.3f})",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))

    # ±1σ and ±2σ marker lines
    for _n, _label in [(1, "±1σ"), (2, "±2σ")]:
        for _sign, _show in [(1, True), (-1, False)]:
            _fig3.add_trace(go.Scatter(
                x=_x, y=(seasonal_price + _sign * _n * _s).tolist(),
                mode="lines",
                line=dict(color="rgba(31,119,180,0.7)", width=1, dash="dash"),
                name=_label, showlegend=_show,
                hovertemplate=f"{_label} %{{y:.2f}} €/MWh<extra></extra>",
            ))

    # Seasonal mean
    _fig3.add_trace(go.Scatter(
        x=_x, y=seasonal_price.tolist(),
        mode="lines", line=dict(color="#1f77b4", width=2.5),
        name="Seasonal mean",
        hovertemplate="Day %{x}<br>%{y:.2f} €/MWh<extra></extra>",
    ))

    # AR(1) simulated year — re-generated every time the button is pressed
    _r = ar1_simulate(phi_hat, sigma_hat)
    _sim_prices = np.maximum(0, seasonal_price + _r).tolist()

    _fig3.add_trace(go.Scatter(
        x=_x, y=_sim_prices,
        mode="lines", line=dict(color="rgba(220,50,50,0.75)", width=1.5),
        name="Simulated year",
        hovertemplate="Day %{x}<br>%{y:.2f} €/MWh<extra></extra>",
    ))

    _fig3.update_layout(
        title="Average seasonal price with uncertainty bands",
        xaxis=dict(
            title="",
            tickvals=_tick_doys, ticktext=_tick_labels,
            showgrid=True, gridcolor="#e5e5e5",
        ),
        yaxis=dict(
            title="€/MWh",
            showgrid=True, gridcolor="#e5e5e5",
            range=[
                float(np.min(seasonal_price - 2 * _s)) - 10,
                float(np.max(seasonal_price + 2 * _s)) + 10,
            ],
        ),
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=60, r=20),
        height=400,
    )
    mo.vstack([generate_btn, mo.ui.plotly(_fig3)])


if __name__ == "__main__":
    app.run()
