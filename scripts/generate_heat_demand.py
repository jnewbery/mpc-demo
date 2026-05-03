#!/usr/bin/env python3
"""Generate heat demand scenarios using the Fourier + AR(1) model from heat_demand.py."""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np
import polars as pl

from src.fourier import fourier_seasonal_fit
from src.heat_demand import calibrate, compute_demand
from src.simulation import ar1_fit, ar1_simulate

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_TEMP_FILE = _REPO_ROOT / "data" / "goerlitz_daily_weather.txt.gz"
_DHN_FILE = _REPO_ROOT / "data" / "dhn" / "heatgrids_Sachsen.csv"

DEFAULT_HARMONICS = 3
DEFAULT_SCENARIOS = 1
DEFAULT_OUTPUT = "heat_scenario"


def _load_temperatures() -> pl.DataFrame:
    return (
        pl.read_csv(
            _TEMP_FILE,
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


def _load_annual_demand() -> float:
    return float(
        pl.read_csv(_DHN_FILE)
        .filter(pl.col("GEN") == "Görlitz")
        .select(pl.col("DH_demand").sum())
        .item()
    )


def _available_years(df: pl.DataFrame) -> list[int]:
    return (
        df.filter(pl.col("tmk").is_not_null())
        .select("year")
        .unique()
        .sort("year")
        .get_column("year")
        .to_list()
    )


def _prompt(label: str, default: str) -> str:
    try:
        value = input(f"{label} [{default}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return value if value else default


def _parse_years(raw: str, available: list[int]) -> list[int]:
    try:
        years = [int(y.strip()) for y in raw.split(",") if y.strip()]
    except ValueError:
        print(f"Error: could not parse years from {raw!r}", file=sys.stderr)
        sys.exit(1)
    invalid = [y for y in years if y not in available]
    if invalid:
        print(f"Error: years not in data: {invalid}", file=sys.stderr)
        sys.exit(1)
    return years


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate heat demand scenarios using the Fourier + AR(1) model."
    )
    parser.add_argument("--interactive", dest="interactive", action="store_true")
    parser.add_argument("--no-interactive", "-n", dest="interactive", action="store_false")
    parser.set_defaults(interactive=True)
    parser.add_argument("--included-years", type=str, default=None)
    parser.add_argument("--harmonics", type=int, default=DEFAULT_HARMONICS)
    parser.add_argument("--scenarios", type=int, default=DEFAULT_SCENARIOS)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = _load_temperatures()
    available = _available_years(df)
    default_years = ",".join(str(y) for y in available)

    if args.interactive:
        years_raw = _prompt(
            f"Included years (available: {' '.join(str(y) for y in available)})",
            args.included_years or default_years,
        )
        harmonics_str = _prompt("Harmonics K", str(args.harmonics))
        scenarios_str = _prompt("Number of scenarios", str(args.scenarios))
        output = _prompt("Output base name", args.output)

        try:
            harmonics = int(harmonics_str)
            scenarios = int(scenarios_str)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        years_raw = args.included_years or default_years
        harmonics = args.harmonics
        scenarios = args.scenarios
        output = args.output

    included_years = _parse_years(years_raw, available)

    # Build design arrays from included years
    subset = df.filter(
        pl.col("tmk").is_not_null() & pl.col("year").is_in(included_years)
    )
    days = subset.get_column("day_of_year").to_numpy().astype(float)
    temps = subset.get_column("tmk").to_numpy()

    # Fourier seasonal fit
    seasonal = fourier_seasonal_fit(days, temps, harmonics)

    # AR(1) parameter estimation (within-year residuals only)
    year_residuals = []
    for year in included_years:
        ydf = (
            df.filter(pl.col("year") == year)
            .sort("day_of_year")
            .filter(pl.col("tmk").is_not_null())
        )
        idx = np.clip(ydf.get_column("day_of_year").to_numpy() - 1, 0, 364)
        year_residuals.append(ydf.get_column("tmk").to_numpy() - seasonal[idx])

    phi, sigma, _ = ar1_fit(year_residuals)

    # Calibrate heat demand model
    annual_demand_mwh = _load_annual_demand()
    k, p_base = calibrate(seasonal, annual_demand_mwh)

    print(
        f"Fitted: K={harmonics} harmonics, φ={phi:.3f}, σ={sigma:.2f} °C "
        f"({len(included_years)} years)"
    )
    print(f"Calibrated: k={k:.2f} MW/K, P_base={p_base:.1f} MWh/day "
          f"(annual demand={annual_demand_mwh:,.0f} MWh)")

    # Generate and save scenarios
    days_col = list(range(1, 366))
    for i in range(1, scenarios + 1):
        r = ar1_simulate(phi, sigma)
        sim_temps = (seasonal + r).tolist()
        heat_demand = compute_demand(np.array(sim_temps), k, p_base).tolist()
        out_path = pathlib.Path(f"{output}_{i}.csv")
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["day_of_year", "seasonal_temp", "temperature", "heat_demand"])
            writer.writerows(zip(days_col, seasonal.tolist(), sim_temps, heat_demand))
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
