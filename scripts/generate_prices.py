#!/usr/bin/env python3
"""Generate day-ahead price scenarios using the Fourier + AR(1) model from prices.py."""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np
import polars as pl

_REPO_ROOT = pathlib.Path(__file__).parent.parent
# sys.path.insert(0, str(_REPO_ROOT))
from src.fourier import fourier_seasonal_fit
from src.simulation import ar1_fit, ar1_simulate

_DATA_FILE = _REPO_ROOT / "data" / "day_ahead_prices.csv"
_PRICE_COL = "Germany/Luxembourg [€/MWh] Calculated resolutions"

DEFAULT_HARMONICS = 3
DEFAULT_SCENARIOS = 1
DEFAULT_OUTPUT = "scenario"


def _load_df() -> pl.DataFrame:
    return (
        pl.read_csv(
            _DATA_FILE,
            separator=";",
            encoding="utf8-lossy",
            infer_schema=False,
        )
        .with_columns(
            pl.col("Start date").str.strptime(pl.Date, "%b %d, %Y").alias("date"),
            pl.col(_PRICE_COL)
            .str.replace("^-$", "")
            .cast(pl.Float64, strict=False)
            .alias("price"),
        )
        .with_columns(
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.ordinal_day().alias("day_of_year"),
        )
        .select(["date", "year", "day_of_year", "price"])
    )


def _available_years(df: pl.DataFrame) -> list[int]:
    return (
        df.filter(pl.col("price").is_not_null() & (pl.col("year") < 2026))
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
        description="Generate price scenarios using the Fourier + AR(1) model."
    )
    parser.add_argument(
        "--interactive", dest="interactive", action="store_true",
    )
    parser.add_argument(
        "--no-interactive", "-n", dest="interactive", action="store_false",
    )
    parser.set_defaults(interactive=True)
    parser.add_argument("--included-years", type=str, default=None)
    parser.add_argument("--harmonics", type=int, default=DEFAULT_HARMONICS)
    parser.add_argument("--scenarios", type=int, default=DEFAULT_SCENARIOS)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = _load_df()
    available = _available_years(df)
    default_years = ",".join(str(y) for y in available if y != 2022)

    # Resolve parameter values, prompting if interactive
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
        pl.col("price").is_not_null() & pl.col("year").is_in(included_years)
    )
    days = subset.get_column("day_of_year").to_numpy().astype(float)
    prices = subset.get_column("price").to_numpy()

    # Fourier seasonal fit
    seasonal = fourier_seasonal_fit(days, prices, harmonics)

    # AR(1) parameter estimation (within-year residuals only)
    year_residuals = []
    for year in included_years:
        ydf = (
            df.filter(pl.col("year") == year)
            .sort("day_of_year")
            .filter(pl.col("price").is_not_null())
        )
        idx = np.clip(ydf.get_column("day_of_year").to_numpy() - 1, 0, 364)
        year_residuals.append(ydf.get_column("price").to_numpy() - seasonal[idx])

    phi, sigma, _ = ar1_fit(year_residuals)

    print(
        f"Fitted: K={harmonics} harmonics, φ={phi:.3f}, σ={sigma:.2f} €/MWh "
        f"({len(included_years)} years)"
    )

    # Generate and save scenarios
    days_col = list(range(1, 366))
    for i in range(1, scenarios + 1):
        r = ar1_simulate(phi, sigma)
        scenario_prices = np.maximum(0.0, seasonal + r).tolist()
        out_path = pathlib.Path(f"{output}_{i}.csv")
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["day_of_year", "seasonal", "price"])
            writer.writerows(zip(days_col, seasonal.tolist(), scenario_prices))
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
