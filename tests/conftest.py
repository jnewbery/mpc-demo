import pytest
import pathlib
import numpy as np

from src.storage_lp import StorageParams

_DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


@pytest.fixture
def test_sp() -> StorageParams:
    """StorageParams scaled for the pre-generated scenario CSVs (demand up to ~500 MWh/day)."""
    return StorageParams(
        s_max=500.0, s_min=0.0, s0=250.0,
        u_plus_max=300.0, u_minus_max=300.0,
        eta=0.90, cop=3.0, h_max=600.0,
    )


@pytest.fixture
def test_prices(rows: int | None = None) -> np.ndarray:
    arr = np.genfromtxt(
        _DATA_DIR / "price_scenarios" / "scenario_1.csv",
        delimiter=",", skip_header=1, usecols=(2,),
    )
    return arr if rows is None else arr[:rows]

@pytest.fixture
def test_seasonal_prices(scenario: int = 1, rows: int | None = None) -> np.ndarray:
    arr = np.genfromtxt(
        _DATA_DIR / "price_scenarios" / f"scenario_{scenario}.csv",
        delimiter=",", skip_header=1, usecols=(1,),  # 'seasonal' column
    )
    return arr if rows is None else arr[:rows]

@pytest.fixture
def test_demand(rows: int | None = None) -> np.ndarray:
    arr = np.genfromtxt(
        _DATA_DIR / "heat_demand_scenarios" / "scenario_1.csv",
        delimiter=",", skip_header=1, usecols=(4,),
    )
    return arr if rows is None else arr[:rows]
