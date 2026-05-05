import pytest
import pathlib
import numpy as np

_DATA_DIR = pathlib.Path(__file__).parent.parent / "data"

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
        delimiter=",", skip_header=1, usecols=(2,),
    )
    return arr if rows is None else arr[:rows]
