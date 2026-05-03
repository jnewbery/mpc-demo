from __future__ import annotations

import numpy as np


def fourier_seasonal_fit(doys: np.ndarray, values: np.ndarray, k: int) -> np.ndarray:
    """Fit a Fourier regression and return the seasonal curve for days 1–365.

    Model: f(d) = a0 + Σ_{j=1}^{k} [a_j cos(2πjd/365.25) + b_j sin(2πjd/365.25)]

    Parameters
    ----------
    doys   : day-of-year values (1-based)
    values : observed values, same length as doys
    k      : number of harmonics

    Returns
    -------
    seasonal : shape-(365,) array; seasonal[i] is the fitted value for day i+1
    """
    def _features(d: np.ndarray) -> np.ndarray:
        cols = [np.ones(len(d))]
        for j in range(1, k + 1):
            cols.append(np.cos(2 * np.pi * j * d / 365.25))
            cols.append(np.sin(2 * np.pi * j * d / 365.25))
        return np.column_stack(cols)

    d = np.asarray(doys, dtype=float)
    coeffs, _, _, _ = np.linalg.lstsq(_features(d), values, rcond=None)
    return _features(np.arange(1, 366, dtype=float)) @ coeffs
