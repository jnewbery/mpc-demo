"""
autoregression.py - AR(1) parameter estimation and simulation for price residuals.
"""

import numpy as np

def ar1_fit(year_residuals: list[np.ndarray]) -> tuple[float, float, float]:
    """Estimate AR(1) parameters from per-year residual sequences.

    Consecutive lags are computed within each year only, so year boundaries
    are never crossed in the lag pairs.

    Parameters
    ----------
    year_residuals : list of 1-D arrays, one per year

    Returns
    -------
    (phi, sigma, sigma_stationary)
        phi              — AR(1) autocorrelation (OLS estimate)
        sigma            — innovation std dev
        sigma_stationary — empirical stationary std dev (std of all residuals)
    """
    r0 = np.concatenate([r[:-1] for r in year_residuals])
    r1 = np.concatenate([r[1:] for r in year_residuals])
    phi = float(np.dot(r0, r1) / np.dot(r0, r0))
    sigma = float(np.std(r1 - phi * r0, ddof=1))
    sigma_stationary = float(np.std(np.concatenate(year_residuals), ddof=1))
    return phi, sigma, sigma_stationary


def ar1_simulate(
    phi: float,
    sigma: float,
    n: int = 365,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw a single AR(1) residual path of length n.

    The initial state is drawn from the parametric stationary distribution
    N(0, sigma^2 / (1 - phi^2)).

    Parameters
    ----------
    phi   : AR(1) autocorrelation
    sigma : innovation std dev
    n     : path length (default 365)
    rng   : random generator; a fresh default_rng() is used if None

    Returns
    -------
    np.ndarray of shape (n,)
    """
    if rng is None:
        rng = np.random.default_rng()
    sigma_stat = sigma / np.sqrt(1 - phi ** 2)
    r = np.empty(n)
    r[0] = rng.normal(0, sigma_stat)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + rng.normal(0, sigma)
    return r
