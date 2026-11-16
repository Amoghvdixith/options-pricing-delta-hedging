"""Simulated stock-price paths under geometric Brownian motion (GBM).

Exact discretisation (no Euler error):

    S(t + dt) = S(t) * exp[(mu - 0.5 sigma^2) dt + sigma sqrt(dt) Z],  Z ~ N(0, 1)

All data produced here are SIMULATED, not market data.
"""

from __future__ import annotations

import math

import numpy as np


def simulate_gbm_paths(
    initial_price: float,
    drift: float,
    volatility: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """Simulate GBM paths.

    Parameters
    ----------
    initial_price : S(0) > 0.
    drift : annualised drift ``mu`` used to generate the paths.
    volatility : annualised volatility ``sigma >= 0`` used to generate the paths.
    maturity : horizon in years (> 0).
    n_steps : number of equal time steps (>= 1).
    n_paths : number of independent paths (>= 1).
    seed : seed for ``numpy.random.default_rng`` (same seed -> same paths).

    Returns
    -------
    Array of shape ``(n_paths, n_steps + 1)``; column 0 equals ``initial_price``.
    """
    for name, value in (("initial_price", initial_price), ("drift", drift),
                        ("volatility", volatility), ("maturity", maturity)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if initial_price <= 0.0:
        raise ValueError("initial_price must be > 0")
    if volatility < 0.0:
        raise ValueError("volatility must be >= 0")
    if maturity <= 0.0:
        raise ValueError("maturity must be > 0")
    if int(n_steps) != n_steps or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if int(n_paths) != n_paths or n_paths < 1:
        raise ValueError("n_paths must be a positive integer")

    rng = np.random.default_rng(seed)
    dt = maturity / n_steps
    shocks = rng.standard_normal((int(n_paths), int(n_steps)))
    log_increments = (drift - 0.5 * volatility**2) * dt + volatility * math.sqrt(dt) * shocks
    log_paths = np.concatenate(
        [np.zeros((int(n_paths), 1)), np.cumsum(log_increments, axis=1)], axis=1
    )
    return initial_price * np.exp(log_paths)


def time_grid(maturity: float, n_steps: int) -> np.ndarray:
    """Return the time points 0, dt, ..., maturity (length ``n_steps + 1``)."""
    return np.linspace(0.0, maturity, int(n_steps) + 1)
