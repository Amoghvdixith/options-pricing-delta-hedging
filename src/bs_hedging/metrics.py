"""Summary statistics for pathwise terminal hedging P&L.

Conventions
-----------
* Standard deviation is the SAMPLE standard deviation (``ddof = 1``).
* RMSE is measured relative to a target of zero terminal P&L:
  ``sqrt(mean(pnl ** 2))``. It equals sqrt(std_pop^2 + mean^2), so it differs
  from the standard deviation whenever the mean is not zero.
* Percentiles use NumPy's default ``method="linear"`` (Hyndman & Fan type 7).
* P&L and transaction costs are terminal/nominal monetary amounts in the same
  units as the stock price (per one option on one share); nothing is discounted.
* Transaction cost per path = cumulative nominal cost paid over the life of
  the hedge (initial purchase + rebalances + liquidation).
"""

from __future__ import annotations

import math

import numpy as np

PERCENTILE_METHOD = "linear"


def rmse_to_zero(pnl: np.ndarray) -> float:
    """Root-mean-square terminal P&L relative to a perfect hedge (zero)."""
    pnl = np.asarray(pnl, dtype=float)
    return float(math.sqrt(np.mean(pnl**2)))


def summarize_pnl(pnl: np.ndarray, transaction_costs: np.ndarray | None = None) -> dict[str, float]:
    """Return the summary statistics used in the result tables."""
    pnl = np.asarray(pnl, dtype=float)
    if pnl.ndim != 1 or pnl.size < 2:
        raise ValueError("pnl must be a 1-D array with at least two values")
    if not np.all(np.isfinite(pnl)):
        raise ValueError("pnl must be finite")
    mean_cost = 0.0 if transaction_costs is None else float(np.mean(transaction_costs))
    return {
        "n_paths": int(pnl.size),
        "mean_pnl": float(np.mean(pnl)),
        "std_pnl": float(np.std(pnl, ddof=1)),
        "rmse_pnl": rmse_to_zero(pnl),
        "p05_pnl": float(np.percentile(pnl, 5, method=PERCENTILE_METHOD)),
        "median_pnl": float(np.percentile(pnl, 50, method=PERCENTILE_METHOD)),
        "p95_pnl": float(np.percentile(pnl, 95, method=PERCENTILE_METHOD)),
        "min_pnl": float(np.min(pnl)),
        "max_pnl": float(np.max(pnl)),
        "mean_transaction_cost": mean_cost,
    }
