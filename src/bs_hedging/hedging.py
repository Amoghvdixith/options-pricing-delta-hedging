"""Self-financing delta hedge of a SHORT European call (one option on one share).

Portfolio: short 1 call, long ``delta`` shares, and a cash account that earns
(or pays) the continuously compounded risk-free rate ``r``.

Time grid: t_k = k * dt, k = 0..N, dt = T / N.

* k = 0: receive premium; buy Delta_0 shares; pay cost c * |Delta_0| * S_0.
      cash_0 = premium - Delta_0 S_0 - c |Delta_0| S_0
* k = 1..N-1: cash_k = cash_{k-1} e^{r dt}; if k is a rebalance step
  (k % rebalance_every == 0), trade to Delta_k:
      cash_k -= (Delta_k - Delta_old) S_k + c |Delta_k - Delta_old| S_k
* k = N (expiry): cash_N = cash_{N-1} e^{r dt}; NO Greek is evaluated and NO
  final rebalance is made. Sell remaining shares, pay liquidation cost and the
  option payoff exactly once:
      cash_N += Delta_held S_N - c |Delta_held| S_N - max(S_N - K, 0)

Terminal hedging P&L = cash_N (shares are 0 and the option is settled).
Positive P&L = profit for the hedged option seller. Amounts are terminal,
undiscounted, in the same currency units as the stock price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .greeks import call_delta_array
from .pricing import bs_price

DeltaFunction = Callable[[np.ndarray, float, int], np.ndarray]


@dataclass
class HedgeResult:
    """Outcome of one hedging strategy over all simulated paths."""

    terminal_pnl: np.ndarray
    transaction_costs: np.ndarray
    premium: float
    rebalance_steps: list[int] = field(default_factory=list)


def rebalance_steps(n_steps: int, rebalance_every: int) -> list[int]:
    """Steps at which the hedge is set or adjusted: 0, m, 2m, ... strictly before N.

    Example: N = 252, m = 5 -> [0, 5, ..., 250]; the last interval (250 -> 252)
    is two steps long and the position is then liquidated at expiry.
    """
    if int(n_steps) != n_steps or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if int(rebalance_every) != rebalance_every or rebalance_every < 1:
        raise ValueError("rebalance_every must be a positive integer")
    return list(range(0, int(n_steps), int(rebalance_every)))


def simulate_delta_hedge(
    paths: np.ndarray,
    strike: float,
    maturity: float,
    rate: float,
    hedging_volatility: float,
    rebalance_every: int,
    transaction_cost_rate: float = 0.0,
    delta_function: DeltaFunction | None = None,
    premium: float | None = None,
) -> HedgeResult:
    """Run the self-financing short-call delta hedge on every path.

    Parameters
    ----------
    paths : array (n_paths, N + 1) of simulated stock prices; all paths must
        start at the same price.
    strike, maturity, rate : option contract and risk-free rate.
    hedging_volatility : volatility used for the premium and hedge ratios.
    rebalance_every : rebalance every this many steps (1 = every step).
    transaction_cost_rate : proportional cost on traded stock notional
        (0.0005 = 5 basis points). Applied to the initial purchase,
        intermediate trades and final liquidation.
    delta_function : optional ``f(spots, time_to_maturity, step) -> deltas``.
        Defaults to the Black-Scholes call delta with ``hedging_volatility``.
        Tests use it to inject simple hand-chosen hedge ratios.
    premium : optional premium override; defaults to the Black-Scholes call
        price at S_0 with ``hedging_volatility``.
    """
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2 or paths.shape[1] < 2:
        raise ValueError("paths must have shape (n_paths, n_steps + 1) with n_steps >= 1")
    if not np.all(np.isfinite(paths)) or np.any(paths <= 0.0):
        raise ValueError("paths must contain finite, strictly positive prices")
    if not np.allclose(paths[:, 0], paths[0, 0]):
        raise ValueError("all paths must start from the same initial price")
    if transaction_cost_rate < 0.0:
        raise ValueError("transaction_cost_rate must be >= 0")
    if maturity <= 0.0:
        raise ValueError("maturity must be > 0")

    n_steps = paths.shape[1] - 1
    dt = maturity / n_steps
    growth = math.exp(rate * dt)
    schedule = rebalance_steps(n_steps, rebalance_every)
    schedule_set = set(schedule)

    if delta_function is None:
        def delta_function(spots: np.ndarray, tau: float, step: int) -> np.ndarray:
            return call_delta_array(spots, strike, tau, rate, hedging_volatility)

    if premium is None:
        premium = bs_price(paths[0, 0], strike, maturity, rate, hedging_volatility, "call")

    # k = 0: sell the call, buy the initial hedge.
    spot = paths[:, 0]
    held = np.asarray(delta_function(spot, maturity, 0), dtype=float) * np.ones_like(spot)
    costs = transaction_cost_rate * np.abs(held) * spot
    cash = premium - held * spot - costs

    # k = 1 .. N-1: accrue interest, rebalance on schedule.
    for step in range(1, n_steps):
        cash = cash * growth
        if step in schedule_set:
            spot = paths[:, step]
            time_to_maturity = maturity - step * dt
            target = np.asarray(delta_function(spot, time_to_maturity, step), dtype=float)
            trade = target - held
            trade_cost = transaction_cost_rate * np.abs(trade) * spot
            cash = cash - trade * spot - trade_cost
            costs = costs + trade_cost
            held = target * np.ones_like(spot)

    # k = N: accrue, liquidate, settle the option once. No Greek at expiry.
    cash = cash * growth
    final_spot = paths[:, n_steps]
    liquidation_cost = transaction_cost_rate * np.abs(held) * final_spot
    cash = cash + held * final_spot - liquidation_cost
    costs = costs + liquidation_cost
    cash = cash - np.maximum(final_spot - strike, 0.0)

    return HedgeResult(
        terminal_pnl=cash,
        transaction_costs=costs,
        premium=float(premium),
        rebalance_steps=schedule,
    )


def compare_strategies(
    paths: np.ndarray,
    strategies: dict[str, int],
    strike: float,
    maturity: float,
    rate: float,
    hedging_volatility: float,
    transaction_cost_rate: float = 0.0,
) -> dict[str, HedgeResult]:
    """Run several rebalancing frequencies on the SAME path array."""
    return {
        name: simulate_delta_hedge(
            paths, strike, maturity, rate, hedging_volatility, every, transaction_cost_rate
        )
        for name, every in strategies.items()
    }
