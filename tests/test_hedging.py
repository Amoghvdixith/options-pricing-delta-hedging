"""Tests for the self-financing delta-hedge accounting.

The hand-calculated cases inject simple hedge ratios (0.5 then 0.8) through
``delta_function`` so the expected cash balances below are worked out by hand,
independently of the Black-Scholes formulas.

Common setup: path 100 -> 110 -> 105, N = 2 steps, strike 100, premium 10.
"""

import math

import numpy as np
import pytest

from bs_hedging.hedging import compare_strategies, rebalance_steps, simulate_delta_hedge
from bs_hedging.pricing import call_price
from bs_hedging.simulation import simulate_gbm_paths

PATH = np.array([[100.0, 110.0, 105.0]])


def injected_delta(spots, tau, step):
    return np.full_like(spots, {0: 0.5, 1: 0.8}[step])


def run(rate=0.0, cost=0.0, every=1, maturity=2.0):
    return simulate_delta_hedge(
        PATH, strike=100.0, maturity=maturity, rate=rate, hedging_volatility=0.2,
        rebalance_every=every, transaction_cost_rate=cost,
        delta_function=injected_delta, premium=10.0,
    )


def test_hand_calculated_no_interest_no_costs():
    # t0: cash = 10 - 0.5*100 = -40
    # t1: buy 0.3 at 110 -> cash = -40 - 33 = -73
    # t2: sell 0.8 at 105 (+84) -> 11; pay payoff 5 -> 6
    result = run()
    assert result.terminal_pnl[0] == pytest.approx(6.0, abs=1e-12)
    assert result.transaction_costs[0] == 0.0


def test_hand_calculated_with_interest():
    # r = 10% p.a., T = 2, dt = 1 year, growth g = e^0.1 = 1.1051709180756477
    g = 1.1051709180756477
    cash_t0 = -40.0
    cash_t1 = cash_t0 * g - 33.0           # -77.20683672...
    cash_t2 = cash_t1 * g + 84.0 - 5.0     # accrue, liquidate, pay payoff
    assert cash_t2 == pytest.approx(-6.3267506, abs=1e-6)  # hand value
    assert run(rate=0.10).terminal_pnl[0] == pytest.approx(cash_t2, abs=1e-12)


def test_hand_calculated_with_transaction_costs():
    # c = 1%: t0 cost 0.5*100*0.01 = 0.50, cash = 10 - 50 - 0.5 = -40.5
    # t1: trade 0.3 at 110: notional 33, cost 0.33 -> cash = -73.83
    # t2: liquidate 0.8 at 105: +84, cost 0.84; payoff 5 -> 4.33
    result = run(cost=0.01)
    assert result.terminal_pnl[0] == pytest.approx(4.33, abs=1e-12)
    assert result.transaction_costs[0] == pytest.approx(0.50 + 0.33 + 0.84, abs=1e-12)


def test_initial_and_final_costs_counted_once():
    # Rebalance only at t0 (every = 2): cost at t0 = 0.5, at liquidation 0.5*105*0.01 = 0.525
    result = run(cost=0.01, every=2)
    assert result.transaction_costs[0] == pytest.approx(0.5 + 0.525, abs=1e-12)
    # cash: 10 - 50 - 0.5 = -40.5; + 52.5 - 0.525 - 5 = 6.475
    assert result.terminal_pnl[0] == pytest.approx(6.475, abs=1e-12)


def test_payoff_paid_once_with_zero_hedge():
    """With delta = 0 the P&L must be premium accrued at r minus one payoff."""
    paths = np.array([[100.0, 103.0, 98.0, 120.0]])
    result = simulate_delta_hedge(
        paths, 100.0, 1.0, 0.05, 0.2, 1, delta_function=lambda s, t, k: np.zeros_like(s), premium=7.0
    )
    assert result.terminal_pnl[0] == pytest.approx(7.0 * math.exp(0.05) - 20.0, abs=1e-12)


def test_no_greek_evaluated_at_expiry():
    seen = []

    def recording_delta(spots, tau, step):
        seen.append((step, tau))
        assert tau > 0.0
        return np.full_like(spots, 0.5)

    simulate_delta_hedge(np.array([[100.0, 101, 102, 103, 104]]), 100, 1.0, 0.0, 0.2, 1,
                         delta_function=recording_delta, premium=5.0)
    assert [s for s, _ in seen] == [0, 1, 2, 3]
    assert seen[-1][1] == pytest.approx(0.25)


@pytest.mark.parametrize(
    "n_steps, every, expected",
    [(10, 5, [0, 5]), (10, 1, list(range(10))), (12, 5, [0, 5, 10]), (2, 5, [0])],
)
def test_rebalance_schedules(n_steps, every, expected):
    assert rebalance_steps(n_steps, every) == expected


def test_baseline_weekly_schedule_for_252_steps():
    schedule = rebalance_steps(252, 5)
    assert schedule[0] == 0 and schedule[-1] == 250 and len(schedule) == 51
    assert 252 not in schedule
    assert rebalance_steps(252, 1) == list(range(252))


def test_strategies_use_identical_paths():
    paths = simulate_gbm_paths(100, 0.05, 0.2, 1.0, 20, 50, seed=3)
    snapshot = paths.copy()
    results = compare_strategies(paths, {"daily": 1, "weekly": 5}, 100, 1.0, 0.05, 0.2, 0.0005)
    np.testing.assert_array_equal(paths, snapshot)  # paths not mutated
    for name, every in {"daily": 1, "weekly": 5}.items():
        direct = simulate_delta_hedge(snapshot, 100, 1.0, 0.05, 0.2, every, 0.0005)
        np.testing.assert_array_equal(results[name].terminal_pnl, direct.terminal_pnl)


def test_premium_defaults_to_black_scholes_price():
    paths = simulate_gbm_paths(100, 0.05, 0.2, 1.0, 5, 3, seed=0)
    result = simulate_delta_hedge(paths, 100, 1.0, 0.05, 0.2, 1)
    assert result.premium == pytest.approx(call_price(100, 100, 1.0, 0.05, 0.2))


def test_deterministic_deep_in_the_money_hedge_is_exact():
    """Stock grows deterministically at r, strike 80, tiny hedging volatility:
    delta is 1 throughout, premium = 100 - 80 e^{-r}, cash = -80 e^{-r} grows
    to -80, and liquidation (+S_T) minus payoff (S_T - 80) returns +80 -> P&L 0."""
    paths = simulate_gbm_paths(100, 0.05, 0.0, 1.0, 252, 1, seed=0)
    result = simulate_delta_hedge(paths, 80, 1.0, 0.05, 0.01, 1)
    assert result.premium == pytest.approx(100 - 80 * math.exp(-0.05), abs=1e-12)
    assert result.terminal_pnl[0] == pytest.approx(0.0, abs=1e-9)


def test_hedging_reduces_dispersion_relative_to_unhedged():
    paths = simulate_gbm_paths(100, 0.05, 0.2, 1.0, 252, 2000, seed=11)
    hedged = simulate_delta_hedge(paths, 100, 1.0, 0.05, 0.2, 1).terminal_pnl
    unhedged = simulate_delta_hedge(paths, 100, 1.0, 0.05, 0.2, 1,
                                    delta_function=lambda s, t, k: np.zeros_like(s)).terminal_pnl
    assert np.std(hedged, ddof=1) < np.std(unhedged, ddof=1) / 5


def test_invalid_inputs():
    with pytest.raises(ValueError):
        simulate_delta_hedge(np.array([100.0, 101.0]), 100, 1, 0.05, 0.2, 1)  # 1-D
    with pytest.raises(ValueError):
        simulate_delta_hedge(PATH, 100, 1, 0.05, 0.2, 1, transaction_cost_rate=-0.1)
    with pytest.raises(ValueError):
        simulate_delta_hedge(np.array([[100.0, 101.0], [99.0, 100.0]]), 100, 1, 0.05, 0.2, 1)
    with pytest.raises(ValueError):
        rebalance_steps(252, 0)
