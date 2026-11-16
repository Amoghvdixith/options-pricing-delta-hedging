"""Tests for P&L summary statistics against hand-computed values."""

import math

import numpy as np
import pytest

from bs_hedging.metrics import rmse_to_zero, summarize_pnl


def test_symmetric_array_hand_values():
    pnl = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    stats = summarize_pnl(pnl, transaction_costs=np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    assert stats["n_paths"] == 5
    assert stats["mean_pnl"] == 0.0
    assert stats["std_pnl"] == pytest.approx(math.sqrt(10 / 4))   # ddof = 1
    assert stats["rmse_pnl"] == pytest.approx(math.sqrt(10 / 5))  # mean of squares
    # linear percentile: position p*(n-1); 5% -> 0.2 -> -2 + 0.2*1 = -1.8
    assert stats["p05_pnl"] == pytest.approx(-1.8)
    assert stats["median_pnl"] == 0.0
    assert stats["p95_pnl"] == pytest.approx(1.8)
    assert stats["min_pnl"] == -2.0 and stats["max_pnl"] == 2.0
    assert stats["mean_transaction_cost"] == pytest.approx(0.3)


def test_rmse_differs_from_std_when_mean_nonzero():
    pnl = np.array([1.0, 2.0, 3.0, 4.0])
    stats = summarize_pnl(pnl)
    assert stats["mean_pnl"] == 2.5
    assert stats["std_pnl"] == pytest.approx(math.sqrt(5 / 3))      # 1.2910
    assert stats["rmse_pnl"] == pytest.approx(math.sqrt(30 / 4))    # 2.7386
    # identity: rmse^2 = population variance + mean^2
    assert stats["rmse_pnl"] ** 2 == pytest.approx(np.var(pnl) + 2.5**2)
    # 5th percentile: position 0.15 -> 1.15
    assert stats["p05_pnl"] == pytest.approx(1.15)
    assert stats["mean_transaction_cost"] == 0.0


def test_rmse_function():
    assert rmse_to_zero(np.array([3.0, -4.0])) == pytest.approx(math.sqrt(12.5))


def test_invalid_inputs():
    with pytest.raises(ValueError):
        summarize_pnl(np.array([1.0]))
    with pytest.raises(ValueError):
        summarize_pnl(np.array([1.0, np.nan]))
