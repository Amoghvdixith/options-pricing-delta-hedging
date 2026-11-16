"""Tests for GBM path simulation. Fixed seeds make every check deterministic;
tolerances are set at several standard errors so they are not knife-edge."""

import math

import numpy as np
import pytest

from bs_hedging.simulation import simulate_gbm_paths, time_grid


def test_reproducible_with_same_seed():
    a = simulate_gbm_paths(100, 0.05, 0.2, 1.0, 50, 200, seed=7)
    b = simulate_gbm_paths(100, 0.05, 0.2, 1.0, 50, 200, seed=7)
    c = simulate_gbm_paths(100, 0.05, 0.2, 1.0, 50, 200, seed=8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_shape_initial_price_and_positivity():
    paths = simulate_gbm_paths(100, 0.05, 0.9, 2.0, 252, 300, seed=1)
    assert paths.shape == (300, 253)
    assert np.all(paths[:, 0] == 100.0)
    assert np.all(paths > 0.0)


def test_zero_volatility_is_deterministic():
    paths = simulate_gbm_paths(100, 0.05, 0.0, 1.0, 10, 3, seed=0)
    expected = 100.0 * np.exp(0.05 * time_grid(1.0, 10))
    for row in paths:
        np.testing.assert_allclose(row, expected, rtol=1e-12)


def test_log_return_mean_and_variance():
    mu, sigma, T, n = 0.05, 0.2, 1.0, 100_000
    paths = simulate_gbm_paths(100, mu, sigma, T, 4, n, seed=123)
    log_ret = np.log(paths[:, -1] / paths[:, 0])
    expected_mean = (mu - 0.5 * sigma**2) * T
    standard_error = sigma * math.sqrt(T / n)
    assert abs(log_ret.mean() - expected_mean) < 5 * standard_error
    assert log_ret.var(ddof=1) == pytest.approx(sigma**2 * T, rel=0.03)


def test_terminal_mean_matches_lognormal_expectation():
    s0, mu, sigma, T, n = 100.0, 0.08, 0.3, 1.0, 100_000
    terminal = simulate_gbm_paths(s0, mu, sigma, T, 12, n, seed=99)[:, -1]
    expected = s0 * math.exp(mu * T)
    sd = s0 * math.sqrt(math.exp(2 * mu * T) * (math.exp(sigma**2 * T) - 1))
    assert abs(terminal.mean() - expected) < 5 * sd / math.sqrt(n)


def test_increments_are_independent_across_steps():
    paths = simulate_gbm_paths(100, 0.0, 0.2, 1.0, 2, 50_000, seed=5)
    r1 = np.log(paths[:, 1] / paths[:, 0])
    r2 = np.log(paths[:, 2] / paths[:, 1])
    assert abs(np.corrcoef(r1, r2)[0, 1]) < 5 / math.sqrt(50_000)


@pytest.mark.parametrize(
    "kwargs",
    [dict(initial_price=0), dict(volatility=-0.1), dict(maturity=0), dict(n_steps=0),
     dict(n_paths=0), dict(n_steps=2.5), dict(drift=float("nan"))],
)
def test_invalid_inputs(kwargs):
    base = dict(initial_price=100, drift=0.05, volatility=0.2, maturity=1.0, n_steps=10, n_paths=5, seed=0)
    base.update(kwargs)
    with pytest.raises(ValueError):
        simulate_gbm_paths(**base)
