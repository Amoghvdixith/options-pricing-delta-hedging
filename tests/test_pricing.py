"""Tests for analytical Black-Scholes pricing."""

import math

import pytest

from bs_hedging.pricing import bs_price, call_price, price_bounds, put_price

GRID = [
    (spot, strike, maturity, rate, vol)
    for spot in (60.0, 100.0, 140.0)
    for strike in (80.0, 100.0, 120.0)
    for maturity in (0.05, 1.0, 3.0)
    for rate in (-0.01, 0.0, 0.05)
    for vol in (0.05, 0.2, 0.6)
]


# --- Independently specified reference values -------------------------------

def test_reference_hull_textbook_example():
    """Hull, 'Options, Futures, and Other Derivatives', worked example:
    S=42, K=40, r=10%, sigma=20%, T=0.5 -> call ~ 4.76, put ~ 0.81 (2 d.p.)."""
    assert call_price(42.0, 40.0, 0.5, 0.10, 0.20) == pytest.approx(4.76, abs=0.005)
    assert put_price(42.0, 40.0, 0.5, 0.10, 0.20) == pytest.approx(0.81, abs=0.005)


def test_reference_at_the_money_one_year():
    """Widely quoted value: S=K=100, T=1, r=5%, sigma=20% -> call 10.4506 (4 d.p.)."""
    assert call_price(100.0, 100.0, 1.0, 0.05, 0.20) == pytest.approx(10.4506, abs=5e-5)


# --- Structural relationships ----------------------------------------------

@pytest.mark.parametrize("spot, strike, maturity, rate, vol", GRID)
def test_put_call_parity(spot, strike, maturity, rate, vol):
    lhs = call_price(spot, strike, maturity, rate, vol) - put_price(spot, strike, maturity, rate, vol)
    rhs = spot - strike * math.exp(-rate * maturity)
    assert lhs == pytest.approx(rhs, abs=1e-9)


@pytest.mark.parametrize("spot, strike, maturity, rate, vol", GRID)
def test_no_arbitrage_bounds(spot, strike, maturity, rate, vol):
    for option_type in ("call", "put"):
        lower, upper = price_bounds(spot, strike, maturity, rate, option_type)
        price = bs_price(spot, strike, maturity, rate, vol, option_type)
        assert lower - 1e-10 <= price <= upper + 1e-10


def test_price_increases_with_volatility():
    prices = [call_price(100, 100, 1, 0.05, v) for v in (0.1, 0.2, 0.4, 0.8)]
    assert prices == sorted(prices)


# --- Boundary cases ----------------------------------------------------------

@pytest.mark.parametrize("spot", [80.0, 100.0, 120.0])
def test_expiry_returns_intrinsic_value(spot):
    assert call_price(spot, 100.0, 0.0, 0.05, 0.2) == max(spot - 100.0, 0.0)
    assert put_price(spot, 100.0, 0.0, 0.05, 0.2) == max(100.0 - spot, 0.0)


def test_zero_volatility_discounted_deterministic_payoff():
    discounted_strike = 100.0 * math.exp(-0.05 * 1.0)  # 95.1229...
    assert call_price(100.0, 100.0, 1.0, 0.05, 0.0) == pytest.approx(100.0 - discounted_strike)
    assert put_price(100.0, 100.0, 1.0, 0.05, 0.0) == 0.0
    assert put_price(90.0, 100.0, 1.0, 0.05, 0.0) == pytest.approx(discounted_strike - 90.0)
    assert call_price(90.0, 100.0, 1.0, 0.05, 0.0) == 0.0


def test_small_volatility_approaches_zero_volatility_limit():
    assert call_price(100, 100, 1, 0.05, 1e-6) == pytest.approx(call_price(100, 100, 1, 0.05, 0.0), abs=1e-8)


def test_negative_rate_is_allowed():
    price = call_price(100.0, 100.0, 1.0, -0.01, 0.2)
    assert math.isfinite(price) and price > 0.0


# --- Invalid inputs ----------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        dict(spot=0.0), dict(spot=-1.0), dict(strike=0.0), dict(maturity=-0.1),
        dict(volatility=-0.2), dict(spot=float("nan")), dict(rate=float("inf")),
        dict(option_type="straddle"), dict(spot="abc"),
    ],
)
def test_invalid_inputs_raise(kwargs):
    base = dict(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2, option_type="call")
    base.update(kwargs)
    with pytest.raises(ValueError):
        bs_price(**base)
