"""Tests for implied volatility."""

import math

import pytest

from bs_hedging.implied_vol import ImpliedVolatilityError, implied_volatility
from bs_hedging.pricing import bs_price, price_bounds

ROUND_TRIP_CASES = [
    (spot, strike, maturity, rate, vol, kind)
    for spot in (80.0, 100.0, 125.0)
    for strike in (90.0, 100.0, 110.0)
    for maturity in (0.1, 1.0, 2.5)
    for rate in (-0.005, 0.03)
    for vol in (0.08, 0.2, 0.55, 1.5)
    for kind in ("call", "put")
]


@pytest.mark.parametrize("spot, strike, maturity, rate, vol, kind", ROUND_TRIP_CASES)
def test_round_trip(spot, strike, maturity, rate, vol, kind):
    """known vol -> price -> recovered vol -> repriced option."""
    price = bs_price(spot, strike, maturity, rate, vol, kind)
    lower, _ = price_bounds(spot, strike, maturity, rate, kind)
    if price - lower < 1e-9:  # deep ITM/OTM: price carries almost no volatility information
        pytest.skip("price indistinguishable from its lower bound at double precision")
    recovered = implied_volatility(price, spot, strike, maturity, rate, kind)
    repriced = bs_price(spot, strike, maturity, rate, recovered, kind)
    assert repriced == pytest.approx(price, abs=1e-7)
    assert recovered == pytest.approx(vol, rel=1e-5)


def test_well_conditioned_round_trip_is_very_accurate():
    price = bs_price(100, 100, 1, 0.05, 0.2, "call")
    assert implied_volatility(price, 100, 100, 1, 0.05, "call") == pytest.approx(0.2, abs=1e-9)


def test_tolerance_is_configurable():
    price = bs_price(100, 100, 1, 0.05, 0.2, "put")
    loose = implied_volatility(price, 100, 100, 1, 0.05, "put", tolerance=1e-3)
    assert loose == pytest.approx(0.2, abs=2e-3)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_price_above_upper_bound_rejected(kind):
    _, upper = price_bounds(100, 100, 1, 0.05, kind)
    with pytest.raises(ImpliedVolatilityError):
        implied_volatility(upper, 100, 100, 1, 0.05, kind)
    with pytest.raises(ImpliedVolatilityError):
        implied_volatility(upper + 1.0, 100, 100, 1, 0.05, kind)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_price_below_lower_bound_rejected(kind):
    lower, _ = price_bounds(120 if kind == "call" else 80, 100, 1, 0.05, kind)
    spot = 120 if kind == "call" else 80
    with pytest.raises(ImpliedVolatilityError):
        implied_volatility(lower - 0.5, spot, 100, 1, 0.05, kind)


def test_negative_and_non_finite_prices_rejected():
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ImpliedVolatilityError):
            implied_volatility(bad, 100, 100, 1, 0.05, "call")


def test_price_at_lower_bound_returns_zero_volatility():
    lower = 100.0 - 100.0 * math.exp(-0.05)
    assert implied_volatility(lower, 100, 100, 1, 0.05, "call") == 0.0


def test_expiry_rejected():
    with pytest.raises(ImpliedVolatilityError):
        implied_volatility(5.0, 105, 100, 0.0, 0.05, "call")


def test_bracketing_failure_reports_search_limit():
    price = bs_price(100, 100, 1, 0.05, 3.0, "call")  # needs sigma = 3
    with pytest.raises(ImpliedVolatilityError, match="search limit"):
        implied_volatility(price, 100, 100, 1, 0.05, "call", max_upper=2.0)
    # A larger search limit recovers it: the limit is not a mathematical bound.
    assert implied_volatility(price, 100, 100, 1, 0.05, "call", max_upper=8.0) == pytest.approx(3.0, rel=1e-6)


def test_invalid_contract_inputs_rejected():
    with pytest.raises(ValueError):
        implied_volatility(5.0, -100, 100, 1, 0.05, "call")
    with pytest.raises(ValueError):
        implied_volatility(5.0, 100, 100, 1, 0.05, "binary")
