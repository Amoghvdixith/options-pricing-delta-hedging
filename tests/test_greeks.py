"""Tests for the Greeks: finite differences of the price, call/put relations,
an independent textbook reference, boundary policy and unit scaling."""

import math

import numpy as np
import pytest

from bs_hedging.greeks import (
    BoundaryGreekError,
    all_greeks,
    call_delta_array,
    delta,
    gamma,
    rho,
    scaled_greeks,
    theta,
    vega,
)
from bs_hedging.pricing import bs_price

INTERIOR_POINTS = [
    (100.0, 100.0, 1.0, 0.05, 0.20),
    (80.0, 100.0, 0.5, 0.02, 0.30),
    (120.0, 100.0, 2.0, -0.01, 0.15),
    (95.0, 105.0, 0.25, 0.08, 0.45),
]


# --- Interior: finite differences of the pricing function -------------------

@pytest.mark.parametrize("S, K, T, r, v", INTERIOR_POINTS)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_finite_difference_checks(S, K, T, r, v, kind):
    def price(S=S, T=T, r=r, v=v):
        return bs_price(S, K, T, r, v, kind)

    hs, hv, hr, ht = 1e-3 * S, 1e-4, 1e-5, 1e-5
    fd_delta = (price(S=S + hs) - price(S=S - hs)) / (2 * hs)
    fd_gamma = (price(S=S + hs) - 2 * price() + price(S=S - hs)) / hs**2
    fd_vega = (price(v=v + hv) - price(v=v - hv)) / (2 * hv)
    fd_rho = (price(r=r + hr) - price(r=r - hr)) / (2 * hr)
    fd_theta = -(price(T=T + ht) - price(T=T - ht)) / (2 * ht)  # dV/dt = -dV/dT

    assert delta(S, K, T, r, v, kind) == pytest.approx(fd_delta, rel=1e-5, abs=1e-7)
    assert gamma(S, K, T, r, v) == pytest.approx(fd_gamma, rel=1e-4, abs=1e-7)
    assert vega(S, K, T, r, v) == pytest.approx(fd_vega, rel=1e-6, abs=1e-7)
    assert rho(S, K, T, r, v, kind) == pytest.approx(fd_rho, rel=1e-6, abs=1e-7)
    assert theta(S, K, T, r, v, kind) == pytest.approx(fd_theta, rel=1e-5, abs=1e-6)


def test_hull_textbook_reference_greeks():
    """Hull's running example: S=49, K=50, r=5%, sigma=20%, T=20 weeks (0.3846 yr),
    call. Reported (rounded): delta 0.522, gamma 0.066, vega 12.1 (per 1.0 vol),
    theta -4.31 per year, rho 8.91 (per 1.0 rate)."""
    args = (49.0, 50.0, 0.3846, 0.05, 0.20)
    assert delta(*args, "call") == pytest.approx(0.522, abs=0.0006)
    assert gamma(*args) == pytest.approx(0.066, abs=0.0006)
    assert vega(*args) == pytest.approx(12.1, abs=0.06)
    assert theta(*args, "call") == pytest.approx(-4.31, abs=0.006)
    assert rho(*args, "call") == pytest.approx(8.91, abs=0.006)


# --- Call/put relationships --------------------------------------------------

@pytest.mark.parametrize("S, K, T, r, v", INTERIOR_POINTS)
def test_call_put_relationships(S, K, T, r, v):
    assert delta(S, K, T, r, v, "call") - delta(S, K, T, r, v, "put") == pytest.approx(1.0)
    # From parity C - P = S - K e^{-rT}:
    assert theta(S, K, T, r, v, "call") - theta(S, K, T, r, v, "put") == pytest.approx(
        -r * K * math.exp(-r * T)
    )
    assert rho(S, K, T, r, v, "call") - rho(S, K, T, r, v, "put") == pytest.approx(
        T * K * math.exp(-r * T)
    )
    assert 0.0 < delta(S, K, T, r, v, "call") < 1.0
    assert gamma(S, K, T, r, v) > 0.0 and vega(S, K, T, r, v) > 0.0


def test_vectorised_delta_matches_scalar_delta():
    spots = np.array([70.0, 90.0, 100.0, 110.0, 150.0])
    expected = [delta(s, 100.0, 0.7, 0.03, 0.25, "call") for s in spots]
    np.testing.assert_allclose(call_delta_array(spots, 100.0, 0.7, 0.03, 0.25), expected, rtol=1e-12)


# --- Boundary policy (kept separate from interior checks) --------------------

@pytest.mark.parametrize("func", [delta, gamma, vega, theta, rho])
def test_expiry_raises_boundary_error(func):
    with pytest.raises(BoundaryGreekError):
        func(100.0, 100.0, 0.0, 0.05, 0.2)


@pytest.mark.parametrize("func", [delta, gamma, vega, theta, rho])
def test_zero_volatility_raises_boundary_error(func):
    with pytest.raises(BoundaryGreekError):
        func(100.0, 100.0, 1.0, 0.05, 0.0)


def test_vectorised_delta_boundary_raises():
    with pytest.raises(BoundaryGreekError):
        call_delta_array(np.array([100.0]), 100.0, 0.0, 0.05, 0.2)


def test_boundary_error_is_value_error():
    assert issubclass(BoundaryGreekError, ValueError)


# --- Unit scaling -------------------------------------------------------------

def test_scaled_units():
    raw = all_greeks(100.0, 100.0, 1.0, 0.05, 0.2, "call")
    scaled = scaled_greeks(raw)
    assert scaled["vega_per_vol_point"] == pytest.approx(raw["vega_per_unit_vol"] / 100)
    assert scaled["rho_per_rate_point"] == pytest.approx(raw["rho_per_unit_rate"] / 100)
    assert scaled["theta_per_calendar_day"] == pytest.approx(raw["theta_per_year"] / 365)


def test_vega_per_point_matches_one_point_price_change():
    """A 1 percentage-point volatility move changes the price by roughly vega/100."""
    raw = all_greeks(100.0, 100.0, 1.0, 0.05, 0.2, "call")
    actual_change = bs_price(100, 100, 1, 0.05, 0.21) - bs_price(100, 100, 1, 0.05, 0.20)
    assert scaled_greeks(raw)["vega_per_vol_point"] == pytest.approx(actual_change, rel=0.01)
