"""Analytical Black-Scholes prices for European calls and puts (no dividends).

Notation
--------
S  spot price, K strike, T time to maturity in years, r continuously
compounded risk-free rate, sigma annualised volatility.

    d1 = [ln(S/K) + (r + sigma^2/2) T] / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)
    call = S N(d1) - K e^{-rT} N(d2)
    put  = K e^{-rT} N(-d2) - S N(-d1)

Boundary cases
--------------
* ``T == 0``: the option is worth its intrinsic value, max(S-K, 0) or max(K-S, 0).
* ``sigma == 0`` and ``T > 0``: the stock grows deterministically at r under the
  pricing measure, so the price is the discounted deterministic payoff,
  max(S - K e^{-rT}, 0) for a call and max(K e^{-rT} - S, 0) for a put.
"""

from __future__ import annotations

import math

from scipy.special import ndtr

from .validation import validate_option_inputs


def d1_d2(spot: float, strike: float, maturity: float, rate: float, volatility: float) -> tuple[float, float]:
    """Return (d1, d2). Requires ``maturity > 0`` and ``volatility > 0``."""
    if maturity <= 0.0 or volatility <= 0.0:
        raise ValueError("d1/d2 are only defined for maturity > 0 and volatility > 0")
    vol_sqrt_t = volatility * math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: str = "call",
) -> float:
    """Black-Scholes price of one European option on one share."""
    spot, strike, maturity, rate, volatility, option_type = validate_option_inputs(
        spot, strike, maturity, rate, volatility, option_type
    )

    if maturity == 0.0:
        if option_type == "call":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)

    discounted_strike = strike * math.exp(-rate * maturity)
    if volatility == 0.0:
        if option_type == "call":
            return max(spot - discounted_strike, 0.0)
        return max(discounted_strike - spot, 0.0)

    d1, d2 = d1_d2(spot, strike, maturity, rate, volatility)
    if option_type == "call":
        return float(spot * ndtr(d1) - discounted_strike * ndtr(d2))
    return float(discounted_strike * ndtr(-d2) - spot * ndtr(-d1))


def call_price(spot: float, strike: float, maturity: float, rate: float, volatility: float) -> float:
    """Convenience wrapper for a European call."""
    return bs_price(spot, strike, maturity, rate, volatility, "call")


def put_price(spot: float, strike: float, maturity: float, rate: float, volatility: float) -> float:
    """Convenience wrapper for a European put."""
    return bs_price(spot, strike, maturity, rate, volatility, "put")


def price_bounds(
    spot: float, strike: float, maturity: float, rate: float, option_type: str = "call"
) -> tuple[float, float]:
    """Model-free no-arbitrage bounds (lower, upper) for a European option, no dividends.

    * call: max(S - K e^{-rT}, 0) <= C <= S
    * put : max(K e^{-rT} - S, 0) <= P <= K e^{-rT}
    """
    spot, strike, maturity, rate, _, option_type = validate_option_inputs(
        spot, strike, maturity, rate, 0.0, option_type
    )
    discounted_strike = strike * math.exp(-rate * maturity)
    if option_type == "call":
        return max(spot - discounted_strike, 0.0), spot
    return max(discounted_strike - spot, 0.0), discounted_strike
