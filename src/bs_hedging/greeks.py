"""Black-Scholes Greeks for European options (no dividends).

Raw units returned by the functions in this module
--------------------------------------------------
* delta : change in option value per 1.0 change in the stock price.
* gamma : change in delta per 1.0 change in the stock price.
* vega  : change in option value per 1.0 absolute change in volatility
          (i.e. from 0.20 to 1.20). Divide by 100 for "per volatility point".
* theta : dV/dt in calendar time, per YEAR, with S, sigma and r held fixed
          (theta = -dV/dT). Divide by 365 for "per calendar day".
* rho   : change in option value per 1.0 absolute change in the rate.
          Divide by 100 for "per percentage point".

Use :func:`scaled_greeks` for human-readable units; it always labels its keys.

Boundary policy
---------------
Greeks are only computed for ``maturity > 0`` and ``volatility > 0``. At
``T == 0`` or ``sigma == 0`` the price is a kinked function of the spot (and
gamma is a Dirac delta at the kink), so the functions raise
:class:`BoundaryGreekError` rather than substituting an arbitrary epsilon.
The hedging simulation never asks for a Greek at expiry.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr

from .pricing import d1_d2
from .validation import validate_option_inputs

DAYS_PER_YEAR_FOR_THETA = 365.0


class BoundaryGreekError(ValueError):
    """Raised when a Greek is requested at T == 0 or sigma == 0."""


def _prepare(spot, strike, maturity, rate, volatility, option_type="call"):
    """Validate inputs, enforce the boundary policy and return (inputs, d1, d2)."""
    spot, strike, maturity, rate, volatility, option_type = validate_option_inputs(
        spot, strike, maturity, rate, volatility, option_type
    )
    if maturity == 0.0:
        raise BoundaryGreekError(
            "Greeks are not provided at maturity == 0: the payoff is kinked at the strike, "
            "so delta is discontinuous and gamma is not a finite number there."
        )
    if volatility == 0.0:
        raise BoundaryGreekError(
            "Greeks are not provided at volatility == 0: the price is kinked at "
            "S = K e^{-rT}, so the analytical Greeks are not finite/continuous there."
        )
    d1, d2 = d1_d2(spot, strike, maturity, rate, volatility)
    return spot, strike, maturity, rate, volatility, option_type, d1, d2


def _pdf(x: float) -> float:
    """Standard normal density."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def delta(spot, strike, maturity, rate, volatility, option_type="call") -> float:
    """Delta: N(d1) for a call, N(d1) - 1 for a put."""
    *_, option_type, d1, _ = _prepare(spot, strike, maturity, rate, volatility, option_type)
    return float(ndtr(d1)) if option_type == "call" else float(ndtr(d1)) - 1.0


def gamma(spot, strike, maturity, rate, volatility) -> float:
    """Gamma (identical for calls and puts): n(d1) / (S sigma sqrt(T))."""
    spot, _, maturity, _, volatility, _, d1, _ = _prepare(spot, strike, maturity, rate, volatility)
    return _pdf(d1) / (spot * volatility * math.sqrt(maturity))


def vega(spot, strike, maturity, rate, volatility) -> float:
    """Vega per 1.0 change in volatility (identical for calls and puts): S n(d1) sqrt(T)."""
    spot, _, maturity, _, _, _, d1, _ = _prepare(spot, strike, maturity, rate, volatility)
    return spot * _pdf(d1) * math.sqrt(maturity)


def theta(spot, strike, maturity, rate, volatility, option_type="call") -> float:
    """Theta per year of calendar time (dV/dt = -dV/dT)."""
    spot, strike, maturity, rate, volatility, option_type, d1, d2 = _prepare(
        spot, strike, maturity, rate, volatility, option_type
    )
    decay = -spot * _pdf(d1) * volatility / (2.0 * math.sqrt(maturity))
    discounted_strike = strike * math.exp(-rate * maturity)
    if option_type == "call":
        return decay - rate * discounted_strike * float(ndtr(d2))
    return decay + rate * discounted_strike * float(ndtr(-d2))


def rho(spot, strike, maturity, rate, volatility, option_type="call") -> float:
    """Rho per 1.0 change in the interest rate."""
    _, strike, maturity, rate, _, option_type, _, d2 = _prepare(
        spot, strike, maturity, rate, volatility, option_type
    )
    discounted_strike = strike * math.exp(-rate * maturity)
    if option_type == "call":
        return maturity * discounted_strike * float(ndtr(d2))
    return -maturity * discounted_strike * float(ndtr(-d2))


def call_delta_array(
    spots: np.ndarray, strike: float, maturity: float, rate: float, volatility: float
) -> np.ndarray:
    """Vectorised call delta N(d1) for an array of spot prices.

    Used by the hedging simulation, where thousands of paths are rebalanced at
    once. The same boundary policy applies: ``maturity`` and ``volatility``
    must be strictly positive. Tests check this against :func:`delta`.
    """
    if not (maturity > 0.0 and volatility > 0.0):
        raise BoundaryGreekError("call_delta_array requires maturity > 0 and volatility > 0")
    spots = np.asarray(spots, dtype=float)
    if np.any(~np.isfinite(spots)) or np.any(spots <= 0.0):
        raise ValueError("all spot prices must be finite and > 0")
    vol_sqrt_t = volatility * math.sqrt(maturity)
    d1 = (np.log(spots / strike) + (rate + 0.5 * volatility**2) * maturity) / vol_sqrt_t
    return ndtr(d1)


def all_greeks(spot, strike, maturity, rate, volatility, option_type="call") -> dict[str, float]:
    """Return all raw Greeks in a dictionary (raw units, see module docstring)."""
    return {
        "delta": delta(spot, strike, maturity, rate, volatility, option_type),
        "gamma": gamma(spot, strike, maturity, rate, volatility),
        "vega_per_unit_vol": vega(spot, strike, maturity, rate, volatility),
        "theta_per_year": theta(spot, strike, maturity, rate, volatility, option_type),
        "rho_per_unit_rate": rho(spot, strike, maturity, rate, volatility, option_type),
    }


def scaled_greeks(raw: dict[str, float]) -> dict[str, float]:
    """Convert raw Greeks from :func:`all_greeks` to human-readable units.

    * vega per 1 volatility percentage point = raw vega / 100
    * rho per 1 rate percentage point        = raw rho / 100
    * theta per calendar day                 = raw theta / 365
    """
    return {
        "vega_per_vol_point": raw["vega_per_unit_vol"] / 100.0,
        "rho_per_rate_point": raw["rho_per_unit_rate"] / 100.0,
        "theta_per_calendar_day": raw["theta_per_year"] / DAYS_PER_YEAR_FOR_THETA,
    }
