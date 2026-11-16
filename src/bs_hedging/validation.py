"""Input validation shared by the pricing, Greeks and implied-volatility modules.

All checks raise ``ValueError`` with a message that names the offending input,
so that command-line users get a clear explanation instead of a NaN.
"""

from __future__ import annotations

import math

VALID_OPTION_TYPES = ("call", "put")


def validate_option_type(option_type: str) -> str:
    """Return the normalised option type ("call" or "put") or raise ``ValueError``."""
    if not isinstance(option_type, str):
        raise ValueError(f"option_type must be a string, got {type(option_type).__name__}")
    normalised = option_type.strip().lower()
    if normalised not in VALID_OPTION_TYPES:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    return normalised


def _require_finite(name: str, value: float) -> float:
    """Convert ``value`` to float and check that it is finite."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def validate_option_inputs(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: str = "call",
) -> tuple[float, float, float, float, float, str]:
    """Validate Black-Scholes inputs and return them as floats.

    Rules
    -----
    * every numeric input must be finite;
    * ``spot > 0`` and ``strike > 0``;
    * ``maturity >= 0`` (years; 0 means at expiry);
    * ``volatility >= 0`` (annualised, as a decimal, e.g. 0.20);
    * ``rate`` may be negative (continuously compounded, annualised);
    * ``option_type`` must be "call" or "put".
    """
    spot = _require_finite("spot", spot)
    strike = _require_finite("strike", strike)
    maturity = _require_finite("maturity", maturity)
    rate = _require_finite("rate", rate)
    volatility = _require_finite("volatility", volatility)
    option_type = validate_option_type(option_type)

    if spot <= 0.0:
        raise ValueError(f"spot must be > 0, got {spot}")
    if strike <= 0.0:
        raise ValueError(f"strike must be > 0, got {strike}")
    if maturity < 0.0:
        raise ValueError(f"maturity must be >= 0, got {maturity}")
    if volatility < 0.0:
        raise ValueError(f"volatility must be >= 0, got {volatility}")
    return spot, strike, maturity, rate, volatility, option_type
