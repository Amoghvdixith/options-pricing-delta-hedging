"""Implied volatility by Brent's method on the project's own Black-Scholes price.

Policy
------
1. Inputs are validated as for pricing. ``maturity == 0`` is rejected: at expiry
   the price equals intrinsic value for every volatility, so no implied
   volatility is identified.
2. The market price must satisfy the no-arbitrage bounds of
   :func:`bs_hedging.pricing.price_bounds`:
   * price below the lower bound or at/above the upper bound -> ``ImpliedVolatilityError``
     (no finite volatility reproduces it);
   * price equal to the lower bound (within ``bound_tolerance``) -> returns 0.0,
     because sigma = 0 attains the lower bound and the price is strictly
     increasing in sigma for sigma > 0 when T > 0.
3. Bracketing: the root is searched in [0, upper]. ``upper`` starts at
   ``initial_upper`` and is doubled until the model price at ``upper`` exceeds
   the target, or until ``max_upper`` is exceeded. ``max_upper`` is a
   computational search limit, **not** a mathematical bound on volatility; a
   failure to bracket raises ``ImpliedVolatilityError`` explaining this.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq

from .pricing import bs_price, price_bounds
from .validation import validate_option_inputs


class ImpliedVolatilityError(ValueError):
    """Raised when no unique finite implied volatility can be returned."""


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    option_type: str = "call",
    tolerance: float = 1e-10,
    initial_upper: float = 1.0,
    max_upper: float = 10.0,
    bound_tolerance: float = 1e-12,
) -> float:
    """Return the Black-Scholes implied volatility of a European option price."""
    spot, strike, maturity, rate, _, option_type = validate_option_inputs(
        spot, strike, maturity, rate, 0.0, option_type
    )
    try:
        price = float(price)
    except (TypeError, ValueError) as exc:
        raise ImpliedVolatilityError(f"price must be a real number, got {price!r}") from exc
    if not math.isfinite(price):
        raise ImpliedVolatilityError(f"price must be finite, got {price!r}")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be > 0")
    if initial_upper <= 0.0 or max_upper < initial_upper:
        raise ValueError("require 0 < initial_upper <= max_upper")
    if maturity == 0.0:
        raise ImpliedVolatilityError(
            "maturity == 0: the option price equals intrinsic value for every volatility, "
            "so implied volatility is not identified."
        )

    lower, upper = price_bounds(spot, strike, maturity, rate, option_type)
    if price < lower - bound_tolerance:
        raise ImpliedVolatilityError(
            f"price {price} is below the no-arbitrage lower bound {lower:.10g}; "
            "no volatility reproduces it."
        )
    if price >= upper:
        raise ImpliedVolatilityError(
            f"price {price} is at or above the no-arbitrage upper bound {upper:.10g}; "
            "no finite volatility reproduces it."
        )
    if abs(price - lower) <= bound_tolerance:
        return 0.0

    def objective(sigma: float) -> float:
        return bs_price(spot, strike, maturity, rate, sigma, option_type) - price

    sigma_high = initial_upper
    while objective(sigma_high) < 0.0:
        sigma_high *= 2.0
        if sigma_high > max_upper:
            raise ImpliedVolatilityError(
                f"could not bracket the implied volatility below the search limit "
                f"max_upper={max_upper}. This is a computational search limit, not a "
                "mathematical bound; increase max_upper if a larger volatility is plausible."
            )

    return float(brentq(objective, 0.0, sigma_high, xtol=tolerance, maxiter=500))
