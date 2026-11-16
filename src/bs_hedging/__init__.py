"""bs_hedging: Black-Scholes pricing, Greeks, implied volatility and a
simulated delta-hedging study for European options (no dividends).

Plotting lives in :mod:`bs_hedging.plotting` and is not imported here, so the
numerical modules can be used without Matplotlib.
"""

from .greeks import BoundaryGreekError, all_greeks, delta, gamma, rho, scaled_greeks, theta, vega
from .hedging import HedgeResult, compare_strategies, rebalance_steps, simulate_delta_hedge
from .implied_vol import ImpliedVolatilityError, implied_volatility
from .metrics import rmse_to_zero, summarize_pnl
from .pricing import bs_price, call_price, price_bounds, put_price
from .simulation import simulate_gbm_paths

__version__ = "0.1.0"

__all__ = [
    "BoundaryGreekError",
    "HedgeResult",
    "ImpliedVolatilityError",
    "all_greeks",
    "bs_price",
    "call_price",
    "compare_strategies",
    "delta",
    "gamma",
    "implied_volatility",
    "price_bounds",
    "put_price",
    "rebalance_steps",
    "rho",
    "rmse_to_zero",
    "scaled_greeks",
    "simulate_delta_hedge",
    "simulate_gbm_paths",
    "summarize_pnl",
    "theta",
    "vega",
]
