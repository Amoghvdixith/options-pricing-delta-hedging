"""Matplotlib charts. Uses the non-interactive "Agg" backend so figures can be
saved without a graphical desktop session. Every function saves one PNG and
closes its figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .greeks import delta  # noqa: E402
from .pricing import bs_price  # noqa: E402

DPI = 150
FIGSIZE = (8.0, 5.0)
COLORS = {"call": "#1f77b4", "put": "#d62728", "daily": "#1f77b4", "weekly": "#ff7f0e"}


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_option_prices(path, strike, maturity, rate, volatility, spots: Sequence[float]) -> Path:
    """European call and put price versus stock price (analytical, not simulated)."""
    spots = np.asarray(spots, dtype=float)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for kind in ("call", "put"):
        prices = [bs_price(s, strike, maturity, rate, volatility, kind) for s in spots]
        ax.plot(spots, prices, color=COLORS[kind], label=f"European {kind}")
    ax.axvline(strike, color="grey", linestyle=":", label=f"Strike K = {strike:g}")
    _style(ax, f"Black-Scholes prices (K={strike:g}, T={maturity:g} yr, r={rate:g}, sigma={volatility:g})",
           "Stock price S (currency units)", "Option price (currency units per share)")
    ax.legend()
    return _save(fig, path)


def plot_option_deltas(path, strike, maturity, rate, volatility, spots: Sequence[float]) -> Path:
    """Call and put delta versus stock price (analytical, not simulated)."""
    spots = np.asarray(spots, dtype=float)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for kind in ("call", "put"):
        values = [delta(s, strike, maturity, rate, volatility, kind) for s in spots]
        ax.plot(spots, values, color=COLORS[kind], label=f"{kind.capitalize()} delta")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(strike, color="grey", linestyle=":", label=f"Strike K = {strike:g}")
    _style(ax, f"Black-Scholes delta (K={strike:g}, T={maturity:g} yr, r={rate:g}, sigma={volatility:g})",
           "Stock price S (currency units)", "Delta (shares per option)")
    ax.legend()
    return _save(fig, path)


def plot_sample_paths(path, times: np.ndarray, paths: np.ndarray, strike: float | None = None) -> Path:
    """A few simulated GBM stock-price paths."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for row in paths:
        ax.plot(times, row, linewidth=1.0, alpha=0.85)
    if strike is not None:
        ax.axhline(strike, color="black", linestyle="--", linewidth=1.0, label=f"Strike K = {strike:g}")
        ax.legend()
    _style(ax, f"SIMULATED GBM stock-price paths (n = {len(paths)} sample paths)",
           "Time (years)", "Simulated stock price (currency units)")
    return _save(fig, path)


def plot_pnl_histograms(path, pnl_by_strategy: dict[str, np.ndarray], title: str, n_bins: int = 80) -> Path:
    """Overlaid terminal P&L histograms using COMMON bin edges."""
    all_values = np.concatenate([np.asarray(v) for v in pnl_by_strategy.values()])
    edges = np.linspace(all_values.min(), all_values.max(), n_bins + 1)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for name, values in pnl_by_strategy.items():
        ax.hist(values, bins=edges, alpha=0.55, color=COLORS.get(name), label=f"{name} rebalancing",
                edgecolor="none")
    ax.axvline(0.0, color="black", linewidth=1.0, label="Zero P&L (perfect hedge)")
    _style(ax, title, "Terminal hedging P&L (currency units per option, undiscounted)",
           "Number of simulated paths")
    ax.legend()
    return _save(fig, path)


def plot_rmse_comparison(path, rows: list[dict]) -> Path:
    """Grouped bar chart of RMSE (vs zero P&L) by strategy and cost scenario.

    ``rows`` items need keys: ``scenario``, ``strategy``, ``rmse_pnl``.
    """
    scenarios = list(dict.fromkeys(r["scenario"] for r in rows))
    strategies = list(dict.fromkeys(r["strategy"] for r in rows))
    lookup = {(r["scenario"], r["strategy"]): r["rmse_pnl"] for r in rows}
    x = np.arange(len(scenarios))
    width = 0.8 / max(len(strategies), 1)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, strategy in enumerate(strategies):
        values = [lookup.get((s, strategy), np.nan) for s in scenarios]
        bars = ax.bar(x + (i - (len(strategies) - 1) / 2) * width, values, width,
                      color=COLORS.get(strategy), label=f"{strategy} rebalancing")
        ax.bar_label(bars, fmt="%.3f", padding=2)
    ax.set_xticks(x, [s.replace("_", " ") for s in scenarios])
    finite = [v for v in lookup.values() if np.isfinite(v)]
    if finite:
        ax.set_ylim(0.0, max(finite) * 1.25)  # head-room for bar labels and legend
    _style(ax, "SIMULATED hedging RMSE of terminal P&L (relative to zero)",
           "Cost scenario", "RMSE (currency units per option)")
    ax.legend()
    return _save(fig, path)
