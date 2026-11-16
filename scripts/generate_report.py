"""Build results/report.md and the summary figures from SAVED outputs only.

Run from the project root after both experiments:

    python scripts/generate_report.py

Required inputs (in the results directory): baseline_summary.csv,
transaction_cost_summary.csv, baseline_terminal_pnl.csv,
transaction_cost_terminal_pnl.csv and run_metadata.json. If any is missing the
script stops with a non-zero exit code and lists what is missing.
example_option.json is optional; if absent, that report section is marked missing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
import pandas as pd
from _common import display_path, environment_info, resolve_output_dir

from bs_hedging.plotting import plot_option_deltas, plot_option_prices, plot_rmse_comparison

REQUIRED = (
    "baseline_summary.csv",
    "transaction_cost_summary.csv",
    "baseline_terminal_pnl.csv",
    "transaction_cost_terminal_pnl.csv",
    "run_metadata.json",
)
COMPARABLE_KEYS = (
    "initial_stock_price", "strike", "maturity_years", "risk_free_rate", "simulation_drift",
    "simulation_volatility", "hedging_volatility", "trading_steps", "n_paths", "random_seed", "strategies",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate results/report.md and summary figures from saved experiment outputs.",
    )
    parser.add_argument("--results-dir", default=None,
                        help="Directory containing experiment outputs (default: <project root>/results).")
    return parser


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def results_table(summary: pd.DataFrame) -> str:
    header = ("| Scenario | Strategy | Hedge dates (incl. t=0) | Mean P&L | Std (ddof=1) | Std. error of mean | "
              "RMSE vs 0 | 5th pct | Median | 95th pct | Min | Max | Mean cost |")
    lines = [header, "|" + "---|" * 13]
    for _, r in summary.iterrows():
        se = r["std_pnl"] / math.sqrt(r["n_paths"])
        lines.append(
            f"| {r['scenario']} | {r['strategy']} | {int(r['n_rebalance_dates'])} | {fmt(r['mean_pnl'])} | "
            f"{fmt(r['std_pnl'])} | {fmt(se)} | {fmt(r['rmse_pnl'])} | {fmt(r['p05_pnl'])} | "
            f"{fmt(r['median_pnl'])} | {fmt(r['p95_pnl'])} | {fmt(r['min_pnl'])} | {fmt(r['max_pnl'])} | "
            f"{fmt(r['mean_transaction_cost'])} |"
        )
    return "\n".join(lines)


def paired_analysis(pnl: dict[str, pd.DataFrame], strategies: list[str]) -> tuple[str, dict]:
    """Path-by-path comparisons that are possible because all runs share the same paths."""
    facts: dict = {}
    lines = ["| Scenario | Share of paths with smaller absolute P&L under daily than weekly | "
             "Mean (daily - weekly) P&L | Std. error of that mean |", "|---|---|---|---|"]
    for scenario, frame in pnl.items():
        daily, weekly = frame["daily_terminal_pnl"].to_numpy(), frame["weekly_terminal_pnl"].to_numpy()
        share = float(np.mean(np.abs(daily) < np.abs(weekly)))
        diff = daily - weekly
        se = float(np.std(diff, ddof=1) / math.sqrt(diff.size))
        facts[scenario] = {"share_daily_closer": share, "mean_diff": float(diff.mean()), "se_diff": se}
        lines.append(f"| {scenario} | {share:.2%} | {fmt(diff.mean())} | {fmt(se)} |")

    lines += ["", "| Strategy | Mean nominal cost paid | Mean P&L reduction caused by costs "
              "(baseline P&L - cost-scenario P&L, same path) |", "|---|---|---|"]
    base, cost = pnl["baseline"], pnl["transaction_cost"]
    for strategy in strategies:
        reduction = (base[f"{strategy}_terminal_pnl"] - cost[f"{strategy}_terminal_pnl"]).to_numpy()
        nominal = cost[f"{strategy}_transaction_cost"].to_numpy()
        facts[f"cost_{strategy}"] = {"reduction": float(reduction.mean()), "nominal": float(nominal.mean())}
        lines.append(f"| {strategy} | {fmt(nominal.mean())} | {fmt(reduction.mean())} |")
    return "\n".join(lines), facts


def interpretation(summary: pd.DataFrame, facts: dict) -> list[str]:
    """Plain statements derived only from the saved numbers."""
    get = lambda s, k, col: float(summary[(summary.scenario == s) & (summary.strategy == k)][col].iloc[0])  # noqa: E731
    out = []
    for scenario in ("baseline", "transaction_cost"):
        d, w = get(scenario, "daily", "rmse_pnl"), get(scenario, "weekly", "rmse_pnl")
        relation = "lower" if d < w else "higher" if d > w else "equal"
        out.append(
            f"- In the **{scenario}** scenario, daily rebalancing had a {relation} RMSE than weekly "
            f"rebalancing ({fmt(d)} vs {fmt(w)}; ratio weekly/daily = {w / d:.2f})."
        )
        f = facts[scenario]
        out.append(
            f"  On {f['share_daily_closer']:.1%} of the identical simulated paths the daily strategy ended "
            f"closer to zero P&L than the weekly strategy, so it was not better on every path."
        )
    for strategy in ("daily", "weekly"):
        m0, m1 = get("baseline", strategy, "mean_pnl"), get("transaction_cost", strategy, "mean_pnl")
        c = facts[f"cost_{strategy}"]
        out.append(
            f"- With 5 bp costs, the mean P&L of **{strategy}** rebalancing moved from {fmt(m0)} to "
            f"{fmt(m1)}; mean nominal costs were {fmt(c['nominal'])} and the same-path P&L reduction was "
            f"{fmt(c['reduction'])} (slightly larger than nominal costs because cash paid for costs "
            f"would otherwise have earned interest)."
        )
    dc, wc = facts["cost_daily"]["nominal"], facts["cost_weekly"]["nominal"]
    out.append(
        f"- Daily rebalancing paid {dc / wc:.2f} times the mean transaction cost of weekly rebalancing "
        f"({fmt(dc)} vs {fmt(wc)}), illustrating the trade-off between hedging error and trading cost "
        f"under this particular cost level and parameter set."
    )
    for scenario in ("baseline", "transaction_cost"):
        for strategy in ("daily", "weekly"):
            mean = get(scenario, strategy, "mean_pnl")
            se = get(scenario, strategy, "std_pnl") / math.sqrt(get(scenario, strategy, "n_paths"))
            inside = abs(mean) <= 2 * se
            out.append(
                f"- {scenario}/{strategy}: mean P&L {fmt(mean)} with standard error {fmt(se)} -> "
                + ("not distinguishable from zero at roughly two standard errors."
                   if inside else "more than two standard errors away from zero.")
            )
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_dir = resolve_output_dir(args.results_dir)
    missing = [name for name in REQUIRED if not (results_dir / name).exists()]
    if missing:
        print("error: cannot generate the report; missing required inputs in "
              f"{display_path(results_dir)}: {', '.join(missing)}\n"
              "Run scripts/run_experiments.py with configs/baseline.json and "
              "configs/transaction_costs.json first.", file=sys.stderr)
        return 1

    metadata = json.loads((results_dir / "run_metadata.json").read_text(encoding="utf-8"))
    scenarios = metadata.get("scenarios", {})
    if not {"baseline", "transaction_cost"} <= set(scenarios):
        print("error: run_metadata.json must contain both 'baseline' and 'transaction_cost' sections.",
              file=sys.stderr)
        return 1
    base_cfg, cost_cfg = scenarios["baseline"]["config"], scenarios["transaction_cost"]["config"]
    same_paths = all(base_cfg[k] == cost_cfg[k] for k in COMPARABLE_KEYS)

    summary = pd.concat([pd.read_csv(results_dir / "baseline_summary.csv"),
                         pd.read_csv(results_dir / "transaction_cost_summary.csv")], ignore_index=True)
    pnl = {"baseline": pd.read_csv(results_dir / "baseline_terminal_pnl.csv"),
           "transaction_cost": pd.read_csv(results_dir / "transaction_cost_terminal_pnl.csv")}
    strategies = list(base_cfg["strategies"])

    fig_dir = results_dir / "figures"
    example_path = results_dir / "example_option.json"
    example = json.loads(example_path.read_text(encoding="utf-8")) if example_path.exists() else None
    K, T, r, vol = base_cfg["strike"], base_cfg["maturity_years"], base_cfg["risk_free_rate"], base_cfg["hedging_volatility"]
    spots = np.linspace(0.5 * K, 1.5 * K, 201)
    plot_option_prices(fig_dir / "option_prices.png", K, T, r, vol, spots)
    plot_option_deltas(fig_dir / "option_delta.png", K, T, r, vol, spots)
    plot_rmse_comparison(fig_dir / "hedging_rmse_comparison.png", summary.to_dict("records"))

    paired_md, facts = paired_analysis(pnl, strategies)
    figures_present = {name: (fig_dir / name).exists() for name in (
        "option_prices.png", "option_delta.png", "sample_stock_paths.png", "baseline_hedging_pnl.png",
        "transaction_cost_hedging_pnl.png", "hedging_rmse_comparison.png")}

    md = [
        "# Report: Delta-Hedging a Short European Call (Simulated Data)",
        "",
        f"_Generated by `python scripts/generate_report.py` at {environment_info()['executed_at_utc']} "
        "from the saved files in this folder. All stock prices are **simulated** with geometric "
        "Brownian motion; no market data are used._",
        "",
        "## Overview",
        "",
        "A trader sells one European call on one share at its Black-Scholes price, holds the model "
        "delta in stock, and finances trades through a cash account at the risk-free rate. The "
        "terminal cash after liquidating the stock and paying the option payoff is the hedging P&L "
        "(positive = profit for the option seller; terminal, undiscounted, currency units per option). "
        "Daily and weekly rebalancing are compared on identical simulated paths, without and with a "
        "5 basis-point proportional stock transaction cost.",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |", "|---|---|",
    ]
    for key in ("initial_stock_price", "strike", "maturity_years", "risk_free_rate", "simulation_drift",
                "simulation_volatility", "hedging_volatility", "dividend_yield", "trading_steps", "n_paths",
                "random_seed"):
        md.append(f"| {key} | {base_cfg[key]} |")
    md += [
        f"| strategies (rebalance every N steps) | {base_cfg['strategies']} |",
        f"| transaction_cost_rate (baseline / cost scenario) | {base_cfg['transaction_cost_rate']} / "
        f"{cost_cfg['transaction_cost_rate']} |",
        "",
        "Same simulated paths in both scenarios (identical seed and simulation parameters in the two "
        f"saved configs): **{'yes' if same_paths else 'NO - comparisons across scenarios are not paired'}**.",
        "",
        "## Example option",
        "",
    ]
    if example is None:
        md.append("**Missing:** `example_option.json` was not found. Run `python scripts/price_option.py "
                  "... --output-json results/example_option.json`.")
    else:
        g, s = example.get("greeks_raw") or {}, example.get("greeks_scaled") or {}
        i = example["inputs"]
        md += [
            f"European {i['option_type']}, S={i['spot']}, K={i['strike']}, T={i['maturity_years']}, "
            f"r={i['rate']}, sigma={i['volatility']}:",
            "",
            "| Quantity | Value |", "|---|---|",
            f"| Price | {example['price']:.6f} |",
        ]
        for label, key, src in (("Delta", "delta", g), ("Gamma", "gamma", g),
                                ("Vega per 1.0 vol", "vega_per_unit_vol", g),
                                ("Vega per vol point", "vega_per_vol_point", s),
                                ("Theta per year", "theta_per_year", g),
                                ("Theta per calendar day (/365)", "theta_per_calendar_day", s),
                                ("Rho per 1.0 rate", "rho_per_unit_rate", g),
                                ("Rho per rate point", "rho_per_rate_point", s)):
            if key in src:
                md.append(f"| {label} | {src[key]:.6f} |")
    md += [
        "",
        "## Results",
        "",
        results_table(summary),
        "",
        "### Paired comparisons (same simulated paths)",
        "",
        paired_md,
        "",
        "## Figures",
        "",
    ]
    captions = {
        "option_prices.png": "Black-Scholes call and put prices versus stock price",
        "option_delta.png": "Call and put delta versus stock price",
        "sample_stock_paths.png": "Sample of simulated GBM stock paths",
        "baseline_hedging_pnl.png": "Terminal P&L, daily vs weekly, no transaction costs",
        "transaction_cost_hedging_pnl.png": "Terminal P&L, daily vs weekly, 5 bp transaction costs",
        "hedging_rmse_comparison.png": "RMSE comparison across strategies and cost scenarios",
    }
    for name, present in figures_present.items():
        md.append(f"![{captions[name]}](figures/{name})" if present else f"**Missing figure:** `{name}`")
        md.append("")
    md += ["## Interpretation (derived from the numbers above)", ""] + interpretation(summary, facts)
    md += [
        "",
        "## Limitations",
        "",
        "- Results come from one simulated parameter set and one random seed; other parameters, "
        "costs or seeds can change magnitudes and, potentially, rankings.",
        "- The hedging volatility equals the simulated volatility, so model risk from mis-specified "
        "volatility is not studied here.",
        "- Constant volatility and rates, no dividends, no jumps, continuous-price GBM, no option "
        "bid-ask spread, proportional stock costs only.",
        "- Standard errors reflect Monte Carlo sampling noise only, not model uncertainty.",
        "- P&L is terminal and undiscounted, per option on one share.",
        "",
    ]
    report_path = results_dir / "report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Saved {display_path(report_path)}")
    for name in ("option_prices.png", "option_delta.png", "hedging_rmse_comparison.png"):
        print(f"Saved {display_path(fig_dir / name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
