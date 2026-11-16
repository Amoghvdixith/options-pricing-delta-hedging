"""Run one delta-hedging experiment (daily vs weekly rebalancing) from a JSON config.

Run from the project root:

    python scripts/run_experiments.py --config configs/baseline.json
    python scripts/run_experiments.py --config configs/transaction_costs.json

Each config writes its own, differently named output files, so the two runs
do not overwrite each other. Both runs add their own section to
``run_metadata.json`` and keep the other section.
All stock prices are SIMULATED (geometric Brownian motion), not market data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from _common import display_path, environment_info, resolve_input_path, resolve_output_dir

from bs_hedging.hedging import compare_strategies, rebalance_steps
from bs_hedging.metrics import summarize_pnl
from bs_hedging.plotting import plot_pnl_histograms, plot_sample_paths
from bs_hedging.simulation import simulate_gbm_paths, time_grid

REQUIRED_KEYS = (
    "scenario_name", "initial_stock_price", "strike", "maturity_years", "risk_free_rate",
    "simulation_drift", "simulation_volatility", "hedging_volatility", "dividend_yield",
    "trading_steps", "n_paths", "random_seed", "transaction_cost_rate", "strategies", "outputs",
)

CONVENTIONS = {
    "pnl_sign": "positive = profit for the hedged option seller (short one call, one share)",
    "pnl_units": "terminal, undiscounted currency units per option on one share",
    "std": "sample standard deviation, ddof = 1",
    "rmse": "sqrt(mean(pnl^2)), i.e. relative to zero terminal P&L",
    "percentiles": "numpy.percentile method='linear' (Hyndman-Fan type 7)",
    "transaction_costs": "cumulative nominal (undiscounted) stock-trading cost per path: "
                         "initial hedge + rebalances + final liquidation; no option bid-ask",
    "rebalancing": "rebalance at steps k = 0, m, 2m, ... with k < N; no rebalance and no "
                   "Greek evaluation at expiry; stock liquidated and payoff paid at step N",
    "premium": "Black-Scholes call price at t=0 using the hedging volatility",
    "data": "SIMULATED geometric Brownian motion paths, exact log-normal transitions",
}


def load_config(path) -> dict:
    """Read and check a JSON experiment configuration."""
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"error: config file not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: config file is not valid JSON ({exc})")
    missing = [k for k in REQUIRED_KEYS if k not in config]
    if missing:
        raise SystemExit(f"error: config is missing keys: {', '.join(missing)}")
    if config["dividend_yield"] != 0:
        raise SystemExit("error: this project supports dividend_yield = 0 only")
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate self-financing delta hedging of a short European call on SIMULATED "
        "GBM paths and compare rebalancing frequencies on identical paths.",
    )
    parser.add_argument("--config", required=True,
                        help="Path to a JSON config, e.g. configs/baseline.json (relative to the project root).")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for outputs (default: <project root>/results). Figures go to <output-dir>/figures.")
    return parser


def update_metadata(out_dir, scenario: str, section: dict) -> None:
    """Merge this scenario's metadata into run_metadata.json, keeping other scenarios."""
    meta_path = out_dir / "run_metadata.json"
    metadata = {}
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
    metadata.setdefault("scenarios", {})[scenario] = section
    metadata["numerical_conventions"] = CONVENTIONS
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = resolve_input_path(args.config)
    config = load_config(config_path)
    out_dir = resolve_output_dir(args.output_dir)
    fig_dir = out_dir / "figures"
    outputs = config["outputs"]
    scenario = config["scenario_name"]
    n_steps = int(config["trading_steps"])

    print(f"[{scenario}] simulating {config['n_paths']} SIMULATED GBM paths x {n_steps} steps "
          f"(seed {config['random_seed']}) ...")
    start = time.perf_counter()
    paths = simulate_gbm_paths(
        config["initial_stock_price"], config["simulation_drift"], config["simulation_volatility"],
        config["maturity_years"], n_steps, int(config["n_paths"]), int(config["random_seed"]),
    )
    results = compare_strategies(
        paths, config["strategies"], config["strike"], config["maturity_years"],
        config["risk_free_rate"], config["hedging_volatility"], config["transaction_cost_rate"],
    )
    elapsed = time.perf_counter() - start

    rows = []
    pnl_table = {"path_id": np.arange(paths.shape[0])}
    for name, every in config["strategies"].items():
        result = results[name]
        stats = summarize_pnl(result.terminal_pnl, result.transaction_costs)
        rows.append({
            "scenario": scenario,
            "strategy": name,
            "rebalance_every_steps": every,
            "n_rebalance_dates": len(rebalance_steps(n_steps, every)),
            "transaction_cost_rate": config["transaction_cost_rate"],
            "premium": result.premium,
            **stats,
        })
        pnl_table[f"{name}_terminal_pnl"] = result.terminal_pnl
        pnl_table[f"{name}_transaction_cost"] = result.transaction_costs

    summary = pd.DataFrame(rows)
    summary_path = out_dir / outputs["summary_csv"]
    summary.to_csv(summary_path, index=False, float_format="%.10g")
    pnl_path = out_dir / outputs["terminal_pnl_csv"]
    pd.DataFrame(pnl_table).to_csv(pnl_path, index=False, float_format="%.10g")

    cost_label = (f"{config['transaction_cost_rate'] * 1e4:g} bp transaction costs"
                  if config["transaction_cost_rate"] > 0 else "no transaction costs")
    figure_path = plot_pnl_histograms(
        fig_dir / outputs["pnl_figure"],
        {name: results[name].terminal_pnl for name in config["strategies"]},
        f"SIMULATED terminal delta-hedging P&L, {cost_label}\n"
        f"({paths.shape[0]:,} GBM paths, identical for both strategies)",
    )
    written = [summary_path, pnl_path, figure_path]

    if outputs.get("save_sample_paths"):
        n_sample = int(config.get("n_sample_paths_to_save", 10))
        times = time_grid(config["maturity_years"], n_steps)
        sample = pd.DataFrame({"step": np.arange(n_steps + 1), "time_years": times})
        for i in range(n_sample):
            sample[f"path_{i:02d}"] = paths[i]
        sample_path = out_dir / "sample_paths.csv"
        sample.to_csv(sample_path, index=False, float_format="%.10g")
        written.append(sample_path)
        written.append(plot_sample_paths(fig_dir / "sample_stock_paths.png", times, paths[:n_sample],
                                         config["strike"]))

    config_arg = display_path(config_path)
    command = f"python scripts/run_experiments.py --config {config_arg}"
    if args.output_dir is not None:
        command += f" --output-dir {display_path(out_dir)}"
    update_metadata(out_dir, scenario, {
        "config_file": config_arg,
        "config": config,
        "random_seed": config["random_seed"],
        "command": command,
        "runtime_seconds": round(elapsed, 3),
        "outputs": [display_path(p) for p in written],
        **environment_info(),
    })

    print(summary[["strategy", "mean_pnl", "std_pnl", "rmse_pnl", "mean_transaction_cost"]]
          .to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    for p in written + [out_dir / "run_metadata.json"]:
        print(f"Saved {display_path(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
