"""Price one European option and show its Greeks.

Run from the project root, for example:

    python scripts/price_option.py --spot 100 --strike 100 --maturity 1 \
        --rate 0.05 --volatility 0.20 --option-type call

Add ``--output-json results/example_option.json`` to save the output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _common import display_path, environment_info, resolve_output_dir

from bs_hedging.greeks import BoundaryGreekError, all_greeks, scaled_greeks
from bs_hedging.implied_vol import ImpliedVolatilityError, implied_volatility
from bs_hedging.pricing import bs_price


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Black-Scholes price and Greeks for one European option on one share "
        "(no dividends). Raw Greeks use per-1.0 units; scaled Greeks are also shown.",
    )
    parser.add_argument("--spot", type=float, required=True, help="Current stock price S (> 0).")
    parser.add_argument("--strike", type=float, required=True, help="Strike price K (> 0).")
    parser.add_argument("--maturity", type=float, required=True, help="Time to maturity T in years (>= 0).")
    parser.add_argument("--rate", type=float, required=True,
                        help="Continuously compounded risk-free rate, decimal (0.05 = 5%%). May be negative.")
    parser.add_argument("--volatility", type=float, required=True,
                        help="Annualised volatility, decimal (0.20 = 20%%), >= 0.")
    parser.add_argument("--option-type", choices=["call", "put"], default="call", help="Option type.")
    parser.add_argument("--output-json", default=None,
                        help="Optional path of a JSON file to write (e.g. results/example_option.json).")
    return parser


def price_report(spot, strike, maturity, rate, volatility, option_type) -> dict:
    """Compute price, Greeks (raw and scaled) and an implied-volatility round trip."""
    price = bs_price(spot, strike, maturity, rate, volatility, option_type)
    report = {
        "inputs": {
            "spot": spot, "strike": strike, "maturity_years": maturity, "rate": rate,
            "volatility": volatility, "option_type": option_type, "dividend_yield": 0.0,
            "contract_size_shares": 1,
        },
        "price": price,
    }
    try:
        raw = all_greeks(spot, strike, maturity, rate, volatility, option_type)
        report["greeks_raw"] = raw
        report["greeks_scaled"] = scaled_greeks(raw)
    except BoundaryGreekError as exc:
        report["greeks_raw"] = None
        report["greeks_note"] = str(exc)
    try:
        report["implied_volatility_round_trip"] = implied_volatility(
            price, spot, strike, maturity, rate, option_type
        )
    except ImpliedVolatilityError as exc:
        report["implied_volatility_round_trip"] = None
        report["implied_volatility_note"] = str(exc)
    report["units"] = {
        "price": "currency units per option on one share",
        "delta": "per 1.0 change in stock price",
        "gamma": "per 1.0 change in stock price, per 1.0 change in stock price",
        "vega_per_unit_vol": "per 1.0 absolute change in volatility (e.g. 0.20 -> 1.20)",
        "vega_per_vol_point": "per 1 percentage-point change in volatility = raw / 100",
        "theta_per_year": "calendar-time decay per year, other inputs fixed",
        "theta_per_calendar_day": "raw theta / 365",
        "rho_per_unit_rate": "per 1.0 absolute change in the rate",
        "rho_per_rate_point": "per 1 percentage-point change in the rate = raw / 100",
    }
    return report


def format_report(report: dict) -> str:
    i = report["inputs"]
    lines = [
        f"European {i['option_type']}  S={i['spot']:g}  K={i['strike']:g}  T={i['maturity_years']:g} yr  "
        f"r={i['rate']:g}  sigma={i['volatility']:g}  (q=0, one share)",
        f"Price                         : {report['price']:.6f}",
    ]
    raw = report.get("greeks_raw")
    if raw:
        scaled = report["greeks_scaled"]
        lines += [
            "Raw Greeks",
            f"  delta                       : {raw['delta']:.6f}",
            f"  gamma                       : {raw['gamma']:.6f}",
            f"  vega  (per 1.0 vol)         : {raw['vega_per_unit_vol']:.6f}",
            f"  theta (per year)            : {raw['theta_per_year']:.6f}",
            f"  rho   (per 1.0 rate)        : {raw['rho_per_unit_rate']:.6f}",
            "Scaled Greeks",
            f"  vega  (per 1 vol point)     : {scaled['vega_per_vol_point']:.6f}",
            f"  theta (per calendar day/365): {scaled['theta_per_calendar_day']:.6f}",
            f"  rho   (per 1 rate point)    : {scaled['rho_per_rate_point']:.6f}",
        ]
    else:
        lines.append(f"Greeks: not available - {report.get('greeks_note')}")
    iv = report.get("implied_volatility_round_trip")
    lines.append(
        f"Implied vol from this price   : {iv:.10f}" if iv is not None
        else f"Implied vol: not available - {report.get('implied_volatility_note')}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = price_report(args.spot, args.strike, args.maturity, args.rate, args.volatility, args.option_type)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_report(report))
    if args.output_json:
        target = Path(args.output_json)
        out_path = resolve_output_dir(target.parent) / target.name
        env = environment_info()
        report["generated"] = {
            "command": "python scripts/price_option.py " + " ".join(
                (argv if argv is not None else sys.argv[1:])
            ).replace("\\", "/"),
            "executed_at_utc": env["executed_at_utc"],
            "python_version": env["python_version"],
        }
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved {display_path(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
