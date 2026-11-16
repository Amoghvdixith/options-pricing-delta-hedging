# Validation

## Actual test outcome

Command (from the project root, inside the project virtual environment):

```
python -m pytest -q --junitxml=results/test_results.xml
```

Final recorded result (2026-11-15, Python 3.14.4, Windows):

| Outcome | Count |
|---|---|
| Passed | 992 |
| Failed | 0 |
| Errors | 0 |
| Skipped | 16 |
| Total collected | 1008 |

Machine-readable results: [`results/test_results.xml`](../results/test_results.xml).
After the run, `python scripts/sanitize_test_xml.py` replaced the `hostname`
attribute with `"removed"` and rewrote the absolute file paths that pytest
records for skipped tests as project-relative paths, so that no machine or user
name is published. Test names, counts, outcomes and timings were not changed.

Per file:

| File | Passed | Skipped |
|---|---|---|
| `tests/test_pricing.py` | 504 | 0 |
| `tests/test_greeks.py` | 28 | 0 |
| `tests/test_implied_vol.py` | 427 | 16 |
| `tests/test_simulation.py` | 13 | 0 |
| `tests/test_hedging.py` | 16 | 0 |
| `tests/test_metrics.py` | 4 | 0 |

The large counts come from parametrised grids (e.g. 243 parameter combinations
× parity and bounds checks in pricing).

**The 16 skips** are implied-volatility round-trip cases on the test grid where
the option is so deep in or out of the money that its price is within 1e‑9 of
the no-arbitrage lower bound. At double precision such a price carries almost
no information about volatility, so recovering the original σ is not a
meaningful requirement. They are skipped deliberately and reported, not hidden.

## Test categories and independent checks

### Pricing (`test_pricing.py`)
* **Independent reference values**: Hull's textbook worked example
  (S=42, K=40, r=10%, σ=20%, T=0.5 → call 4.76, put 0.81 to 2 d.p.) and the
  widely quoted S=K=100, r=5%, σ=20%, T=1 call value 10.4506.
* Put–call parity on a 243-point grid including negative rates (abs tol 1e‑9).
* No-arbitrage bounds for calls and puts on the same grid.
* Expiry payoff, zero-volatility discounted payoff, continuity as σ → 0.
* Invalid inputs (zero/negative spot or strike, negative T or σ, NaN, inf,
  bad option type, non-numeric) raise `ValueError`.

### Greeks (`test_greeks.py`)
* Central **finite differences of the price function** (not of the Greek
  formulas) at four interior points for calls and puts. Tolerances: delta
  rel 1e‑5, gamma rel 1e‑4, vega and rho rel 1e‑6, theta rel 1e‑5.
* **Independent reference**: Hull's Greek-letters example (S=49, K=50, r=5%,
  σ=20%, T=0.3846): delta 0.522, gamma 0.066, vega 12.1, theta −4.31/yr,
  rho 8.91, matched to the published rounding.
* Call/put relations derived from parity: Δc − Δp = 1, Θc − Θp = −rKe^{−rT},
  ρc − ρp = TKe^{−rT}.
* Vectorised delta used by the hedging engine equals the scalar delta.
* Boundary policy (separate tests): every Greek raises `BoundaryGreekError` at
  T = 0 and at σ = 0.
* Unit scaling: vega/100, rho/100, theta/365; and vega per vol point ≈ the
  actual price change for σ 0.20 → 0.21 (within 1%).

### Implied volatility (`test_implied_vol.py`)
* Round trip σ → price → σ̂ → price for calls and puts across 432 cases
  (σ from 0.08 to 1.5, negative and positive r); repriced error ≤ 1e‑7 and
  relative σ error ≤ 1e‑5.
* Configurable tolerance; prices at/above the upper bound and below the lower
  bound rejected; NaN/inf/negative prices rejected; price at the lower bound
  returns 0; T = 0 rejected.
* Bracketing failure: σ = 3 cannot be bracketed with `max_upper=2` (error text
  mentions the search limit) but is recovered with `max_upper=8`.

### Simulation (`test_simulation.py`)
* Same seed → identical paths; different seed → different paths.
* Shape (n_paths, N+1), first column = S0, all prices > 0 (even σ = 0.9).
* σ = 0 gives S0·e^{μt} exactly.
* Distribution checks with fixed seeds and tolerances of 5 standard errors:
  mean of log-returns, variance (3% relative), E[S_T] = S0e^{μT}, and near-zero
  correlation of successive increments.

### Hedging (`test_hedging.py`)
* **Hand-calculated cash balances** on the path 100 → 110 → 105 with injected
  hedge ratios 0.5 and 0.8 (independent of Black–Scholes):
  no interest/no costs (P&L 6), with 10% interest (−6.3267506), with 1% costs
  (P&L 4.33, costs 1.67).
* Initial and liquidation costs counted exactly once (rebalance only at t=0).
* Zero hedge → P&L = premium·e^{rT} − payoff (payoff paid once).
* No Greek requested at expiry (recorded steps 0..N−1, all with τ > 0).
* Schedules: 10/5 → [0,5], 12/5 → [0,5,10], 252/5 → 51 dates ending at 250.
* Strategies use the identical, unmodified path array.
* Deterministic deep-in-the-money path is replicated exactly (P&L 0 to 1e‑9).
* Delta hedging reduces P&L dispersion by more than 5× versus no hedge.

### Metrics (`test_metrics.py`)
* Hand values for [−2, −1, 0, 1, 2]: std √2.5 (ddof 1), RMSE √2, 5th/95th
  percentiles ±1.8 (linear method).
* For [1, 2, 3, 4]: RMSE (2.7386) ≠ std (1.2910), and RMSE² = variance + mean².

## End-to-end checks performed

* `scripts/price_option.py` output matches the reference value 10.4506 and
  recovers implied volatility 0.2000000000.
* Both experiment configs were run at full size (10,000 paths × 252 steps);
  each wrote its own files without overwriting the other's.
* The report was generated only from saved CSV/JSON files.

## What the tests do not establish

* They do not show that Black–Scholes describes real markets.
* Distributional checks use fixed seeds; they confirm the generator is
  implemented correctly for these seeds, not every possible seed.
* Hedging tests confirm the accounting, not that any rebalancing frequency is
  optimal.
* No test covers dividends, American exercise, stochastic volatility or option
  bid–ask spreads, because these are out of scope.
* The plotting and CLI scripts are exercised by running them, not by unit tests.
