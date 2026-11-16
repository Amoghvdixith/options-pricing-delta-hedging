# Methodology

This document defines every formula and convention used in the code. Code
references are to modules in `src/bs_hedging/`.

## 1. Black–Scholes model and assumptions

Variables:

| Symbol | Meaning | Units |
|---|---|---|
| S | current stock price | currency units |
| K | strike price | currency units |
| T | time to maturity | years |
| r | continuously compounded risk-free rate (may be negative) | per year, decimal |
| σ | volatility of log returns | per √year, decimal |
| q | dividend yield, fixed at 0 in this project | — |
| N(·), n(·) | standard normal CDF and density | — |

Assumptions: frictionless market (except for the explicit transaction-cost
experiment), continuous trading in the model, constant r and σ, no dividends,
log-normal stock price, European exercise. One option is written on **one**
share; no contract multiplier is applied.

Prices (`pricing.py`):

```
d1 = [ln(S/K) + (r + σ²/2) T] / (σ √T),   d2 = d1 − σ √T
Call = S N(d1) − K e^{−rT} N(d2)
Put  = K e^{−rT} N(−d2) − S N(−d1)
```

Boundary cases:

* **T = 0** → intrinsic value: max(S − K, 0) or max(K − S, 0).
* **σ = 0, T > 0** → under the pricing measure the stock grows deterministically
  at r, so the price is the discounted deterministic payoff:
  call = max(S − K e^{−rT}, 0), put = max(K e^{−rT} − S, 0).

No-arbitrage bounds (`price_bounds`), no dividends:
max(S − K e^{−rT}, 0) ≤ C ≤ S and max(K e^{−rT} − S, 0) ≤ P ≤ K e^{−rT}.

Input validation (`validation.py`): all inputs finite, S > 0, K > 0, T ≥ 0,
σ ≥ 0, option type "call" or "put"; negative r allowed.

## 2. Greeks and units (`greeks.py`)

| Greek | Call | Put | Raw unit |
|---|---|---|---|
| Delta | N(d1) | N(d1) − 1 | per 1.0 change in S |
| Gamma | n(d1) / (S σ √T) | same | per 1.0 change in S, per 1.0 change in S |
| Vega | S n(d1) √T | same | per **1.0** absolute change in σ |
| Theta | −S n(d1) σ / (2√T) − r K e^{−rT} N(d2) | −S n(d1) σ / (2√T) + r K e^{−rT} N(−d2) | per **year** of calendar time |
| Rho | T K e^{−rT} N(d2) | −T K e^{−rT} N(−d2) | per **1.0** absolute change in r |

Theta is dV/dt = −dV/dT with S, σ and r fixed.

Human-readable scaling (`scaled_greeks`, always under separate, labelled keys):

* vega per volatility percentage point = raw vega / 100
* rho per rate percentage point = raw rho / 100
* theta per calendar day = raw theta / **365**

**Boundary policy.** Greeks are computed only for T > 0 and σ > 0. At T = 0 or
σ = 0 the price is a kinked function of S, delta is discontinuous at the kink
and gamma is not a finite number there, so all Greek functions raise
`BoundaryGreekError` (a `ValueError`). No small-epsilon substitution is made.
The hedging simulation never requests a Greek at expiry.

## 3. Implied volatility (`implied_vol.py`)

Solve `bs_price(σ) − market_price = 0` with `scipy.optimize.brentq`, using the
project's own pricing function.

1. **T = 0** is rejected: the price equals intrinsic value for every σ, so σ is
   not identified.
2. The price is checked against the no-arbitrage bounds:
   * below the lower bound → error (no σ reproduces it);
   * at or above the upper bound → error (only reached as σ → ∞, no finite σ);
   * equal to the lower bound (within `bound_tolerance`, default 1e‑12) →
     returns 0.0, because σ = 0 attains the lower bound and the price is
     strictly increasing in σ for σ > 0 (vega > 0).
3. **Bracketing.** The root is searched on [0, σ_hi]. σ_hi starts at
   `initial_upper` (1.0) and is doubled until `bs_price(σ_hi) ≥ price`. If σ_hi
   would exceed `max_upper` (default 10.0) an `ImpliedVolatilityError` is raised.
   `max_upper` is a **computational search limit**, not a mathematical upper
   bound on volatility; callers can raise it.
4. **Tolerance.** `tolerance` is passed to brentq as `xtol` (default 1e‑10 in
   volatility units).

## 4. GBM simulation (`simulation.py`)

Exact log-normal transitions on an equal grid dt = T/N:

```
S(t + dt) = S(t) · exp[(μ − σ_sim²/2) dt + σ_sim √dt · Z],  Z ~ N(0,1) i.i.d.
```

`numpy.random.default_rng(seed)` generates Z. The output array has shape
(n_paths, N + 1) and its first column equals S(0).

Three volatility/drift parameters are kept separate:

* `simulation_drift` μ — drift of the simulated (real-world) paths;
* `simulation_volatility` σ_sim — volatility used to generate paths;
* `hedging_volatility` σ_hedge — volatility used for the premium and deltas.

The baseline sets μ = r = 0.05. This is a **modelling choice** for the
experiment, not a claim that real stocks are expected to earn the risk-free
rate. In the Black–Scholes model the hedging error does not depend on μ in the
continuous-time limit, but finite-sample discrete-hedging results can.

## 5. Self-financing delta hedge (`hedging.py`)

Position: short one European call, long Δ shares, cash account B.
Grid t_k = k·dt, k = 0..N. Proportional cost rate c on traded stock notional.

**t = 0**

```
premium = BS call price(S0, K, T, r, σ_hedge)
Δ0      = N(d1) at (S0, T, σ_hedge)
B0      = premium − Δ0·S0 − c·|Δ0|·S0
```

**k = 1, …, N−1**

```
B ← B · e^{r·dt}                              (interest on the elapsed step)
if k is a rebalance step:
    Δnew = N(d1) at (S_k, T − t_k, σ_hedge)
    B   ← B − (Δnew − Δold)·S_k − c·|Δnew − Δold|·S_k
```

**k = N (expiry)** — no Greek is evaluated and no final rebalance is made:

```
B ← B · e^{r·dt}
B ← B + Δheld·S_N − c·|Δheld|·S_N             (liquidate stock)
B ← B − max(S_N − K, 0)                       (pay the option payoff once)
terminal P&L = B
```

### Rebalancing schedules

Rebalance steps are k = 0, m, 2m, … with k < N.

* **Daily** (m = 1): k = 0, 1, …, 251 → 252 hedge-setting dates including t = 0.
* **Weekly** (m = 5): k = 0, 5, …, 250 → 51 dates. Because 252 is not divisible
  by 5, the final hedge (set at step 250) is held for **two** steps until
  expiry at step 252, where the stock is liquidated without a rebalance.

Both strategies start at t = 0, settle at the same expiry and use exactly the
same path array.

### Transaction costs

Cost per trade = c × |shares traded| × price. Applied to the initial hedge,
every intermediate rebalance and the final liquidation. There is no option
bid–ask spread and no option execution fee. The reported per-path cost is the
cumulative **nominal** (undiscounted, not interest-accrued) sum. Because costs
are paid from the cash account, their effect on terminal P&L is slightly larger
than the nominal sum when r > 0 (lost interest).

### P&L definition

* Positive P&L = profit for the hedged option seller; negative = loss.
* Terminal, **undiscounted** currency amounts per option on one share.

## 6. Metrics (`metrics.py`)

For each strategy and scenario, over n paths with terminal P&L x_i:

| Metric | Definition |
|---|---|
| n_paths | n |
| mean_pnl | (1/n) Σ x_i |
| std_pnl | sample standard deviation, ddof = 1 |
| rmse_pnl | √[(1/n) Σ x_i²] — error relative to zero P&L (a perfect hedge) |
| p05 / median / p95 | `numpy.percentile(..., method="linear")` (Hyndman & Fan type 7) |
| min / max | sample extremes |
| mean_transaction_cost | mean of cumulative nominal cost per path |

RMSE² = (population variance) + mean², so RMSE and the standard deviation
differ whenever the mean is not zero; they are never relabelled as each other.
The report additionally shows the standard error of the mean, std/√n.
