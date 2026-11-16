# Learning guide

A plain-language walk-through of the project, written to help you explain it
confidently.

## 1. What each module does

| Module | In one sentence |
|---|---|
| `validation.py` | Checks inputs (positive prices, non-negative time and volatility, finite numbers, "call"/"put") so mistakes fail loudly instead of returning NaN. |
| `pricing.py` | Turns the Black–Scholes formula into call and put prices, including the special cases at expiry and with zero volatility, and gives the no-arbitrage price bounds. |
| `greeks.py` | Computes delta, gamma, vega, theta and rho (raw units), converts some of them to everyday units, and refuses to compute them at expiry or zero volatility. |
| `implied_vol.py` | Works backwards from a price to the volatility that produces it, using Brent's root-finding method. |
| `simulation.py` | Generates simulated stock-price paths with geometric Brownian motion and a fixed random seed. |
| `hedging.py` | Plays the role of a trader who sold a call and keeps rebalancing a stock hedge, tracking every cash flow. |
| `metrics.py` | Summarises the end-of-life profit and loss across thousands of paths. |
| `plotting.py` | Draws and saves the charts. |

## 2. How data flows

```
config JSON ──► simulate_gbm_paths ──► paths (10,000 × 253)
                                         │
                       ┌─────────────────┴──────────────────┐
                       ▼                                    ▼
          simulate_delta_hedge (daily)        simulate_delta_hedge (weekly)
                       │      uses pricing.bs_price + greeks.call_delta_array
                       ▼                                    ▼
                terminal P&L, costs                 terminal P&L, costs
                       └──────────► metrics.summarize_pnl ◄─┘
                                         │
                       CSV files, histogram, run_metadata.json
                                         │
                       generate_report.py ──► report.md + summary charts
```

## 3. Pricing versus hedging

* **Pricing** answers: *what is a fair price today if the model is right?* It is
  a single number from a formula.
* **Hedging** answers: *if I sell the option for that price, can I actually
  cover the payoff by trading the stock?* It is a process that runs over the
  life of the option and can go well or badly on each path.

Black–Scholes says that with continuous, costless trading the hedge would cover
the payoff exactly. This project asks what happens when you can only trade
daily or weekly, and when trading costs money.

## 4. Delta hedging and residual risk

Delta is how many shares make your position insensitive to a *small* stock
move. After you sell a call you buy delta shares. When the stock moves, delta
changes (that is gamma), so you must trade again. Between trades your hedge is
stale, and big moves between rebalances leave a residual gain or loss. That
residual is the hedging error we measure. Selling an option means you are
**short gamma**: large moves in either direction tend to hurt you, and in
return you collect time decay (theta).

## 5. Why cash accounting matters

The hedge is *self-financing*: no money is added or removed after the start.
Every share purchase is paid from the cash account; every sale adds to it; the
cash earns (or pays) interest. If you ignore interest or forget to charge the
liquidation cost, the P&L is wrong even if the deltas are right. That is why
the tests check cash balances on small paths worked out by hand.

The final step matters too: at expiry we sell the remaining shares and pay the
option payoff **once**. We do not compute a delta at expiry because it is not
well defined at the strike.

## 6. Why trading more often changes both error and cost

* More frequent rebalancing keeps the hedge closer to the model delta, so the
  hedging error shrinks (in our run, RMSE 0.436 daily vs 0.953 weekly without
  costs).
* But every trade pays a proportional cost, so more trades cost more (0.309
  daily vs 0.175 weekly at 5 bp).
* The best frequency depends on the cost level and on what you care about
  (spread of outcomes versus expected loss). In our run daily still had lower
  RMSE with 5 bp costs, but weekly had a less negative mean P&L.

## 7. What to understand before an interview

* State the Black–Scholes assumptions and which ones break in practice.
* Explain put–call parity and why the price bounds exist.
* Know the units of each Greek and why vega is often quoted per vol point.
* Explain implied volatility and why it can fail to exist (price outside bounds,
  or at expiry).
* Describe the cash account equations for the hedge, including interest and
  costs.
* Explain why daily hedging error is not zero (discrete trading, gamma) and why
  it is not better on every path (randomness).
* Explain the difference between standard deviation and RMSE, and why they
  diverge when costs make the mean negative.
* Be honest about limitations: simulated data, constant volatility, one
  parameter set, one seed, no option bid–ask.

## 8. Self-check questions

1. Why is the price at σ = 0 equal to max(S − K e^{−rT}, 0) rather than max(S − K, 0)?
2. Why does the code raise an error instead of returning a gamma at T = 0?
3. Which no-arbitrage condition makes an implied volatility impossible for a call priced above S?
4. In the weekly strategy with 252 steps, when is the last rebalance and how long is the hedge then held?
5. Why does the cost-scenario P&L fall by slightly more than the nominal transaction cost?
6. If mean P&L is −0.3 and the standard deviation is 0.45, roughly what is the RMSE?
7. Why must both strategies use the same simulated paths?
8. What would you expect to change if the hedging volatility were set to 0.25 while paths are simulated at 0.20?
