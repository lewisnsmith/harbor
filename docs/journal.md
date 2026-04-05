# HANGAR — Project Journal

> How HANGAR (Hierarchical Agentic Risk Based Optimization Routine) grew from a high-school competition project into a semi-autonomous portfolio management system — and the lessons learned along the way.

**Main Goal:** Produce a semi-autonomous Asset and Risk Management Algorithm for Retail-Level / Individual Investors.

---

## Iteration 0.5 — Monte Carlo Simulation Engine

*During the Wharton Global High School Investment Competition*

The very first step: a Monte Carlo simulation engine running on a portfolio of equities with predetermined allocation weights. A naive but exciting first exposure to quantitative investing tools — just seeing how a portfolio *might* behave across random futures.

---

## Iteration 1 — Quantitative Factor Screening

*During the Wharton Global High School Investment Competition*

Replaced gut-feel stock picks with a structured factor-scoring model inspired by AQR. Scored stocks across quality, valuation, and momentum, then ranked them to build a portfolio.

**What it fixed:** Emotional decision-making gave way to disciplined, comparable analysis across companies and sectors.

**Where it fell short:** Factor crowding risk, no forward-looking stress testing, static allocations that would crumble under regime change, and unaddressed correlation risk.

---

## Iteration 2 — Factor Screening + Monte Carlo Simulation

*During the Wharton Global High School Investment Competition*

Combined the factor model with Monte Carlo simulations to stress-test portfolio combinations — not just rank individual stocks.

**What it fixed:** Revealed tail risks, downside clustering, drawdown profiles, and recovery paths. Informed better position sizing.

**Where it fell short:** Simulations were anchored to historical data with no forward-looking projections, no logic for optimal capital allocation, and portfolio construction was still ad hoc.

---

## Iteration 3 — Mean-Variance Optimization

*During the Wharton Global High School Investment Competition*

The beginning of more formal portfolio construction. Discovered Mean-Variance Optimization (MVO) as a mathematically clean way to translate factor scores into weights while minimizing variance.

**What it fixed:** Removed subjective sizing, quantified risk-return trade-offs, and created real allocation logic.

**Where it fell short:** Assumed stable, normally distributed returns; overweighted low-volatility assets regardless of thesis; produced mathematically "optimal" but thematically misaligned portfolios; extremely sensitive to return estimation inputs; generated excessive noise with large stock baskets.

---

## Iteration 4 — Thesis-Aligned Constraints & Clustering

*Bridge to HANGAR*

Introduced behavioral constraints (caps, defensive minimums), multiple risk-level outputs, and early clustering concepts to detect hidden correlations — ensuring the optimizer served the investment thesis rather than overriding it.

**What it fixed:** Prevented overconcentration in "efficient" but irrelevant assets, reduced hidden correlation risk, and improved alignment with the researched thesis logic.

**Where it fell short:** Constraints were manual and rigid; clustering was rudimentary.

---

## Iteration 5 — HANGAR v0

The algorithm got its name. Layered in Black-Litterman expected-return adjustment, hierarchical clustering for behavioral diversification, macro-event stress testing, and an early reinforcement-learning module for adaptive weighting. Each layer existed to fix a specific failure observed in earlier iterations.

**What it fixed:** Factors selected quality; clustering prevented hidden risk; behavioral rules blocked human bias; Black-Litterman corrected statistical distortions; Monte Carlo tested reality instead of averages; RL enabled adaptation without overreaction.

**Where it fell short:** Still not autonomous. RL wasn't fully deployed. No performance metrics, dashboard, backtesting, or statistical evaluation.

---

## Iteration 6 — HANGAR v1 / Phase 1

*First paper-trading system — slightly autonomous*

Transitioned from Colab notebooks to a deployable system using Alpaca's API for live market data and order execution. Added portfolio monitoring, risk-parameter enforcement, daily rebalancing, and automated order generation.

**What it fixed:** Real-time decision-making, execution-layer validation, and monitoring infrastructure.

**Where it fell short:** Market orders only (no price improvement), arbitrary daily rebalancing, no transaction-cost modeling, hardcoded risk parameters, and only basic total-return attribution.

---

## Iteration 7 — HANGAR v2 / Phase 2

*Backtesting engine, performance analytics, transaction-cost model*

Built a historical backtesting framework, Sharpe/drawdown metrics, transaction-cost estimation, factor-contribution analysis, rolling performance windows, and SPY benchmark comparison.

**What it fixed:** Enabled systematic strategy evaluation across market regimes. Discovered that daily rebalancing was destroying alpha through transaction costs and that short-window factor momentum had minimal predictive power.

**Where it fell short:** No survivorship-bias correction, historical volatility underestimated tail events, no regime detection, static factor weights, and unstable clustering under changing correlations.

---

## Iteration 8 — HANGAR v2.5

*Regime detection, dynamic factor weighting, improved clustering*

Added a Hidden Markov Model for regime classification, dynamic factor weights, Hierarchical Risk Parity for clustering, volatility targeting, and regime-conditional rebalancing thresholds.

**What it fixed:** The algorithm now responded to changing market conditions; factor importance adjusted on rolling performance; portfolio construction accounted for time-varying correlations; risk exposure scaled down during high-uncertainty periods.

**Where it fell short:** HMM lagged actual regime changes; dynamic weighting risked overfitting; regime states had no economic rationale (purely statistical); live paper trading revealed gaps backtests missed; computational costs grew significantly.

---

## HANGAR v3 — Simplifying for Clarity

After eight iterations of relentless layering — Monte Carlo, MVO, Black-Litterman, HMM regime detection, Hierarchical Risk Parity, reinforcement learning — it became clear that the system had grown too complex for its own good.

The honest truth: I got caught up. Every new mathematical concept I encountered was fascinating. Black-Litterman's elegant Bayesian framework? Had to have it. Hidden Markov Models for regime detection? Too cool to leave out. Reinforcement learning for adaptive weights? The idea alone was irresistible. I kept discovering powerful techniques and convincing myself that each one was essential, that more sophistication automatically meant better results.

It didn't.

What actually happened was that each new component introduced its own failure modes, made debugging harder, increased computational overhead, and — most critically — generated noise that obscured whether the core strategy even worked. I was spending more time tuning the interactions between layers than evaluating the investment logic itself. The system had become theoretically impressive but practically fragile.

The decision for v3 was to step back and ask a harder question: *which of these components actually improve out-of-sample performance, and which just make the backtest look better?*

The answer was humbling. Much of the mathematical machinery I'd added was either overfitting to recent data, introducing latency that negated its value, or solving problems that could be addressed more simply. Regime detection via HMM, for instance, consistently identified regime changes *after* they'd already occurred — by the time the model adjusted, the damage was done. Dynamic factor weighting showed beautiful in-sample improvement but suspiciously inconsistent live results.

So v3 went simpler: fewer mathematical layers, tighter focus on the components with proven out-of-sample value, and a priority shift from maximizing theoretical performance to building a system that is **robust, interpretable, and trustworthy** with real capital. Production reliability, explainability, and graceful failure modes matter more than squeezing out another 50 basis points — especially for a system targeting retail-level investors who need to understand and trust what the algorithm is doing with their money.

The lesson: sophistication is seductive, but simplicity you can trust will always beat complexity you can't fully verify.

---

## Reflections

This project spans roughly 18 months, from competition-driven toy models to something approaching a production system. The progression was never linear — I went down dead ends, overfit to historical data multiple times, and underestimated implementation complexity consistently.

**Key lessons:**

1. **Mathematical elegance ≠ practical performance.** MVO produces beautiful Pareto frontiers but requires return forecasts I couldn't reliably generate.
2. **Backtesting creates false confidence.** Every iteration looked better in backtests, but paper trading revealed problems historical data couldn't capture.
3. **Complexity has costs.** Each added component solved a specific problem but increased overhead, introduced new failure modes, and made debugging harder.
4. **Transaction costs matter more than expected.** At realistic position sizes and rebalancing frequencies, costs often exceeded the alpha I was chasing.
5. **The research-to-production gap is wide.** Error handling, API rate limits, data validation, logging, monitoring — none improved the model, but all were necessary.

**What I'd do differently:**

- Start with simpler models; add complexity only when provably helpful out-of-sample.
- Build backtesting and evaluation infrastructure *before* strategy logic.
- Focus more on risk management, less on return optimization.
- Treat transaction costs as first-order constraints from day one.
- Document decisions and results systematically from the beginning.

**Where I'm headed:**

A system I can trust with real capital — one that prioritizes transparency, validation, and graceful failure. But the more interesting question has shifted: as autonomous trading agents (LLM-powered, RL-based, tool-using) become market participants, do they reshape the dynamics that every investor — retail and institutional — depends on? HANGAR's multi-agent simulation now lets me test this directly: what happens to volatility, crowding, and regime structure when 30 momentum agents trade the same market? The early results are striking — agent-influenced markets show 2.6x higher volatility and near-perfect crowding — and the research agenda has evolved from "does my algorithm work?" to "what happens to markets when everyone's algorithm works?"

This project taught me to respect the gap between theory and practice, to be skeptical of my own metrics, and to value simplicity and reliability over sophistication. It also taught me that the most interesting questions aren't about optimizing returns — they're about understanding what autonomous systems do to the markets we all share.
