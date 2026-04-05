# Q1: Do Independent Autonomous Agents Exhibit Emergent Coordination?

**Author:** Lewis Smith
**Date:** April 2026
**Status:** Complete (200-seed run, April 2026)

---

## Question

Do independent autonomous trading agents — given no communication channel and no shared strategy — converge on similar portfolio allocations when exposed to the same market? If so, is this convergence stronger than what random allocation produces?

This is the first empirical test in the Artificial Behavioral Finance (ABF) research program. Traditional behavioral finance studies human biases; ABF asks whether *autonomous agents* create their own characteristic market dynamics.

## Method

### Agent Design

We introduce the **AutonomousAgent**, parameterized by a 2D personality:

- **Risk appetite** (0.1 to 1.0): Controls annualized volatility target (1% to 10%). Higher risk appetite allows larger position dispersion.
- **Reactivity** (0.1 to 1.0): Controls signal blend and rebalance speed. High reactivity favors momentum signals and fast adjustment; low reactivity favors mean-reversion signals and gradual rebalancing.

Each agent computes target weights through:
1. **Momentum signal**: Cumulative return over a lookback window
2. **Mean-reversion signal**: Z-score of reconstructed price vs rolling mean
3. **Signal blend**: `raw = reactivity * znorm(momentum) + (1 - reactivity) * znorm(mean_rev)`
4. **Volatility scaling** (non-directional): Position sizes scaled by `target_vol / realized_vol`
5. **Long-only normalization**: Shift-and-normalize to sum-to-one
6. **Rebalance damping**: `actual = prev_weights + reactivity * (target - prev_weights)`

The lookback window is derived from reactivity: `round(5 + 55 * (1 - reactivity))`, ranging from 5 steps (highly reactive) to 60 steps (slow-moving).

### Experimental Protocol

- **Treatment**: 25 autonomous agents with unique personalities sampled via Latin Hypercube Sampling (LHS, seed=0)
- **Control**: 25 random agents producing uniform random weights each step
- **Market**: Synthetic equity venue with 10 assets, 500 steps, multivariate normal returns with agent-driven price impact
- **Seeds**: 200 independent market seeds (0-199). Personalities are fixed; only market randomness varies.
- **Metric**: Pairwise Weight Similarity (PWS) — mean cosine similarity across all agent pairs at each time step

### Hypothesis

> **H₀**: mean(PWS_treatment[250:500]) = mean(PWS_control[250:500])
> **H₁**: mean(PWS_treatment[250:500]) > mean(PWS_control[250:500])

Tested via one-sided two-sample t-test across 200 seeds, with Cohen's d effect size and bootstrap 95% CI.

## Results

### Primary Test

| Metric | Treatment | Control | t | p (one-sided) | Cohen's d |
|--------|-----------|---------|---|---------------|-----------|
| PWS (final half) | 0.7830 ± 0.0084 | 0.7590 ± 0.0011 | 39.95 | 1.19 × 10⁻¹⁴¹ | 4.00 |

Bootstrap 95% CI of difference: [0.0228, 0.0251]. **H₁ confirmed.**

### Convergence Slope

| Metric | Treatment | Control | t | p (one-sided) |
|--------|-----------|---------|---|---------------|
| PWS slope (steps 100-500) | −4.4 × 10⁻⁶ | 6.4 × 10⁻⁷ | −1.27 | 0.898 |

Slopes are near zero for both conditions — coordination emerges rapidly (within early steps) and stabilizes, rather than accumulating gradually over the simulation.

### Within-Quadrant Similarity

| Quadrant | Description | Mean PWS |
|----------|-------------|----------|
| Q1 | High risk, high reactivity | 0.8115 ± 0.0068 |
| Q2 | Low risk, high reactivity | 0.8771 ± 0.0052 |
| Q3 | Low risk, low reactivity | 0.9491 ± 0.0045 |
| Q4 | High risk, low reactivity | 0.9091 ± 0.0073 |

### Figures

- `results/q1_experiment/figures/pws_timeseries.png` — PWS over time (treatment vs control)
- `results/q1_experiment/figures/distribution.png` — Distribution of coordination metric
- `results/q1_experiment/figures/personality_scatter.png` — Agent personality space

## Discussion

### Interpretation

The experiment confirms **H₁ with overwhelming statistical force** (t = 39.95, p = 1.19 × 10⁻¹⁴¹, d = 4.00). Independent autonomous agents exposed to the same market converge on substantially more similar portfolio allocations than random agents — a mean PWS difference of 0.024 on a [0,1] scale, with essentially zero overlap between conditions across 200 seeds.

The within-quadrant breakdown reveals the mechanism: **low-reactivity agents coordinate most strongly** (Q3: 0.9491, Q4: 0.9091). Slow-moving agents with long lookback windows see largely the same signal — they are all computing mean-reversion z-scores over 50+ steps of shared market history. High-reactivity agents (Q1, Q2) diverge more because their short lookbacks amplify transient price noise differently.

The flat convergence slopes indicate coordination is **structural, not progressive** — it reflects the shared signal architecture, not accumulating social learning. This is an important null result for Q3: if coordination were progressive, it would suggest implicit learning or herding dynamics. The flat slope suggests the coordination observed here is largely a byproduct of shared signal type, which motivates architectural diversity tests.

### Limitations

1. **Synthetic market**: Results are in a simplified simulated environment, not real markets. Price impact is modeled but simplified.
2. **Shared signal architecture**: All autonomous agents use the same momentum + mean-reversion signal framework, differing only in parameterization. Real autonomous agents would have diverse architectures.
3. **No learning**: Agents do not adapt their strategies over time. Q3 (Adversarial Adaptation) will address this.
4. **10 assets, 500 steps**: Small market dimensionality and short horizon. Scaling effects are untested.

### Next Steps

- **Q2 (Regime Manufacturing)**: Do agent populations *create* market regimes (volatility clusters, momentum/reversal cycles)?
- **Architectural diversity**: Test whether coordination persists when agents use fundamentally different signal types
- **Population scaling**: Vary the number of agents (5, 25, 100) to study coordination as a function of population size

## Reproducibility

```bash
# Run the full experiment
python experiments/q1_batch_runner.py --n-seeds 200 --workers 8 --output-dir results/q1_experiment

# Generate analysis and figures
python experiments/q1_analysis.py --input results/q1_experiment
```

All code: `hangar/agents/autonomous_agent.py`, `experiments/q1_batch_runner.py`, `experiments/q1_analysis.py`
