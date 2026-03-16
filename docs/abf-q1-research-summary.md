# Do AI-Driven Trading Agents Create Measurable Momentum and Reversal Patterns?

**Lewis Smith | HARBOR Project | March 2026**

---

## Abstract

We test whether volatility-targeting and trend-following algorithms — representative of the broader class of AI-driven trading agents — create measurable momentum persistence and subsequent reversal in equity returns. Using an event-study design with local projections on S&P 500 daily returns (2010–2025), we find **weak and time-dependent evidence** for the hypothesized mechanism. Post-2020 data shows statistically significant shock-persistence effects at the 1-day horizon (p < 0.01), but these effects are economically small (~1.3 bps) and do not survive at longer horizons or in the full sample. We interpret this as suggestive but insufficient evidence, motivating a simulation-based causal testing approach in ongoing work.

## 1. Hypothesis and Identification Strategy

**Hypothesis:** When AI-driven agents (volatility-targeting algorithms, trend-followers, and similar systematic strategies) respond to high-volatility shocks, their mechanical deleveraging creates short-term momentum persistence that subsequently reverses at medium horizons.

**Model:** We estimate local projections with Newey-West HAC standard errors:

> r_{t+h} = α + β_h · Shock_t + γ_h · (Shock_t × VolProxy_t) + Γ · X_t + ε_{t+h}

where h ∈ {1, 5, 21} trading days. The key coefficient is **γ_h** (the interaction between shock occurrence and vol-control pressure), which isolates the effect of agent-driven deleveraging from baseline volatility. The hypothesis predicts γ > 0 at short horizons (persistence) and γ < 0 at longer horizons (reversal).

**Controls:** Baseline realized volatility (21-day), liquidity proxy (21-day mean absolute return), rolling market beta (126-day), day-of-week indicators.

**Shock definition:** Top 5% of 21-day realized volatility changes (primary). Alternatives: VIX jumps, realized volatility level, range-based volatility.

## 2. Data and Sample

- **Universe:** S&P 500 constituents (up to 75 liquid names), daily close prices via yfinance
- **Sample:** January 2010 – December 2025 (3,897 trading days)
- **Vol-control proxy:** Constructed from rolling volatility changes, representing aggregate deleveraging pressure from volatility-targeting strategies
- **Shock events:** 201 identified shock days (top 5% percentile)
- **Caveat:** Current universe uses present-day S&P 500 constituents, introducing survivorship bias. Full historical membership (WRDS/CRSP) is planned but not yet integrated.

## 3. Results

### 3.1 Primary Specification (Full Sample)

| Horizon | β_h (Shock) | γ_h (Interaction) | R² | n |
|---------|------------|-------------------|-----|------|
| h = 1 | +0.0062 | −0.0051 | 0.004 | 3,897 |
| h = 5 | +0.0215 | −0.0205 | 0.016 | 3,893 |
| h = 21 | +0.0317 | −0.0302 | 0.040 | 3,877 |

**Full-sample inference is unreliable.** Standard errors and p-values are NaN for most coefficients at h = 5 and h = 21 due to near-perfect multicollinearity in the control matrix (day-of-week dummies with constant). At h = 1, the shock coefficient has p = 0.9999. **We cannot reject the null in the full sample.**

The coefficient pattern (β increasing, γ negative and approximately offsetting β) is *directionally consistent* with the hypothesis — shocks produce positive forward returns that are dampened when vol-control pressure is present — but the statistical evidence does not support this interpretation.

### 3.2 Sub-Sample Analysis

| Sample | Horizon | β_h | p(β_h) | γ_h | p(γ_h) | R² | n |
|--------|---------|-----|--------|-----|---------|-----|------|
| Post-2020 | h = 1 | +0.0136 | **0.004** | −0.0117 | **0.021** | 0.015 | 1,507 |
| Post-2020 | h = 5 | +0.0374 | 0.999 | −0.0412 | 0.999 | 0.067 | 1,503 |
| Post-2020 | h = 21 | +0.0315 | 0.999 | −0.0359 | 0.999 | 0.087 | 1,487 |
| Pre-2020 | h = 1 | −0.0013 | — | +0.0018 | — | 0.004 | 2,389 |
| Pre-2020 | h = 21 | +0.0240 | 0.999 | −0.0204 | 0.999 | 0.023 | 2,369 |

**Post-2020, h = 1 is the only specification with significant results.** The shock coefficient (+1.36 bps, p = 0.004) and interaction coefficient (−1.17 bps, p = 0.021) are both significant. The signs are as hypothesized: shocks predict positive 1-day forward returns, and vol-control pressure dampens this effect.

However: (a) the effect is economically small (1.4 bps), (b) it does not extend to 5-day or 21-day horizons in the same sub-sample, and (c) the pre-2020 sample shows no effect at any horizon.

### 3.3 Robustness Across Shock Definitions

The range-based volatility shock definition replicates the post-2020 h = 1 finding (β = +0.0129, p = 0.004; γ = −0.0129, p = 0.003). Other shock definitions (VIX jump, realized vol level) show weaker or no significance. The finding is not robust across all specifications.

### 3.4 Cumulative Abnormal Returns

| Horizon | Mean CAR | Median CAR | Range | n events |
|---------|----------|------------|-------|----------|
| h = 1 | −0.09 bps | — | −5.7% to +8.3% | 201 |
| h = 5 | +0.43 bps | — | −4.6% to +4.6% | 201 |
| h = 21 | +1.80 bps | — | −31.6% to +10.7% | 201 |

CARs show no consistent abnormal return pattern following shock events. The wide dispersion and extreme outliers (March 2020 COVID shock) suggest the average effect is dominated by a few extreme events rather than a systematic pattern.

## 4. Interpretation and Limitations

### What the data shows
- Weak, time-dependent evidence of shock → persistence at the 1-day horizon in post-2020 data
- No evidence of the medium-horizon reversal predicted by the hypothesis
- The effect, where present, is economically trivial (~1.3 bps)

### What the data does not show
- A robust, full-sample shock-persistence-reversal cycle
- Economic significance sufficient to inform trading rules or risk mitigations
- Evidence that the mechanism is specifically driven by AI/algorithmic behavior (vs. any other post-2020 structural change)

### Known limitations
1. **Multicollinearity:** The NaN standard errors at h > 1 indicate a specification issue in the control matrix. Dropping one day-of-week dummy or the constant would resolve this and may change full-sample inference. This is a bug to fix, not a fundamental limitation.
2. **Survivorship bias:** Using current S&P 500 constituents biases the sample toward survivors. Historical membership data would strengthen the analysis.
3. **Proxy quality:** The vol-control pressure proxy is constructed, not directly measured. Institutional flow data would provide a cleaner measure of agent deleveraging.
4. **Identification:** Even significant results are correlational. The event-study cannot isolate agent behavior as the causal mechanism. This motivates our planned agent simulation framework (Phase H5), which will test whether synthetic agent populations reproduce these patterns.

## 5. Next Steps

1. **Fix the multicollinearity issue** in the control matrix to recover valid inference at h = 5 and h = 21 across all samples
2. **Agent simulation (H5):** Build a market environment with heterogeneous agent populations (momentum, vol-targeting, mean-reversion) to test whether agent interactions causally produce the persistence-reversal pattern
3. **Q2 — Crowding analysis:** Test whether agent convergence predicts correlation spikes and drawdown amplification, providing additional evidence for the broader thesis
4. **Data upgrade:** Replace yfinance with survivorship-bias-free data (Massive API integration designed, pending implementation)

---

**Code and data:** All analysis is reproducible via `make q1` from the HARBOR repository. Results committed to `results/abf_q1/`.

**Figures:** Shock timeline, coefficient paths, CAR distributions, and robustness heatmap available in `results/abf_q1/figures/`.
