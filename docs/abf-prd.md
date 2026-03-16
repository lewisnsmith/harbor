**Artificial Behavior in Financial Markets: Research PRD**

**Project:** HARBOR Asset Management
**Research Track:** Artificial Behavioral Finance (ABF)
**Version:** 2.0
**Date:** March 15, 2026 (originally February 13, 2026)
**Owner:** Lewis Smith

**Executive Summary**

This PRD defines the research agenda for investigating how AI models and autonomous agents create endogenous market dynamics through behavioral feedback loops. The core thesis: when market participants deploy AI-driven trading systems — from traditional volatility-targeting algorithms to LLM-based autonomous portfolio managers — at scale, they manufacture autocorrelation regimes, synchronized positioning, and amplified drawdowns that traditional behavioral finance cannot explain because the "agents" are algorithms, not humans.

This research integrates directly with the HARBOR asset management system through a feedback loop: empirical findings inform risk models and portfolio construction rules, which are validated through backtesting, and the results feed back into research refinement. The system also includes an agent simulation framework for causal testing and a retail impact analysis component.

**Research Objectives**

1. Determine whether AI-driven trading agents — including systematic strategies, ML models, and autonomous agents — create measurable return autocorrelation regimes and predictable reversal patterns

2. Quantify whether behavioral convergence among autonomous agents (shared signals, similar architectures, herding dynamics) is associated with synchronized positioning, correlation spikes, and drawdown amplification during stress periods

3. Test causal mechanisms through agent simulation: do synthetic agent populations, operating without access to real market data, independently reproduce the statistical signatures found empirically?

4. Measure the impact of AI-driven market dynamics on retail investor portfolios and validate that HARBOR's regime-aware mitigations provide measurable protection

**Narrative: From Builder to Structural Researcher**

**Stage 1 \- Builder:** Built asset allocation logic, risk frameworks, and a retail-institutional system during the Wharton Global Investment Competition.

**Stage 2 \- Structural Curiosity:** While building, recognized that risk models are covariance structures, clustering is eigenvector behavior, ML agents interact dynamically, and markets are multi-agent systems.

**Stage 3 \- Mathematical Formalization:** Asked "What mathematical structure governs this?" and leaned into linear algebra, statistical learning, optimization, and multi-agent dynamics.

This is not a pivot—it is maturation from implementation to mechanistic understanding.

**Research Questions**

**ABF Question 1 (Core Thesis \- Mandatory)**

**Question:** Do AI-driven trading agents create measurable momentum and reversal patterns in asset prices?

**Mechanism Hypothesis:** After high-volatility shocks, AI-driven agents (volatility-targeting algorithms, trend-followers, LLM-based bots) responding to the same market signals induce mechanical deleveraging (selling pressure) that:

1. Increases short-term momentum persistence (5-day horizon)

2. Increases medium-term reversal probability (1-month horizon)

3. Creates regime-dependent autocorrelation structures that differ from baseline volatility clustering

**Translation:** If large funds use ML-enhanced momentum plus volatility targeting, do they manufacture trends that subsequently reverse?

**Primary Test Plan (Pre-Specified):**

| Component | Specification |
| :---- | :---- |
| Shock Definition | Top 5% of 1-day realized volatility changes or VIX jump \> threshold |
| Persistence Metric | 5-day forward return autocorrelation; sign-preserving return days |
| Reversal Metric | 1-month cumulative abnormal return (CAR); sign-flip probability |
| Comparison | High-vol regime vs low-vol regime; with vs without vol-control proxy |

Table 1: ABF Question 1 test specification

**Key Variables:**

1. **Volatility measures:** Realized volatility, intraday range, VIX (index level), GARCH(1,1) forecasts

2. **Vol-control pressure proxies:** Risk-parity rebalancing proxy, volatility-managed exposure proxy, ETF flow reversals

3. **Controls:** Macro announcement days, liquidity (bid-ask spread, volume), market beta, sector exposure, day-of-week effects

**Success Criteria:**

1. Statistically significant (p\<0.05) and economically meaningful (Sharpe ratio change \>0.2) difference in persistence and reversal metrics across shock vs non-shock regimes

2. Robust to at least 3 alternative shock definitions and 2 sub-sample splits (pre-2020 vs post-2020; high-liquidity vs low-liquidity periods)

3. Mechanism-consistent pattern: strongest persistence appears when vol-control pressure proxy is elevated, not merely when volatility is high (this separates manufactured autocorrelation from natural volatility clustering)

**Identification Strategy:**

Event-study design with local projections:

rt+h=+1Shockt+2(ShocktVolControlProxyt)+Xt+t+h

where h{1,5,21} days, and Xt includes baseline volatility, liquidity, and macro controls.

**Expected outcome:** 2 should be significant and show opposite signs at short vs medium horizons (positive for persistence, negative for reversal).

**ABF Question 2 (Strongly Recommended)**

**Question:** Does behavioral convergence among autonomous agents amplify drawdowns and destabilize correlations?

**Hypothesis:** Periods of cross-asset correlation spikes and amplified drawdowns are partially explained by behavioral convergence among autonomous agents — whether through shared training data, similar architectures, herding dynamics, or convergent LLM reasoning — creating synchronized entry and exit behavior.

**Core Idea:** If many AI agents converge on similar strategies (momentum, liquidity proxies, sentiment features — or through emergent LLM-based reasoning), they may enter and exit positions together, causing:

1. Cross-asset correlation spikes

2. Liquidity cascades

3. Drawdown amplification beyond fundamental shocks

**Testing Approach:**

1. Identify correlation spike windows: periods where cross-asset correlation exceeds 75th percentile of trailing 12-month distribution

2. Build measurable similarity proxies:

   1. Signal correlation across assets (momentum, liquidity, sentiment)

   2. Momentum crowding proxy (e.g., cross-sectional dispersion of returns)

   3. ETF flow proxies (aggregate inflow/outflow pressure)

   4. Vol-control exposure proxies (risk-parity deleveraging indicators)

3. Test whether these proxies explain correlation expansion and drawdown severity beyond baseline volatility and liquidity controls

**Success Criteria:**

1. Similarity proxies predict correlation expansion out-of-sample (AUC \>0.65 for regime classification)

2. Proxies are strongest during stress windows and show incremental explanatory power (R2 increase \>5%) beyond volatility alone

3. At least one similarity proxy Granger-causes correlation spikes with 1–5 day lead time

**Connection to HARBOR:**

This directly informs:

1. Risk systems (correlation assumptions break during synchronized exits)

2. Portfolio construction (diversification fails when signal similarity is high)

3. AMA (Adaptive Multi-Asset) allocation logic (detect and de-risk before crowding unwinds)

**ABF Question 3 (Optional \- Time Permitting)**

**Question:** Does the proliferation of AI-driven strategies erode the effectiveness of established trading signals?

**Hypothesis:** As more autonomous agents exploit the same factors — whether through traditional systematic strategies or LLM-based agents entering markets at scale — crowding compresses returns and accelerates factor decay.

**Test (Optional):**

1. Estimate factor decay over decades using rolling-window factor regressions

2. Compare pre-ML era (pre-2010) vs post-ML era (2015+) with robustness to universe changes and transaction-cost regimes

3. Control for structural market changes (rise of passive investing, RegNMS, decimalization)

**Decision Rule:** Only pursue after Q1 is complete and Q2 has working proxies plus baseline results. This question is philosophically powerful but execution-heavy.

**Data and Universe**

**Asset Universe (Version 1\)**

**Primary universe:** U.S. equities in the S\&P 500, using historical constituent membership by date (not today's list) to avoid survivorship bias\[1\].

**Frequency:** Daily returns with optional intraday volatility proxies (5-minute bars or tick data if available).

**Identifiers:** PERMNO (CRSP identifiers) for robust corporate action handling and time-series consistency across mergers, delistings, and ticker changes\[2\].

**Sample Period:**

1. Baseline: 2010–2026 (captures post-GFC regime, rise of systematic strategies, COVID shock, 2022 vol regime)

2. Robustness: Split into pre-2020 vs post-2020 for regime stability checks

**Data Sources:**

| Data Type | Source | Notes |
| :---- | :---- | :---- |
| Price/Return | CRSP via WRDS | Survivorship-bias-free with adjustments |
| Volatility | CRSP (realized vol) \+ CBOE (VIX) | Intraday data if available |
| Constituents | WRDS S\&P 500 history | Time-varying membership |
| Volume/Liquidity | CRSP | Bid-ask spreads, turnover |
| Sentiment (optional) | AAII, news sentiment APIs | For Q2 signal proxies |
| ETF Flows (optional) | ETF.com, Bloomberg | For crowding proxies |

Table 2: Data sources and coverage

**Universe Caveat:** Do NOT use Wikipedia "current constituents" as backtest universe—this creates severe survivorship bias\[3\]. Always use historical membership dates.

**Unit of Observation**

**For Q1 (baseline):** Index level (SPX/SPY) for shock detection, then cross-sectional constituent-level analysis for heterogeneity checks (sector, size, liquidity terciles).

**For Q2:** Full constituent panel with time-varying membership, since signal similarity and crowding require cross-sectional dispersion measures.

**Agent Simulation Framework (Causal Testing)**

The empirical tests above (Q1-Q3) establish correlational evidence. To test causal mechanisms, HARBOR includes an agent simulation framework (`harbor.agents`) that enables counterfactual experiments impossible with real market data.

**Framework:**
- Price-impact market environment (linear temporary + square-root permanent impact)
- Heterogeneous agent populations: rule-based (momentum, vol-targeting, mean-reversion), ML-based (DRL behavioral agents), and LLM-based (planned)
- Config-driven population management with parameter distributions

**Causal Experiments:**
1. **Ablation:** Remove specific agent types and observe whether market patterns (crowding, correlation spikes) disappear
2. **Dose-response:** Vary agent concentration to test whether crowding increases monotonically
3. **Heterogeneity testing:** Compare high-diversity vs low-diversity populations
4. **Pattern validation:** Compare simulation-generated metrics against real-data metrics from Q1/Q2

**Success Criteria:**
- At least one synthetic metric distribution statistically consistent with real data (KS test p>0.05)
- Ablation experiments demonstrate at least one agent type whose removal significantly changes observed patterns
- Results constitute causal evidence linking agent behavior to empirically observed market dynamics

**Retail Impact Integration**

ABF research findings connect to retail investor outcomes through `harbor.retail`:
- Retail portfolio drawdown exposure during AI-driven regimes (measured from Q1/Q2 findings)
- Cost-of-crowding estimates for typical retail allocations
- Comparison of regime-aware vs naive retail strategies using HARBOR's mitigations
- Publication-quality figures: "retail drawdown amplification during crowding regimes"

This makes the retail investor focus tangible and measurable, moving beyond implicit concern to explicit quantification.

**Non-Goals (Explicit Scope Boundaries)**

1. Prove causality beyond doubt in version 1 (identification is strong, but not RCT-level)

2. Model every investor type (focus is on systematic/quantitative institutions)

3. Cover all asset classes simultaneously (start with equities; futures/crypto are future extensions)

4. Build a full agent-based market simulator in Q1 (synthetic agents are for later validation, not primary evidence)

**Dependencies**

1. **Data access:** WRDS institutional subscription (via UCLA or personal), or equivalent survivorship-bias-free source

2. **Compute:** Moderate (daily panel regressions, bootstrap inference); standard laptop sufficient for v1

3. **Volatility proxy validation:** Need at least one established vol-control proxy (literature survey or proprietary construction)

4. **Transaction cost assumptions:** Define bid-ask/slippage model for rebalancing cost estimates (not critical for Q1 but necessary for Q2 crowding impact)

**Success Metrics and Acceptance Criteria**

**For Research Output**

1. **Statistical rigor:** All primary results significant at p\<0.05 with bootstrap or Newey-West standard errors; at least 2 robustness checks per claim

2. **Economic significance:** Sharpe ratio changes \>0.2, correlation changes \>0.15, or reversal magnitudes \>2% (depending on test)

3. **Reproducibility:** Complete code pipeline (data pull → cleaning → analysis → plots) runnable end-to-end with \<5 commands

4. **Peer credibility:** Draft quality sufficient for UCLA faculty review (1–2 professors willing to provide detailed feedback)

**For HARBOR Integration**

1. Q1 results inform a "shock-aware position sizing rule" (reduce exposure when vol-control pressure proxy is elevated)

2. Q2 results inform a "crowding detection module" (flag high signal-similarity regimes and adjust diversification assumptions)

3. Backtest module can toggle "regime-aware" vs "baseline" modes and compare performance/risk metrics

**Phase 1 Implementation Status (as of February 20, 2026)**

- [x] Baseline Q1 shock configuration captured in `configs/abf/q1_shock_definitions.json`
- [x] Baseline Q2 regime configuration captured in `configs/abf/q2_regime_definitions.json`
- [x] Q1 shock utility scaffolded in `harbor/risk/regime_detection.py` (`detect_vol_shocks`, `vol_control_pressure_proxy`)
- [x] Backtest risk metric baseline scaffolded in `harbor/backtest/metrics.py`
- [x] End-to-end H1 pipeline script added in `experiments/h1_end_to_end_hrp_backtest.py`
- [ ] Replace seed membership data with full WRDS/CRSP constituent history for survivorship-bias-free ABF inference
- [ ] Complete Q1 local projection regressions and finalized figure set for milestone output

**Milestones and Timeline**

| Milestone | Date | Deliverable |
| :---- | :---- | :---- |
| Q1 Spec Frozen | Feb 25, 2026 | Shock definition, metrics, robustness list \+ 1 pilot plot |
| Q1 Baseline Results | Mar 15, 2026 | Event-study regression output, 3 key figures, draft outreach memo |
| UCLA Outreach | Mar 15, 2026 | Email 3–5 faculty with 1-page memo \+ repo link |
| Q2 Proxy Defined | Apr 15, 2026 | Signal similarity proxy validated \+ 1 correlation-spike case study |
| Q1 Draft Complete | Jun 15, 2026 | Working-paper-quality write-up with robustness checks |
| Q2 Results Stable | Jun 15, 2026 | Crowding proxy → correlation expansion results; integration plan |
| First Public Artifact | Sep 2026 | Preprint, workshop submission, or invited talk |

Table 3: Research milestones with target dates

**Weekly Checkpoints (Feb–Jun 2026\)**

1. **Weekly:** 1-page progress update (what worked, what blocked, next 3 priorities)

2. **Biweekly:** Code review \+ reproducibility check (can someone else run the pipeline?)

3. **Monthly:** External feedback loop (advisor, peer, or online community review)

**UCLA Faculty Connection Plan**

**Target Faculty**

| Name | Affiliation | Focus |
| :---- | :---- | :---- |
| Avanidhar Subrahmanyam | Anderson Finance | Stock market activity, behavioral finance\[4\] |
| Shlomo Benartzi | Anderson BDM (Emeritus) | Behavioral decision-making, digital nudges\[5\] |
| IPAM Faculty | Math Institute | Financial mathematics, cross-disciplinary workshops\[6\] |

Table 4: UCLA target faculty for collaboration

**Outreach Strategy**

1. Write 1-page collaboration pitch: problem statement, current repo/system, research questions, what you want from them (feedback, data pointers, co-advising, reading group)

2. Email 3–6 targets (2 finance/behavioral, 2 decision-making, 1–2 math/IPAM-adjacent) by Mar 15, 2026

3. Propose 20-minute meeting and attach 2 specific research questions you will test regardless of their response (shows seriousness)

4. Attend one public seminar/workshop (IPAM programs are designed for cross-field participation), then follow up with "I attended X; here's the 2-graph result from my simulator"

**What to Ask For**

1. Feedback on hypothesis framing and identification strategy

2. Data source recommendations or access (especially for vol-control proxies, ETF flows, or institutional positioning)

3. Co-advising or reading group participation (if they find the work compelling)

4. Introduction to PhD students or postdocs working on related topics

**Open Questions and Decisions Needed**

1. **Shock definition:** Top 5% of realized vol changes, or VIX absolute threshold, or both with robustness?

2. **Vol-control proxy:** Use published risk-parity proxy, construct proprietary proxy from ETF flows, or both?

3. **Universe expansion:** When to add futures (managed-futures universe) or ETFs (for flow analysis)?

4. **Intraday data:** Worth the complexity for v1, or stick to daily and add intraday in v2?

5. **Q3 timing:** Pursue immediately after Q2, or defer to separate paper?

**Integration with HARBOR System**

**The Feedback Loop**

ABF research and HARBOR's asset management stack form a feedback loop:
1. ABF discovers patterns (Q1: momentum/reversal, Q2: crowding/correlation, Q3: signal erosion)
2. HARBOR operationalizes findings into risk rules and portfolio construction
3. Backtesting validates that mitigations protect portfolios (especially retail)
4. Results inform further research refinement

**Module Mapping**

| ABF Research Component | HARBOR Module |
| :---- | :---- |
| Shock detection + vol-control proxy | risk/regime\_detection.py |
| Persistence/reversal metrics | backtest/metrics.py |
| Crowding proxies + signal similarity | abf/q2/proxies.py |
| Correlation spike detection | abf/q2/detection.py |
| Regime-aware position sizing | portfolio/construction.py |
| Agent simulation | agents/ |
| Agent-to-empirical validation | abf/agent\_validation/ |
| Retail impact analysis | retail/ |

Table 5: ABF research to HARBOR codebase mapping

**Backtest Integration**

The backtest engine supports two modes:

1. **Baseline mode:** Standard risk model with fixed correlation/volatility assumptions

2. **Regime-aware mode:** Dynamically adjust position sizing and diversification assumptions based on shock/crowding proxies

Performance comparison (Sharpe ratio, max drawdown, turnover) validates whether ABF insights translate to implementable improvements.

**Agent Simulation Integration**

The agent simulation framework (H5) generates synthetic market data in the same format as real data, enabling the entire ABF analysis pipeline to run on simulated data without modification. This provides the causal testing layer that empirical analysis alone cannot offer.

**References**

\[1\] WRDS. (2026). S\&P 500 Constituent Changes Over Time. Wharton Research Data Services. [https://wrds-www.wharton.upenn.edu/classroom/sp500-introduction/over-time/](https://wrds-www.wharton.upenn.edu/classroom/sp500-introduction/over-time/)

\[2\] CRSP. (2025). CRSP US Stock Databases. Center for Research in Security Prices. [https://www.crsp.org/research/crsp-us-stock-databases/](https://www.crsp.org/research/crsp-us-stock-databases/)

\[3\] Business Law Review. (2018). The (Mis)uses of the S\&P 500\. University of Chicago. [https://businesslawreview.uchicago.edu/print-archive/misuses-sp-500](https://businesslawreview.uchicago.edu/print-archive/misuses-sp-500)

\[4\] UCLA Anderson. (2025). Avanidhar Subrahmanyam Faculty Profile. UCLA Anderson School of Management. [https://www.anderson.ucla.edu/faculty-and-research/finance/faculty/subrahmanyam](https://www.anderson.ucla.edu/faculty-and-research/finance/faculty/subrahmanyam)

\[5\] UCLA Anderson. (2023). Shlomo Benartzi Faculty Profile. UCLA Anderson School of Management. [https://www.anderson.ucla.edu/faculty-and-research/faculty-directory/benartzi](https://www.anderson.ucla.edu/faculty-and-research/faculty-directory/benartzi)

\[6\] IPAM. (2025). Institute for Pure and Applied Mathematics. UCLA. [https://www.ipam.ucla.edu](https://www.ipam.ucla.edu)
