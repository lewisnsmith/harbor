**Artificial Behavior in Financial Markets: Research PRD**

**Project:** HARBOR Asset Management
**Research Track:** Artificial Behavioral Finance (ABF)
**Version:** 3.0
**Date:** March 16, 2026 (originally February 13, 2026)
**Owner:** Lewis Smith

**Executive Summary**

This PRD defines the research agenda for investigating how autonomous trading agents — LLM-powered, RL-based, and tool-using systems that reason, adapt, and interact strategically — create novel market dynamics that traditional behavioral finance and algorithmic trading research cannot explain. The core thesis: autonomous agents are qualitatively different from traditional systematic algorithms because they have agency (goal-setting, reasoning, adaptation), and when deployed at scale, they produce emergent coordination, manufactured regimes, and adversarial dynamics that reshape market structure.

HARBOR is the asset management algorithm that sparked this research. It serves three roles: infrastructure that agents run on, a test participant (HARBOR-as-agent competing against autonomous agents), and the origin story that motivated the questions.

**Narrative: From Builder to Agent Researcher**

**Stage 1 — Builder:** Built asset allocation logic, risk frameworks, and a retail-institutional system during the Wharton Global Investment Competition.

**Stage 2 — Structural Curiosity:** While building, recognized that risk models are covariance structures, clustering is eigenvector behavior, ML agents interact dynamically, and markets are multi-agent systems.

**Stage 3 — Initial Research:** Tested whether traditional vol-targeting algorithms create momentum and reversal patterns. Found weak, time-dependent evidence (post-2020 only, economically small). This motivated a sharper question.

**Stage 4 — The Pivot:** Realized the interesting question isn't about traditional algos — their effects are small because they're simple. The real question is about autonomous agents that reason, adapt, and interact strategically. These are qualitatively different, and their market impact is unstudied.

This is not a pivot away from the original work — it is a deepening. The initial investigation revealed where to look.

**Research Questions**

**ABF Question 1: Emergent Coordination (Core Thesis — Mandatory)**

**Question:** Do independently-built autonomous trading agents converge on similar strategies without explicit coordination?

**Mechanism Hypothesis:** When multiple autonomous agents (LLM-based, RL-based) trade the same market, they may converge on similar positions, timing, and risk management — even if built independently with different prompts, training data, and objectives. This convergence arises from:

1. Shared information environment (same market data, news, signals)
2. Similar reasoning patterns (LLM agents trained on similar financial corpora)
3. Convergent optimization (RL agents discovering similar reward-maximizing strategies)
4. Feedback loops (agents responding to each other's price impact)

**Translation:** If you give 50 different LLM agents the same market data and ask them to trade, do they end up doing the same thing? And if so, what happens to the market?

**Primary Test Plan:**

| Component | Specification |
| :---- | :---- |
| Agent Setup | Diverse autonomous agents: varied LLM prompts, RL training seeds, HARBOR configs |
| Convergence Metrics | Position correlation across agents, strategy similarity index, herding intensity |
| Key Test | Do agents that start different become more similar over time? |
| Empirical Complement | Post-2023 real-market signatures (correlation changes coinciding with LLM agent adoption) |

**Success Criteria:**

1. Statistical evidence that independent agents converge (or clear evidence they don't — a null result is publishable)
2. At least one convergence metric shows clear temporal trend in simulation
3. Empirical comparison with real-market patterns post-2023

**ABF Question 2: Regime Manufacturing (Strongly Recommended)**

**Question:** Do autonomous agent populations CREATE market regimes (momentum, mean-reversion, volatility clustering) that wouldn't exist without them?

**Hypothesis:** When autonomous agents trade at sufficient scale, their collective behavior manufactures statistical regimes — momentum when they herd into positions, mean-reversion when they unwind, volatility clustering when they simultaneously de-risk. These regimes are endogenous to agent behavior, not exogenous market properties.

**Core Idea:** Traditional finance treats market regimes as features of the economic environment. This question tests whether agent populations are the *cause*, not just participants in, regime dynamics.

**Testing Approach:**

1. Ablation experiments: run simulation with and without agent populations, compare regime structure using H2's regime detection tools
2. Dose-response: vary agent concentration from 10% to 80% of market volume, measure regime intensity
3. Regime detection: apply existing `harbor.risk` regime tools to synthetic data
4. Empirical complement: compare regime frequency and intensity pre-2020 vs post-2023

**Success Criteria:**

1. Ablation shows statistically significant regime differences with/without agents
2. Dose-response shows monotonic relationship between agent concentration and regime intensity
3. At least one manufactured regime type clearly identified and characterized

**ABF Question 3: Adversarial Adaptation (Time Permitting)**

**Question:** Do autonomous agents learn to exploit each other, and does this destabilize prices and harm retail investors?

**Hypothesis:** When agents detect other agents' strategies (through price impact, order flow patterns, or explicit reasoning), they may enter an adversarial adaptation cycle — exploiting each other's predictable behavior. This arms race could destabilize prices, widen spreads, and harm retail investors who lack the tools to participate.

**Testing Approach:**

1. Multi-round simulation where agents adapt strategies between rounds
2. Measure: price stability over time, strategy divergence/convergence, Sharpe ratio decay
3. Test whether adaptation leads to arms race (increasing instability), equilibrium (stability), or cycles
4. Retail impact: how does a naive buy-and-hold investor fare as agents adapt?

**Success Criteria:**

1. Clear characterization of adaptation dynamics (one of: arms race, equilibrium, or cycles)
2. Quantified retail portfolio impact during adversarial periods
3. Evidence of whether HARBOR's regime-awareness mitigates the damage

**The Three-Question Arc**

The questions form a progression:
- Q1 establishes whether agents converge (the precondition)
- Q2 tests whether convergence creates regimes (the mechanism)
- Q3 asks what happens when agents respond to the regimes they created (the feedback loop)

Each builds on the previous finding. A null result at any stage redirects the remaining questions.

**Data and Universe**

**Asset Universe (Version 1)**

**Primary universe:** U.S. equities in the S&P 500, using historical constituent membership by date (not today's list) to avoid survivorship bias.

**Frequency:** Daily returns with optional intraday volatility proxies.

**Identifiers:** PERMNO (CRSP identifiers) for robust corporate action handling.

**Sample Period:**

1. Baseline: 2010–2026 (captures post-GFC regime, rise of systematic strategies, COVID shock, 2022 vol regime)
2. Robustness: pre-2020 vs post-2023 split (pre-LLM-agent vs post-LLM-agent era)

**Data Sources:**

| Data Type | Source | Notes |
| :---- | :---- | :---- |
| Price/Return | CRSP via WRDS | Survivorship-bias-free with adjustments |
| Volatility | CRSP (realized vol) + CBOE (VIX) | Intraday data if available |
| Constituents | WRDS S&P 500 history | Time-varying membership |
| Volume/Liquidity | CRSP | Bid-ask spreads, turnover |

**Universe Caveat:** Do NOT use Wikipedia "current constituents" as backtest universe — this creates severe survivorship bias.

**Agent Simulation Framework**

The simulation framework (`harbor.agents`) is the primary research instrument. It enables causal experiments impossible with real market data.

**Architecture:**

```
harbor/agents/
    environment.py      # Market sim: price-impact model, order matching, state
    base_agent.py       # Abstract interface: observe() → decide() → act()
    rule_agents.py      # Momentum, vol-targeting, mean-reversion (baselines)
    llm_agents.py       # LLM-based agents (Claude/GPT via API)
    rl_agents.py        # RL agents (wraps harbor.ml.behavior_agents)
    harbor_agent.py     # HARBOR-as-agent: uses harbor.risk + harbor.portfolio
    adaptation.py       # Agent learning/adaptation between rounds
    population.py       # Population manager: spawn, configure, mix
    experiments.py      # Predefined experiment configs
    config.py           # Population configs, parameter distributions
    metrics.py          # Crowding index, flow imbalance, regime labels
    analysis.py         # Bridge: sim output → ABF-compatible DataFrames
```

**Agent Types:**

| Type | Source | Distinguishing Feature |
|------|--------|----------------------|
| Rule-based | `rule_agents.py` | Simple, deterministic — the "traditional algo" baseline |
| LLM-based | `llm_agents.py` | Reasons about market state, adapts via prompting |
| RL-based | `rl_agents.py` | Learns optimal policy through environment interaction |
| HARBOR | `harbor_agent.py` | Uses HARBOR's own risk/portfolio logic — the system tests itself |

**Key Design Decisions:**

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Price formation | Impact model (not order book) | Captures dynamics we care about; standard in agent-based finance lit |
| LLM agents | Real API calls (Claude/GPT) | Mocking loses the point — real reasoning is what makes agents different |
| HARBOR-as-agent | Wraps existing code | Tests whether the system's mitigations work against autonomous agents |
| Output format | Same DataFrame structure as real data | Reuses entire ABF analysis pipeline |

**Experiment Configurations:**

1. **Homogeneous LLM:** 50 LLM agents, same base prompt → emergent coordination test
2. **Mixed autonomous:** 20 LLM + 20 RL + 10 HARBOR → interaction dynamics
3. **Stress injection:** External vol shock + mixed population → cascading behavior
4. **Adaptation test:** Agents learn between rounds → convergence or divergence
5. **HARBOR resilience:** HARBOR agent in autonomous-agent-dominated market

**HARBOR's Triple Role**

1. **Infrastructure:** Agents use HARBOR's risk models, data pipeline, and portfolio construction
2. **Participant:** HARBOR-as-agent competes against autonomous agents, testing whether regime-awareness provides edge
3. **Origin story:** The asset management algorithm that sparked the research — "I built the system, then asked what happens when many autonomous systems like it exist"

**Non-Goals (Explicit Scope Boundaries)**

1. Study traditional algorithmic trading in depth (traditional algos are baseline context, not the focus)
2. Build a production trading system (HARBOR is research infrastructure, not a hedge fund)
3. Model every possible agent architecture (focus on LLM + RL as representative autonomous agent types)
4. Prove causality beyond doubt in version 1 (simulation evidence is strong but not RCT-level)

**Dependencies**

1. **Data access:** WRDS institutional subscription (via UCLA or personal), or equivalent survivorship-bias-free source
2. **LLM API access:** Claude and/or GPT API for LLM agent experiments
3. **Compute:** Moderate for simulation; LLM API costs scale with experiment size
4. **Existing HARBOR stack:** H1 + H2 complete and tested

**Success Metrics and Acceptance Criteria**

**For Research Output**

1. **Statistical rigor:** All primary results significant at p<0.05 with appropriate standard errors; at least 2 robustness checks per claim
2. **Causal evidence:** Simulation ablation experiments demonstrate at least one agent type whose removal significantly changes observed patterns
3. **Reproducibility:** Complete pipeline runnable end-to-end with <5 commands
4. **Peer credibility:** Draft quality sufficient for faculty review

**For HARBOR Integration**

1. HARBOR-as-agent results quantify whether regime-awareness helps against autonomous agents
2. Backtest module can compare HARBOR performance in "agent-dominated" vs "traditional" market environments
3. Results inform practical risk rules for retail investors

**Integration with HARBOR System**

**Module Mapping:**

| ABF Research Component | HARBOR Module |
| :---- | :---- |
| Market simulation environment | agents/environment.py |
| Agent types + adaptation | agents/llm_agents.py, rl_agents.py, harbor_agent.py |
| Population experiments | agents/population.py, experiments.py |
| Convergence/crowding metrics | agents/metrics.py |
| Regime detection on synthetic data | risk/regime_detection.py |
| HARBOR-as-agent performance | agents/harbor_agent.py + backtest/ |
| Retail impact analysis | retail/ |

**Deprecated: Old Q1 (Vol-Shock Persistence)**

The original Q1 tested whether vol-targeting algorithms create momentum persistence and reversal patterns. Results: weak, time-dependent evidence (post-2020 only, ~1.3bps, economically insignificant). This investigation was valuable as a learning experience and motivated the pivot to agent-specific research. Code has been removed. See `docs/abf-q1-research-summary.md` for the full write-up.

**References**

[1] WRDS. (2026). S&P 500 Constituent Changes Over Time. Wharton Research Data Services.

[2] CRSP. (2025). CRSP US Stock Databases. Center for Research in Security Prices.

[3] Business Law Review. (2018). The (Mis)uses of the S&P 500. University of Chicago.
