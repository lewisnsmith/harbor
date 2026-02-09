**HARBOR: Development History and Technical Evolution**

**Overview**

**HARBOR (Hierarchical Adaptive Risk-Based Optimization Routine) is a quantitative portfolio management system that evolved from basic competition work into a partially autonomous trading algorithm. This document traces the technical development across eight major iterations, documenting what worked, what failed, and why.**

**The progression reflects genuine learning rather than a predetermined roadmap. Each iteration addressed specific failures observed in the previous version. Early versions focused on model sophistication; later versions prioritized reliability and practical execution.**

**Main Goal: Produce a semi-autonomous Asset and Risk Management Algorithm for Retail Level/Individual Investors**

**Iteration 0.5 \- Monte Carlo Simulation Engine**

**\***During Wharton Global High School Investment Competition\*  
Monte Carlo Simulation Engine on a portfolio of equities with predetermined allocation weightings

Naive perspective on testing investing portfolio / first exposure to investing tools

**Iteration 1 \- Quantitative Factor Screening System**  
First Quantitative Framework  
**\***During Wharton Global High School Investment Competition\*

Factor scoring and evaluation model \- Inspired by AQR \- to test a basket of stocks and pick the best ones \+ lay foundation for retesting and reallocating in the future

Had:

- Static position sizing based on factor scores  
- Multi-factor scoring (quality, valuation, momentum)  
- Ranking-based stock selection

Why:  
Replaced random emotional intuition with discipline and comparison across companies and sectors

Fixed:  
Reduced emotional decision-making  
Improved baseline portfolio quality

Faults:

- Factor crowding risk  
- No forward-looking / stress testing awareness  
- Allocations still static and fragile under regime change  
- Correlation risk

**Iteration 2 \- Factor Screening \+ Monte Carlo Simulation**

**\***During Wharton Global High School Investment Competition\*  
Adapted and combined the Factor Scoring and Evaluation Model and the Monte Carlo Simulation Engine for stress testing

Had: 

- Factor Scores for stock selection  
- Monte Carlo Simulations to test portfolios under uncertainty  
- Drawdown and volatility diagnostics from Simulation metrics

Why:   
Ranking stocks wasnt enough for safe investing; we needed to understand how our portfolio combinations could behave across possible futures

Fixed: 

- Revealed tail risks and downside clustering  
- Improved awareness of drawdowns, volatility, and recovery paths  
- Informed better position sizing

Faults:

- Simulations depended heavily on historical assumptions and info (no forward-looking / projections)  
- Still no logic for optimal capital allocation  
- Portfolio construction still ad hoc

**Iteration 3:** 

**\***During Wharton Global High School Investment Competition\*  
This is pretty much where more formal portfolio construction begins  
Discovered and incorporated Mean-variance optimization for optimizing capital allocation 

Had: 

- Factor Driven expected returns  
- Monte Carlo stress testing  
- Mean-variance optimization for capital allocation

Why:  
MVO provided a mathematically clean way to translate scores into weights while minimizing variance

Fixed:

- Removed emotional/subjective sizing decisions  
- Quantified trade-offs between risk and return (helped us make more informed decisions regarding Competition case study client)  
- Created logic for allocation

Faults:

- Assumed stable and normal returns / ignored downside risk  
- Overweighted low-volatility assets regardless of thesis and accepted risk level  
- Produced mathematically optimal but thematically misaligned portfolios  
- Very sensitive to return estimation inputs (which also didn't have a model for determining, making this unoptimal)   
- Also generated a lot of noise when fed a large basket of stocks

**Iteration 4 \- Thesis Aligned Constraints and Clustering**  
Bridge to HARBOR

Introduced multiple levels of risk and diversification commands for the algorithm output to give multiple different possible strong portfolio constructions, behavioral constraints (caps and defensive minimums), and Early Clustering concepts to detect correlations

Had:

- Factor Driven expected returns  
- Monte Carlo stress testing  
- Mean-variance optimization for capital allocation  
- Hard behavioral constraints (caps, defensive minimums)  
- Early clustering concepts to detect correlation overlap  
- Reduced reliance on pure MVO outputs


Why  
Wanted optimization to serve the thesis for the client instead of overriding

Fixed:

- Prevented overconcentration in mathematically “efficient” but irrelevant assets  
- Reduced hidden correlation risk   
- Improved alignment with the researched thesis logic

Faults: 

- Prevented overconcentration in mathematically “efficient” but irrelevant assets  
- Reduced hidden correlation risk  
- Improved alignment with underdog enabler logic

**Iteration 5 \- HARBOR v0 (Titled Algo v5 in HARBOR notebooks)**

**Had:**   
Factor screening for financial quality  
Behavioral risk controls as hard constraints  
Hierarchical clustering for diversification by behavior  
Black-Litterman expected return adjustment  
Enhanced Monte Carlo stress testing with macro event stress testing  
Reinforcement learning aspect for adaptive weighting

Why:

Adapted the algorithm because each simpler version failed under real constraints: client alignment, capital deployment, psychological bias, regime uncertainty, and other mathematical fallacies

Fixed:

- Each layer exists to fix a specific failure observed earlier:  
- Factors select quality  
- Clustering prevents hidden risk  
- Behavioral rules block human bias  
- Black-Litterman corrects statistical distortions  
- Monte Carlo tests reality, not averages  
- Reinforcement learning enables adaptation without overreaction

Faults: 

- Still not autonomous  
- RL wasnt fully deployed  
- Mathematical models arent fully developed  
- No performance metrics or dashboard  
- No Backtesting  
- No Statistical testing or evaluation

**Iteration \- HARBOR V1 / Phase 1**

First paper trading system, slightly autonomous

Had:  
	\- All components from v0 integrated into executable pipeline  
	\- Alpaca API for live market data and order execution  
	\- Basic portfolio monitoring and position tracking  
	\- Risk parameter enforcement (max position size, sector exposure limits)  
	\- Daily rebalancing logic based on factor scores  
	\- Automated order generation and submission

Why:

Transitioned from theoretical framework read in Colab Notebooks to deployable system. Needed to validate whether the model's logic held under real market conditions with execution costs, slippage, and timing constraints.

Fixed:  
	\- Moved from notebook simulations to production environment  
	\- Introduced real-time decision-making capability  
	\- Added execution layer to test capital efficiency  
	\- Built monitoring infrastructure to track performance drift

Faults:  
	\- Execution logic was naive (market orders only, no price improvement)  
	\- Rebalancing frequency was arbitrary (daily) rather than optimized  
	\- No transaction cost modeling in portfolio construction  
	\- Risk parameters were hardcoded rather than derived from portfolio characteristics  
	\- Performance attribution was basic (total return only)  
	\- No systematic evaluation of factor contribution or model decay

Iteration 7 \- HARBOR V2 / Phase 2

Backtesting engine, performance analytics, transaction cost model

Had:  
	\- Historical backtesting framework using yfinance data  
	\- Sharpe ratio, maximum drawdown, and return metrics  
	\- Transaction cost estimation (spread \+ commission)  
	\- Factor contribution analysis  
	\- Rolling performance windows for stability assessment  
	\- Comparison against SPY benchmark

Why:

Needed objective evaluation of strategy performance before committing more capital. Without backtesting, I couldn't distinguish between model skill and market luck. Performance analytics revealed which components added value and which introduced noise.

Fixed:  
	\- Enabled systematic strategy evaluation across different market regimes  
	\- Identified that daily rebalancing was destroying alpha through transaction costs  
	\- Discovered that factor momentum had minimal predictive power in short windows  
	\- Built infrastructure to test parameter sensitivity

Faults:  
	\- Backtests assumed perfect historical data availability (no survivorship bias correction)  
	\- Risk model used historical volatility, which underestimated tail events  
	\- No regime detection—algorithm treated 2020, 2021, and 2022 markets identically  
	\- Factor scoring still relied on static weights rather than adaptive importance  
	\- Clustering implementation was unstable with changing market correlations

Iteration 8 \- HARBOR V2.5

Regime detection, dynamic factor weighting, improved clustering

Had:  
	\- Hidden Markov Model for regime classification (high vol / low vol / transition)  
	\- Dynamic factor weights that adjust based on recent factor performance  
	\- Hierarchical Risk Parity for more stable clustering  
	\- Volatility targeting to maintain consistent risk exposure  
	\- Regime-conditional rebalancing thresholds

Why:

Backtest results showed clear performance degradation during regime transitions. The algorithm needed to recognize when market behavior shifted and adjust its assumptions accordingly. Static factor weights performed well in stable periods but failed when factor returns decorrelated from fundamentals.

Fixed:  
	\- Algorithm now responds to changing market conditions  
	\- Factor importance adjusts based on rolling predictive performance  
	\- Portfolio construction accounts for time-varying correlation structures  
	\- Risk exposure scales down during high uncertainty periods

Faults:  
	\- HMM regime detection lagged actual regime changes (used only price data)  
	\- Dynamic factor weighting introduced overfitting risk  
	\- No economic rationale for regime states—purely statistical  
	\- Backtests showed improved metrics but live paper trading revealed implementation gaps  
	\- Computational cost increased significantly (clustering \+ HMM on every run)

Current State \- HARBOR V3 (In Development)

Focus: Production reliability, interpretability, robustness

Working On:  
	\- Migrating from daily batch processing to event-driven architecture  
	\- Replacing HMM with macro indicator-based regime classification  
	\- Implementing explicit transaction cost optimization in portfolio construction  
	\- Building logging and alerting system for model degradation  
	\- Developing explainability framework for portfolio decisions  
	\- Creating stress testing module with scenario analysis  
	\- Incorporating sentiment data alongside fundamental factors

Recognized Issues:  
	\- Need formal model validation framework separate from backtesting  
	\- Live performance has higher variance than backtests suggested  
	\- Factor model may be overfit to 2020-2024 data regime  
	\- Portfolio turnover still too high during volatile periods  
	\- Need better handling of corporate actions and delistings

Next Steps:  
	\- Implement walk-forward analysis with out-of-sample testing  
	\- Add regime-conditional transaction cost modeling  
	\- Build portfolio simulation environment with realistic execution  
	\- Develop automated model retraining pipeline  
	\- Create dashboard for real-time monitoring and manual override capability

\---

Reflections

This iteration history represents about 18 months of work, starting from competition-driven toy models to something approaching a production system. The progression wasn't linear—I went down several dead ends, overfit to historical data multiple times, and underestimated implementation complexity consistently.

Key lessons:

1\. Mathematical elegance ≠ practical performance. Mean-variance optimization produces beautiful Pareto frontiers but requires return forecasts I couldn't reliably generate. Black-Litterman helped, but the fundamental problem remains: expected returns are hard to estimate.

2\. Backtesting creates false confidence. Every iteration showed metric improvements in backtests, but paper trading revealed problems the historical data didn't capture. Walk-forward analysis helped, but nothing replaces live market interaction.

3\. Complexity has costs. Each added component (clustering, regime detection, RL) solved a specific problem but increased computational overhead, introduced new failure modes, and made debugging harder. Version 2.5 was theoretically superior but practically fragile.

4\. Transaction costs matter more than I expected. Early iterations ignored execution costs because they seemed small compared to expected returns. This was wrong. At realistic position sizes and rebalancing frequencies, transaction costs were often larger than the alpha I was trying to capture.

5\. The gap between research and production is wide. Converting Jupyter notebook logic into reliable scheduled execution required learning about error handling, API rate limits, data validation, logging, monitoring—none of which improved the model but all of which were necessary.

Current limitations:

\- The factor model is still fundamentally backward-looking. I use historical financial metrics to predict future returns, which only works when past patterns persist.
\- Regime detection helps but isn't predictive. The algorithm recognizes regime changes after they've occurred, not before.
\- I haven't solved the overfitting problem, just become more aware of it. Dynamic factor weighting might be extracting signal or might be curve-fitting to recent noise.
\- The system requires constant supervision. I check daily logs, verify execution, and manually intervene when behavior seems wrong. Full autonomy remains distant.

What I'd do differently:
\- Start with simpler models and add complexity only when I could prove (not assume) it helped out-of-sample performance
\- Build the backtesting and evaluation infrastructure before the strategy logic
\- Focus more on risk management and less on return optimization
\- Treat transaction costs as first-order constraints rather than minor adjustments
\- Document decisions and results more systematically from the beginning

Where I'm headed:

V3 focuses on making the system robust and interpretable rather than maximally performant. I'm more interested in understanding why the algorithm makes each decision than in squeezing out another 50 basis points. The goal is a system I can trust with real capital, which requires transparency, validation, and graceful failure modes.

I also have found interest in the impact trading and asset management algorithms and AI/ML have on retail trade.

Longer term, I'm questioning some fundamental assumptions. Maybe factor-based equity selection isn't the right approach. Maybe the alpha I'm chasing doesn't exist at my scale. Maybe systematic strategies require institutional resources I don't have. These are open questions.

The project taught me to respect the gap between theory and practice, to be skeptical of my own metrics, and to value simplicity and reliability over sophistication. It also made me significantly better at Python, deepened my understanding of portfolio theory, and gave me appreciation for why quantitative finance is hard.  
