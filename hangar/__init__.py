"""
HARBOR — Risk and Portfolio Framework
======================================

An open-source risk and portfolio framework with deep mathematical underpinnings
(HRP, Monte Carlo, risk decomposition) and ML/RL/DL extensions.

Module Structure
----------------
harbor.data             SP500 data loading, universe management, feature engineering.
harbor.risk             Covariance estimation, HRP, Monte Carlo VaR/CVaR, regime detection.
harbor.portfolio        Optimization, constraints, and allocation logic.
harbor.backtest         Backtest engine, metrics, and experiment runners.
harbor.features         Feature engineering and crowding proxies.
harbor.ml.volatility    Neural network volatility forecasters (experimental).
harbor.ml.behavior_agents  Deep RL behavioral portfolio agents (experimental).
harbor.abf              Artificial Behavior in Finance research layer.
harbor.abf.q1           Q1: Shock -> Persistence -> Reversal experiments.
harbor.abf.q2           Q2: Similarity, Crowding, Correlation experiments.

Research
--------
ABF (Artificial Behavior in Financial Markets) sits on top of HARBOR and uses it
to test whether systematic ML-driven strategies manufacture autocorrelation regimes,
crowding, and correlation spikes.

See docs/plan.md for the full roadmap and docs/abf-prd.md for the research spec.
"""

__version__ = "0.2.0-dev"
__author__ = "Lewis Smith"
