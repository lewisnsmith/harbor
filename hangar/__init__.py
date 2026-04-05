"""
HANGAR — Risk and Portfolio Framework
======================================

An open-source risk and portfolio framework with deep mathematical underpinnings
(HRP, Monte Carlo, risk decomposition) and ML/RL/DL extensions.

Module Structure
----------------
hangar.data             SP500 data loading, universe management, feature engineering.
hangar.risk             Covariance estimation, HRP, Monte Carlo VaR/CVaR, regime detection.
hangar.portfolio        Optimization, constraints, and allocation logic.
hangar.backtest         Backtest engine, metrics, and experiment runners.
hangar.features         Feature engineering and crowding proxies.
hangar.ml.volatility    Neural network volatility forecasters (experimental).
hangar.ml.behavior_agents  Deep RL behavioral portfolio agents (experimental).
hangar.abf              Artificial Behavior in Finance research layer.
hangar.abf.q1           Q1: Shock -> Persistence -> Reversal experiments.
hangar.abf.q2           Q2: Similarity, Crowding, Correlation experiments.

Research
--------
ABF (Artificial Behavior in Financial Markets) sits on top of HANGAR and uses it
to test whether systematic ML-driven strategies manufacture autocorrelation regimes,
crowding, and correlation spikes.

See docs/plan.md for the full roadmap and docs/abf-prd.md for the research spec.
"""

__version__ = "0.2.0-dev"
__author__ = "Lewis Smith"
