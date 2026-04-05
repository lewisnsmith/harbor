# `harbor.ml` Notes

Subpackages:
- `volatility/`
- `behavior_agents/`

## Explanation
`harbor.ml` is the extension layer for machine learning and reinforcement learning components.

Current status:
- Mostly scaffolded in this repository snapshot.
- Intended to integrate with risk and portfolio modules rather than replace them blindly.

Design principle:
- ML output should be treated as an input to risk-aware decision systems, not as an unconstrained direct trading trigger.

Planned quantitative roles:
1. Volatility forecasting (sequence models).
2. Agent policy learning under risk-aware reward functions.
3. Integration with backtest evaluation under realistic costs.

## Mathematical ideas expected here
- Sequence modeling (LSTM/GRU): nonlinear time-series volatility prediction.
- RL objective (conceptually):
  - maximize expected discounted utility/reward,
  - typically `E[sum_t gamma^t R_t]`.
- Reward shaping to include risk penalties (drawdown, turnover, variance).

## Your Notes
- 
