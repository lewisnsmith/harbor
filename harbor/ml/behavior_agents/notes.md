# `harbor.ml.behavior_agents` Notes

## Explanation
This module is intended for RL-based portfolio agents, including behavior-aware reward shaping.

Current status:
- Namespace scaffold only.

Intended RL setup:
- State: prices/returns/features/risk context.
- Action: portfolio weights or delta-weights.
- Reward: risk-adjusted objective with optional behavioral penalties.

Reward shaping ideas:
- Base return term: `r_p,t`.
- Risk penalty: `-lambda_var * Var` or drawdown penalty.
- Turnover penalty: `-lambda_turn * ||w_t - w_{t-1}||_1`.
- Behavioral terms (later phases): loss aversion, overconfidence, return chasing.

Why this matters for ABF:
- Allows controlled synthetic agents to test crowding/synchronization effects.

## Your Notes
- 
