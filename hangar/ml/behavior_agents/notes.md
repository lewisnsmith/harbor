# `hangar.ml.behavior_agents` Notes

## Overview

This module provides RL-based portfolio agents with behavior-aware reward shaping.

## RL Setup

The state consists of prices, returns, features, and risk context. Actions are portfolio weights or delta-weights. The reward is a risk-adjusted objective with optional behavioral penalties.

## Reward Shaping

The base return term is `r_p,t`. Risk penalty takes the form `-lambda_var * Var` or a drawdown penalty. Turnover penalty is `-lambda_turn * ||w_t - w_{t-1}||_1`. Behavioral terms in later phases include loss aversion, overconfidence, and return chasing.

## ABF Relevance

This module allows controlled synthetic agents to test crowding and synchronization effects as part of the Q2 research question.
