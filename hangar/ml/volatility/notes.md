# `harbor.ml.volatility` Notes

## Overview

This module provides neural-network volatility forecasting used by downstream allocation and risk modules.

## Pipeline

1. Build training set from historical returns and features.
2. Train sequence model (LSTM/GRU).
3. Produce forward volatility estimate `sigma_hat_{t+1}`.
4. Feed into position sizing or covariance regime logic.

## Model Formulation

The regression target is realized volatility: `y_t = RV_{t+1}`. Loss function options include MSE, Huber, and QLIKE-style objectives.

## Integration Points

**Vol targeting** scales exposure by `target_vol / sigma_hat`. **Regime-aware covariance** chooses the estimator or shrinkage intensity based on the predicted volatility state.
