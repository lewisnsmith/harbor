# `harbor.ml.volatility` Notes

## Explanation
This module is intended for neural-network volatility forecasting used by downstream allocation/risk modules.

Current status:
- Namespace scaffold only.

Intended pipeline:
1. Build training set from historical returns/features.
2. Train sequence model (LSTM/GRU).
3. Produce forward volatility estimate `sigma_hat_{t+1}`.
4. Feed into position sizing / covariance regime logic.

Potential model formulations:
- Regression to realized volatility target:
  - `y_t = RV_{t+1}`
  - loss examples: MSE, Huber, QLIKE-style objectives.

Potential integration points:
- Vol targeting:
  - scale exposure by `target_vol / sigma_hat`.
- Regime-aware covariance:
  - choose estimator/shrinkage intensity by predicted volatility state.

## Your Notes
- 
