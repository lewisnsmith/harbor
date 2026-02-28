# `harbor.portfolio` Notes

Files: `construction.py`

## Overview

`harbor.portfolio` translates risk and return estimates into weight vectors that can be backtested. It implements mean-variance optimization, risk parity, an HRP wrapper, and regime-aware gross-exposure scaling.

## Mean-Variance Optimization

The objective is solved with SLSQP: `min_w 0.5 * lambda * w^T Sigma w - mu^T w`, subject to a budget constraint (`sum(w) = target_leverage`) and bounds (long-only by default). The first term penalizes variance (risk), the second rewards expected return, and `lambda` controls risk aversion.

## Risk Parity

The goal is to equalize each asset's contribution to total portfolio variance. Portfolio variance is `sigma_p^2 = w^T Sigma w`. Marginal contribution is `mrc_i = (Sigma w)_i`. Risk contribution is `rc_i = w_i * mrc_i`. The algorithm iteratively updates weights so that each `rc_i` approaches `sigma_p^2 / N`.

## HRP Weights

A thin wrapper around `harbor.risk.hrp_allocation` for API consistency.

## Regime-Aware Position Sizing

Scales all baseline weights by a multiplier derived from a shock or crowding proxy. This preserves relative weights while reducing gross exposure, with the residual interpreted as cash.

## Practical Caveats

MVO depends heavily on expected return estimates. Risk parity and HRP still depend on covariance quality. Regime scaling does not re-optimize weights; it only scales exposure.
