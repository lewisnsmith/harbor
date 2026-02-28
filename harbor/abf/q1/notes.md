# `harbor.abf.q1` Notes

## Overview

Q1 studies the mechanism: volatility shock leads to systematic de-risking pressure, which produces short-horizon persistence followed by medium-horizon reversal.

The namespace is currently scaffolded. Core utilities are implemented in `harbor/risk/regime_detection.py`, `harbor/backtest/metrics.py`, and `configs/abf/q1_shock_definitions.json`.

## Target Pipeline

1. Detect shock dates.
2. Build event windows for horizons `h in {1, 5, 21}`.
3. Compute persistence and reversal outcomes.
4. Run local projections with controls and interaction terms.
5. Produce tables and figures with robustness splits.

## Math Details

Event-study cumulative abnormal return: `CAR_{t,h} = product_{j=1..h}(1 + AR_{t+j}) - 1`, where `AR` is abnormal return relative to a benchmark or mean.

The local projection skeleton is `r_{t+h} = a_h + b_h Shock_t + c_h Shock_t*Proxy_t + Gamma_h X_t + e_{t+h}`. The expectation from the hypothesis is a positive persistence effect at short horizons and a stronger reversal effect at medium horizons.
