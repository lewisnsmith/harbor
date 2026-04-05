# `hangar.abf.q2` Notes

## Overview

Q2 studies whether similarity and crowding among systematic strategies predict correlation spikes and drawdown amplification.

The namespace is currently scaffolded. Supporting stubs and building blocks are in `hangar/features/crowding.py`, `hangar/risk/correlation.py`, and `configs/abf/q2_regime_definitions.json`.

## Target Pipeline

1. Compute crowding and similarity proxies.
2. Label correlation spike regimes.
3. Estimate predictive models and explanatory regressions.
4. Evaluate out-of-sample regime classification and drawdown linkage.

## Math Details

Proxy examples include signal similarity (pairwise correlation or cosine among signals) and cross-sectional dispersion: `Disp_t = std_i(r_{i,t})`, where lower dispersion can indicate synchronized behavior.

The classification target is `Spike_t = 1{Corr_t > Q_{0.75}(Corr_{t-252:t-1})}`.

Model forms include logistic classification (`Pr(Spike_{t+k}=1 | Proxy_t, X_t)`) and linear severity modeling (`Drawdown_t ~ Proxy_t + Controls_t`).
