# `harbor.risk` Notes

Files:
- `covariance.py`
- `hrp.py`
- `monte_carlo.py`
- `regime_detection.py`
- `correlation.py` (Q2 stub)

## Explanation
`harbor.risk` converts raw return series into structured risk objects used by portfolio construction and ABF analysis.

Implemented components in Phase 1:
1. Covariance estimation (sample + shrinkage).
2. Hierarchical Risk Parity allocation engine.
3. Monte Carlo VaR/CVaR estimation.
4. Q1 shock and vol-control proxy utilities.

### 1) Covariance estimation
- Sample covariance: direct empirical estimator.
- Shrinkage covariance (Ledoit-Wolf/OAS): regularized estimator that reduces noise sensitivity.

Math:
- Sample: `Sigma_hat = (1/(T-1)) * sum_t (r_t - r_bar)(r_t - r_bar)^T`
- Shrinkage (conceptually): `Sigma_tilde = delta * F + (1-delta) * Sigma_hat`
  where `F` is a structured target and `delta` chosen to minimize estimator risk.

### 2) HRP engine
HRP avoids direct matrix inversion and instead allocates via correlation structure.

Algorithm:
1. Convert covariance to correlation.
2. Convert correlation to distance:
   `d_ij = sqrt((1 - rho_ij)/2)`
3. Hierarchical clustering (`scipy.linkage`).
4. Quasi-diagonal ordering of leaves.
5. Recursive bisection:
   - split cluster in half
   - estimate each subcluster variance
   - allocate inversely to subcluster variance

Why this matters:
- Often more robust than classical MVO when covariance is noisy or nearly singular.

### 3) Monte Carlo VaR/CVaR
Pipeline:
1. Estimate `(mu, Sigma)` from historical aligned returns.
2. Simulate multivariate normal daily returns for `n_sims x horizon`.
3. Project to portfolio returns using weights.
4. Convert to losses and compute:
   - `VaR_alpha = quantile(losses, alpha)`
   - `CVaR_alpha = mean(losses | losses >= VaR_alpha)`

Interpretation:
- VaR: threshold loss level under confidence alpha.
- CVaR: average severity when losses exceed VaR.

### 4) Regime utilities
- `detect_vol_shocks`: flags dates where absolute change in rolling realized volatility exceeds percentile threshold.
- `vol_control_pressure_proxy`: ratio of short-window vol to long-window vol (clipped), used as a de-risking pressure proxy.

## Pending in this module
- `correlation.py` spike detection for Q2 remains unimplemented.

## Your Notes
- 
