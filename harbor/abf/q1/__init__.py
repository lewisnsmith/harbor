"""harbor.abf.q1 — ABF Question 1: Shock -> Persistence -> Reversal.

Provides local projection analysis, robustness sweeps, and figure generation
for testing whether ML-driven strategies create measurable medium-term
autocorrelation regimes following volatility shocks.
"""

from harbor.abf.q1.analysis import (
    build_control_matrix,
    compute_forward_returns,
    compute_return_autocorrelation,
    fit_local_projection,
    fit_local_projections,
)
from harbor.abf.q1.robustness import (
    apply_shock_definition,
    robustness_sweep,
    split_sample,
)
from harbor.abf.q1.visualization import (
    plot_car_paths,
    plot_coefficient_path,
    plot_robustness_heatmap,
    plot_shock_timeline,
)

__all__ = [
    "apply_shock_definition",
    "build_control_matrix",
    "compute_forward_returns",
    "compute_return_autocorrelation",
    "fit_local_projection",
    "fit_local_projections",
    "plot_car_paths",
    "plot_coefficient_path",
    "plot_robustness_heatmap",
    "plot_shock_timeline",
    "robustness_sweep",
    "split_sample",
]
