"""
harbor.research — DEPRECATED. This module has been removed.

Research functionality has been reorganized into:
- harbor.abf — ABF research experiments (Q1, Q2)
- harbor.abf.q1 — Shock → Persistence → Reversal analysis
- harbor.abf.q2 — Similarity, Crowding, Correlation analysis

See docs/abf-prd.md for the full research specification.

This file is kept only for git history continuity.
"""

import warnings

warnings.warn(
    "harbor.research is deprecated and will be removed. "
    "Use harbor.abf for research experiments.",
    DeprecationWarning,
    stacklevel=2,
)
