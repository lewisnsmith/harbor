"""
harbor.signals — DEPRECATED. This module has been removed.

Signal computation has been reorganized:
- Feature engineering and crowding proxies → harbor.features
- ML-based signal generation → harbor.ml

This file is kept only for git history continuity.
"""

import warnings

warnings.warn(
    "harbor.signals is deprecated and will be removed. "
    "Use harbor.features for signal/feature computation.",
    DeprecationWarning,
    stacklevel=2,
)
