"""hangar.homelab.evaluation — Metrics registry and evaluation pipeline."""

from hangar.homelab.evaluation.registry import MetricsRegistry
from hangar.homelab.evaluation.summary import evaluate

__all__ = ["MetricsRegistry", "evaluate"]
