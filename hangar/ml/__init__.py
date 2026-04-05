"""
harbor.ml — Machine learning and deep learning extensions.

Status: Experimental scaffolding (Layers 2–3). These modules have unit tests but have not been
validated against classical baselines or integrated into the production
pipeline. Formal validation (NN vol vs GARCH/EWMA, DRL vs buy-and-hold) is
required before promotion to production status.

Sub-packages:
- harbor.ml.volatility: LSTM/GRU volatility forecasters.
- harbor.ml.behavior_agents: Deep RL behavioral portfolio agents.

Utilities:
- harbor.ml.checkpoints: File-based model checkpoint registry.
"""

from harbor.ml.checkpoints import (
    CheckpointMeta,
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)

__all__ = [
    "CheckpointMeta",
    "latest_checkpoint",
    "list_checkpoints",
    "load_checkpoint",
    "save_checkpoint",
]
