"""harbor.ml.checkpoints — Simple file-based model registry for ML experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

_DEFAULT_MODEL_DIR: Path = Path(__file__).resolve().parents[2] / "data" / "models"


@dataclass
class CheckpointMeta:
    """Metadata stored alongside a model checkpoint."""

    model_name: str
    model_class: str
    created_at: str
    epoch: int
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    description: str = ""


def save_checkpoint(
    model: nn.Module,
    meta: CheckpointMeta,
    *,
    model_dir: Optional[Path] = None,
    filename: Optional[str] = None,
) -> Path:
    """Save a PyTorch model state dict and JSON metadata to disk.

    Parameters
    ----------
    model
        The PyTorch module whose ``state_dict()`` will be saved.
    meta
        Metadata describing this checkpoint.
    model_dir
        Directory for checkpoint files. Defaults to ``data/models/``.
    filename
        Base filename (without extension). Defaults to
        ``"{model_name}_epoch{epoch}_{timestamp}"``.

    Returns
    -------
    Path
        The path to the saved ``.pt`` file.
    """
    directory = Path(model_dir) if model_dir is not None else _DEFAULT_MODEL_DIR
    directory.mkdir(parents=True, exist_ok=True)

    if filename is None:
        ts = meta.created_at.replace(":", "").replace("-", "").replace("T", "_")
        ts = ts.split(".")[0]  # drop fractional seconds
        filename = f"{meta.model_name}_epoch{meta.epoch}_{ts}"

    pt_path = directory / f"{filename}.pt"
    meta_path = directory / f"{filename}.meta.json"

    torch.save(model.state_dict(), pt_path)

    with open(meta_path, "w") as f:
        json.dump(asdict(meta), f, indent=2)

    return pt_path


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    *,
    device: str = "cpu",
) -> CheckpointMeta:
    """Load a state dict into a model and return its metadata.

    Parameters
    ----------
    model
        An uninitialized (or initialized) model of the same architecture.
    checkpoint_path
        Path to the ``.pt`` checkpoint file. The corresponding
        ``.meta.json`` file must exist alongside it.
    device
        Device to map the state dict to (``"cpu"``, ``"cuda"``, ``"mps"``).

    Returns
    -------
    CheckpointMeta
        The metadata that was stored with this checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)
    meta_path = checkpoint_path.with_suffix(".meta.json")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    with open(meta_path) as f:
        raw = json.load(f)

    return CheckpointMeta(**raw)


def list_checkpoints(
    model_name: Optional[str] = None,
    *,
    model_dir: Optional[Path] = None,
) -> List[CheckpointMeta]:
    """List all available checkpoints, optionally filtered by model name.

    Parameters
    ----------
    model_name
        If provided, only return checkpoints whose ``model_name`` matches.
    model_dir
        Directory to scan. Defaults to ``data/models/``.

    Returns
    -------
    List[CheckpointMeta]
        Sorted by ``created_at`` descending (most recent first).
    """
    directory = Path(model_dir) if model_dir is not None else _DEFAULT_MODEL_DIR
    if not directory.exists():
        return []

    results: List[CheckpointMeta] = []
    for meta_path in directory.glob("*.meta.json"):
        with open(meta_path) as f:
            raw = json.load(f)
        meta = CheckpointMeta(**raw)
        if model_name is None or meta.model_name == model_name:
            results.append(meta)

    results.sort(key=lambda m: m.created_at, reverse=True)
    return results


def latest_checkpoint(
    model_name: str,
    *,
    model_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Return the path to the most recent checkpoint for a given model name.

    Returns
    -------
    Optional[Path]
        Path to the ``.pt`` file, or ``None`` if no checkpoints exist.
    """
    entries = list_checkpoints(model_name, model_dir=model_dir)
    if not entries:
        return None

    directory = Path(model_dir) if model_dir is not None else _DEFAULT_MODEL_DIR

    # Find the .pt file that matches this metadata
    ts = entries[0].created_at.replace(":", "").replace("-", "").replace("T", "_")
    ts = ts.split(".")[0]
    expected = f"{model_name}_epoch{entries[0].epoch}_{ts}.pt"
    pt_path = directory / expected

    if pt_path.exists():
        return pt_path

    # Fallback: scan for any .pt file with matching .meta.json
    for meta_path in sorted(directory.glob("*.meta.json"), reverse=True):
        with open(meta_path) as f:
            raw = json.load(f)
        if raw.get("model_name") == model_name:
            candidate = meta_path.with_suffix(".pt")
            if candidate.exists():
                return candidate

    return None
