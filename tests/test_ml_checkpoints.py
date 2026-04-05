"""Tests for hangar.ml.checkpoints."""

from __future__ import annotations

import torch
import torch.nn as nn

from hangar.ml.checkpoints import (
    CheckpointMeta,
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _make_meta(name: str = "test_model", epoch: int = 0) -> CheckpointMeta:
    return CheckpointMeta(
        model_name=name,
        model_class="_TinyModel",
        created_at=f"2025-01-01T00:00:{epoch:02d}",
        epoch=epoch,
        metrics={"val_mse": 0.001},
        hyperparameters={"lr": 0.001},
    )


def test_save_and_load_checkpoint_roundtrips(tmp_path):
    """Save a model checkpoint and load it back; verify state dict matches."""
    model = _TinyModel()
    original_weight = model.fc.weight.data.clone()

    meta = _make_meta()
    pt_path = save_checkpoint(model, meta, model_dir=tmp_path)

    assert pt_path.exists()
    assert pt_path.with_suffix(".meta.json").exists()

    # Load into a fresh model
    fresh = _TinyModel()
    loaded_meta = load_checkpoint(fresh, pt_path)

    assert loaded_meta.model_name == "test_model"
    assert loaded_meta.epoch == 0
    assert torch.allclose(fresh.fc.weight.data, original_weight)


def test_list_checkpoints_filters_by_model_name(tmp_path):
    """Save two models with different names, verify filtering works."""
    m1 = _TinyModel()
    m2 = _TinyModel()

    save_checkpoint(m1, _make_meta("alpha", epoch=1), model_dir=tmp_path)
    save_checkpoint(m2, _make_meta("beta", epoch=2), model_dir=tmp_path)

    all_ckpts = list_checkpoints(model_dir=tmp_path)
    assert len(all_ckpts) == 2

    alpha_only = list_checkpoints("alpha", model_dir=tmp_path)
    assert len(alpha_only) == 1
    assert alpha_only[0].model_name == "alpha"

    beta_only = list_checkpoints("beta", model_dir=tmp_path)
    assert len(beta_only) == 1
    assert beta_only[0].model_name == "beta"


def test_latest_checkpoint_returns_most_recent(tmp_path):
    """Save multiple checkpoints, verify latest_checkpoint returns the newest."""
    model = _TinyModel()

    save_checkpoint(model, _make_meta("mymodel", epoch=1), model_dir=tmp_path)
    save_checkpoint(model, _make_meta("mymodel", epoch=5), model_dir=tmp_path)

    path = latest_checkpoint("mymodel", model_dir=tmp_path)
    assert path is not None
    assert path.exists()
    assert "epoch5" in path.name


def test_latest_checkpoint_returns_none_when_empty(tmp_path):
    """Verify None is returned when no checkpoints exist."""
    result = latest_checkpoint("nonexistent", model_dir=tmp_path)
    assert result is None
