"""harbor.homelab.recording — Pluggable trace recording (Flight-ready)."""

from harbor.homelab.recording.protocol import Recorder
from harbor.homelab.recording.noop import NoopRecorder
from harbor.homelab.recording.jsonl import JsonlRecorder

__all__ = ["Recorder", "NoopRecorder", "JsonlRecorder"]
