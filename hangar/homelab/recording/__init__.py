"""hangar.homelab.recording — Pluggable trace recording (Flight-ready)."""

from hangar.homelab.recording.protocol import Recorder
from hangar.homelab.recording.noop import NoopRecorder
from hangar.homelab.recording.jsonl import JsonlRecorder

__all__ = ["Recorder", "NoopRecorder", "JsonlRecorder"]
