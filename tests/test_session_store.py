"""Tests for session audio+transcript store components."""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "infra"))


def test_session_stored_event_fields():
    from rehearse.frames import SessionStoredEvent

    ev = SessionStoredEvent(
        session_id="sess-1",
        volume_path="/sessions/sess-1",
        artifacts=["caller_stream.pcm", "provider_stream.pcm", "tokens.jsonl", "mask.jsonl"],
        ts=1.0,
    )
    assert ev.session_id == "sess-1"
    assert ev.volume_path == "/sessions/sess-1"
    assert "mask.jsonl" in ev.artifacts
