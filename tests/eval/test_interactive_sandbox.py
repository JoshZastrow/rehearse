"""Tests for run_interactive_session.

Uses mock backends — no Modal or GPU required.
"""
from __future__ import annotations

import pytest

from rehearse.bus import FrameBus
from rehearse.frames import AudioChunk, EndOfCall
from rehearse.types import Speaker


class _FakeBackend:
    """Backend that publishes AudioChunk then EndOfCall when send_caller_audio is called."""

    def __init__(self) -> None:
        self._bus: FrameBus | None = None
        self._session_id: str = ""

    async def start(self, session_id: str, bus: FrameBus) -> None:
        self._bus = bus
        self._session_id = session_id

    async def send_caller_audio(self, pcm: bytes) -> None:
        assert self._bus is not None
        await self._bus.publish(
            AudioChunk(session_id=self._session_id, speaker=Speaker.COACH, pcm16_16k=pcm, ts=0.0)
        )
        await self._bus.publish(
            EndOfCall(session_id=self._session_id, reason="hangup", ts=0.0)
        )

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_run_interactive_session_returns_session_result(monkeypatch):
    """run_interactive_session must return a SessionResult when EndOfCall is received."""
    from rehearse.eval.environments.interactive_sandbox import (
        SessionResult,
        run_interactive_session,
    )

    monkeypatch.setattr(
        "rehearse.eval.environments.interactive_sandbox.ModalInteractiveBackend",
        lambda endpoint: _FakeBackend(),
    )

    result = await run_interactive_session(
        session_id="test-session",
        provider_endpoint="ws://provider",
        caller_endpoint="ws://caller",
        max_duration_sec=5.0,
    )

    assert isinstance(result, SessionResult)
    assert result.session_id == "test-session"
    assert result.end_reason == "hangup"
    assert result.duration_sec >= 0.0


@pytest.mark.asyncio
async def test_run_interactive_session_times_out(monkeypatch):
    """run_interactive_session must return after max_duration_sec if EndOfCall never arrives."""
    from rehearse.eval.environments.interactive_sandbox import (
        SessionResult,
        run_interactive_session,
    )

    class _SilentBackend:
        async def start(self, session_id: str, bus: FrameBus) -> None:
            pass
        async def send_caller_audio(self, pcm: bytes) -> None:
            pass
        async def close(self) -> None:
            pass

    monkeypatch.setattr(
        "rehearse.eval.environments.interactive_sandbox.ModalInteractiveBackend",
        lambda endpoint: _SilentBackend(),
    )

    result = await run_interactive_session(
        session_id="timeout-session",
        provider_endpoint="ws://provider",
        caller_endpoint="ws://caller",
        max_duration_sec=0.1,  # real wall time; keep small but non-zero
    )

    assert result.end_reason == "timeout"
