"""Eval for consent agent memory: returning callers get brief reminder.

These tests establish the contract BEFORE implementation.  They fail until
rehearse/memory.py is wired into ConsentGate.

Two scenarios verified with InMemoryCallerMemory:
  1. First-time caller  → hears CONSENT_PROMPT (full 55-word prompt)
  2. Returning caller   → hears CONSENT_REMINDER (short one-liner)
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rehearse.bus import FrameBus
from rehearse.consent import ConsentGate, ConsentGateConfig
from rehearse.frames import ConsentResolved
from rehearse.memory import InMemoryCallerMemory
from rehearse.participants import SpeakRequest
from rehearse.personas import CONSENT_PROMPT, CONSENT_REMINDER
from rehearse.storage import LocalFilesystemStore
from rehearse.types import ConsentState, Session, Speaker
from rehearse.frames import TranscriptDelta


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

CALLER_HASH = "abc123def456"


@pytest.fixture
def store(tmp_path: Path) -> tuple[LocalFilesystemStore, str]:
    s = LocalFilesystemStore(tmp_path, "https://example.test")
    session = Session(created_at=datetime.now(UTC), consent=ConsentState.PENDING)
    (s.session_dir(session.id) / "session.json").write_text(
        session.model_dump_json(indent=2)
    )
    return s, session.id


def _yes_frame(session_id: str) -> TranscriptDelta:
    return TranscriptDelta(
        session_id=session_id,
        utterance_id="u-yes",
        speaker=Speaker.USER,
        text="yes",
        is_final=True,
        ts_start=1.0,
        ts_end=1.5,
    )


class _CapturingSpeaker:
    """Records every say() call for assertion."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def say(self, request: SpeakRequest) -> None:
        self.spoken.append(request.text)


async def _run_consent(
    store: LocalFilesystemStore,
    session_id: str,
    *,
    caller_hash: str | None = None,
    memory: InMemoryCallerMemory | None = None,
) -> tuple[list[str], list[ConsentResolved]]:
    """Run a ConsentGate to completion, returning (spoken_lines, resolved_frames)."""
    bus = FrameBus(session_id)
    speaker = _CapturingSpeaker()
    resolved: list[ConsentResolved] = []

    async def _collect() -> None:
        async for frame in bus.subscribe():
            if isinstance(frame, ConsentResolved):
                resolved.append(frame)

    collector = asyncio.create_task(_collect())
    gate = ConsentGate(
        session_id,
        store,
        bus,
        speaker=speaker,
        on_decline=lambda: None,
        config=ConsentGateConfig(prompt_timeout_seconds=10, reprompt_limit=1),
        caller_hash=caller_hash,
        memory=memory,
    )
    task = asyncio.create_task(gate.run(bus.subscribe()))
    await asyncio.sleep(0.01)
    await bus.publish(_yes_frame(session_id))
    await asyncio.sleep(0.05)
    await bus.aclose()
    await task
    await collector
    return speaker.spoken, resolved


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_time_caller_gets_full_consent_prompt(
    store: tuple[LocalFilesystemStore, str],
) -> None:
    """A caller with no prior consent record hears the full prompt."""
    s, session_id = store
    memory = InMemoryCallerMemory()

    spoken, resolved = await _run_consent(
        s, session_id, caller_hash=CALLER_HASH, memory=memory
    )

    assert spoken == [CONSENT_PROMPT], (
        f"Expected full CONSENT_PROMPT but got: {spoken}"
    )
    assert resolved and resolved[0].state == ConsentState.GRANTED


@pytest.mark.asyncio
async def test_returning_caller_gets_brief_reminder(
    store: tuple[LocalFilesystemStore, str],
) -> None:
    """A caller whose consent is already recorded hears the brief reminder."""
    s, session_id = store
    memory = InMemoryCallerMemory()

    # Simulate a prior consent already in memory (as if a previous call ran)
    await memory.record_consent(CALLER_HASH)

    spoken, resolved = await _run_consent(
        s, session_id, caller_hash=CALLER_HASH, memory=memory
    )

    assert spoken == [CONSENT_REMINDER], (
        f"Expected CONSENT_REMINDER but got: {spoken}"
    )
    assert resolved and resolved[0].state == ConsentState.GRANTED


@pytest.mark.asyncio
async def test_consent_records_to_memory_on_grant(
    store: tuple[LocalFilesystemStore, str],
) -> None:
    """Granting consent on first call stores the fact so the next call gets reminder."""
    s, session_id = store
    memory = InMemoryCallerMemory()

    assert not await memory.has_prior_consent(CALLER_HASH)

    await _run_consent(s, session_id, caller_hash=CALLER_HASH, memory=memory)

    assert await memory.has_prior_consent(CALLER_HASH)


@pytest.mark.asyncio
async def test_null_memory_always_gives_full_prompt(
    store: tuple[LocalFilesystemStore, str],
) -> None:
    """When no memory is supplied, every call gets the full consent prompt."""
    s, session_id = store

    spoken, _ = await _run_consent(s, session_id)  # no caller_hash, no memory

    assert spoken == [CONSENT_PROMPT]
