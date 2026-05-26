"""Eval for consent agent memory: returning callers get brief reminder.

Unit tests use InMemoryCallerMemory (no external deps, run always).
Integration tests (marked live_api) use HonchoCallerMemory against the
real Honcho cloud API when HONCHO_API_KEY is set.

Two scenarios verified:
  1. First-time caller  → hears CONSENT_PROMPT, must say "yes"
  2. Returning caller   → hears CONSENT_REMINDER, immediately granted (no "yes")

pg0 spike result: pg0-embedded (embedded Postgres + pgvector) is confirmed
compatible with Honcho's DB_CONNECTION_URI. To run Honcho locally without
Docker, clone the Honcho server repo and set:
  DB_CONNECTION_URI=postgresql://postgres:postgres@127.0.0.1:<port>/postgres
(where the port comes from `Pg0(port=<port>).start()`).

Hindsight embedded (hindsight-all) is blocked on Python 3.13 threading
incompatibility in sentence-transformers. Track upstream fix before
switching HindsightCallerMemory into the integration test suite.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rehearse.bus import FrameBus
from rehearse.phases.consent import ConsentGate, ConsentGateConfig
from rehearse.frames import ConsentResolved, TranscriptDelta
from rehearse.memory.memory import InMemoryCallerMemory
from rehearse.participants import SpeakRequest
from rehearse.personas import CONSENT_PROMPT, CONSENT_REMINDERS
from rehearse.storage import LocalFilesystemStore
from rehearse.types import ConsentState, Session, Speaker


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
        self.said = asyncio.Event()  # set after first say()

    async def say(self, request: SpeakRequest) -> None:
        self.spoken.append(request.text)
        self.said.set()


async def _run_consent(
    store: LocalFilesystemStore,
    session_id: str,
    *,
    caller_hash: str | None = None,
    memory: InMemoryCallerMemory | None = None,
    send_yes: bool = True,
) -> tuple[list[str], list[ConsentResolved]]:
    """Run a ConsentGate to completion, returning (spoken_lines, resolved_frames).

    send_yes=True simulates a first-time caller who speaks "yes".
    send_yes=False is used for returning callers who are auto-granted.
    """
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
    # Wait for the gate to speak the prompt — at that point it has subscribed
    # to the bus and is ready to receive the user's "yes" frame.
    # For live Honcho tests the has_prior_consent API call can take >1s;
    # publishing "yes" before subscription is registered causes a missed frame.
    await asyncio.wait_for(speaker.said.wait(), timeout=15.0)
    if send_yes:
        await bus.publish(_yes_frame(session_id))
    await asyncio.wait_for(task, timeout=15.0)
    await bus.aclose()
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
async def test_returning_caller_gets_brief_reminder_and_is_immediately_granted(
    store: tuple[LocalFilesystemStore, str],
) -> None:
    """A returning caller hears the reminder and is immediately granted — no 'yes' needed."""
    s, session_id = store
    memory = InMemoryCallerMemory()

    # Simulate prior consent already in memory (as if a previous call ran)
    await memory.record_consent(CALLER_HASH)

    # send_yes=False: gate must resolve without user speaking
    spoken, resolved = await _run_consent(
        s, session_id, caller_hash=CALLER_HASH, memory=memory, send_yes=False
    )

    assert spoken[0] in CONSENT_REMINDERS, (
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


# ---------------------------------------------------------------------------
# Integration tests — Honcho server fixture (preferred) or cloud API fallback
#
# Two ways to run:
#   make setup-honcho && pytest -m live_api   (self-hosted, no cloud key)
#   HONCHO_API_KEY=<key> pytest -m live_api   (cloud API)
# ---------------------------------------------------------------------------

def _unique_caller() -> str:
    """Fresh caller hash per test run — prevents Honcho state from leaking across runs."""
    return f"test-caller-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def honcho_memory(honcho_server):
    """HonchoCallerMemory pointed at the session-scoped local Honcho server.

    Falls back to the cloud API when honcho_server is not available (i.e.
    lib/honcho/ not set up) and HONCHO_API_KEY is set in the environment.
    """
    from rehearse.memory.memory import HonchoCallerMemory

    if honcho_server:
        # Self-hosted Honcho via conftest.py fixture — no API key needed.
        return HonchoCallerMemory(base_url=honcho_server, workspace_id="rehearse-test")

    # Cloud fallback.
    api_key = os.environ.get("HONCHO_API_KEY")
    if not api_key:
        pytest.skip(
            "No Honcho available: run 'make setup-honcho' or set HONCHO_API_KEY"
        )
    return HonchoCallerMemory(api_key=api_key, workspace_id="rehearse-test")


@pytest.mark.asyncio
@pytest.mark.live_api
async def test_honcho_first_time_caller_gets_full_prompt(
    store: tuple[LocalFilesystemStore, str],
    honcho_memory,
) -> None:
    """First-time caller gets the full consent prompt via real Honcho."""
    s, session_id = store
    caller_hash = _unique_caller()
    spoken, resolved = await _run_consent(
        s, session_id, caller_hash=caller_hash, memory=honcho_memory
    )
    assert spoken == [CONSENT_PROMPT]
    assert resolved and resolved[0].state == ConsentState.GRANTED


@pytest.mark.asyncio
@pytest.mark.live_api
async def test_honcho_returning_caller_gets_reminder_and_immediate_grant(
    store: tuple[LocalFilesystemStore, str],
    honcho_memory,
) -> None:
    """Returning caller (prior consent in Honcho) gets reminder and is immediately granted."""
    s, session_id = store
    caller_hash = _unique_caller()
    await honcho_memory.record_consent(caller_hash)

    spoken, resolved = await _run_consent(
        s, session_id, caller_hash=caller_hash, memory=honcho_memory, send_yes=False
    )
    assert spoken[0] in CONSENT_REMINDERS
    assert resolved and resolved[0].state == ConsentState.GRANTED
