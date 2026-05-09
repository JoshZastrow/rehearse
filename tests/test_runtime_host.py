"""Verify RuntimeHost drives phases, artifacts, and transport correctly."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from rehearse.runtime import (
    CoachVoiceAdapter,
    RuntimeHost,
    RuntimePhaseTimeoutError,
    SessionArtifacts,
)
from rehearse.phases import PhaseBudgets
from rehearse.storage import LocalFilesystemStore
from rehearse.transport import InMemoryDuplexTransport, TransportEvent
from rehearse.types import ConsentState, Phase, Session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ZERO_DWELL_BUDGETS = PhaseBudgets(
    intake_seconds=120,
    practice_seconds=120,
    feedback_seconds=120,
    intake_min_dwell_seconds=0,
    practice_min_dwell_seconds=0,
)


class ScriptedCoach:
    """Return fixed responses in order.

    Uses a real 50ms sleep so asyncio.to_thread() calls inside LocalFilesystemStore
    (update_session) have time to complete before the coach returns. Without this,
    IntakeComplete arrives at PhaseProcessor AFTER the cue frame, breaking the
    happens-before guarantee.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._idx = 0

    async def respond(self, user_text: str, session_id: str) -> str:
        await asyncio.sleep(0.05)  # let thread-pool file I/O finish
        if self._idx < len(self._responses):
            reply = self._responses[self._idx]
        else:
            reply = "I hear you."
        self._idx += 1
        return reply


class CrashingCoach:
    """Raise on the Nth call to simulate a mid-run failure."""

    def __init__(self, crash_on: int = 2) -> None:
        self._calls = 0
        self._crash_on = crash_on

    async def respond(self, user_text: str, session_id: str) -> str:
        await asyncio.sleep(0.05)
        self._calls += 1
        if self._calls >= self._crash_on:
            raise RuntimeError("scripted coach failure")
        return "Tell me more."


def _make_host(
    tmp_path: Path,
    coach: CoachVoiceAdapter,
    *,
    budgets: PhaseBudgets | None = None,
    phase_timeout_s: float = 10.0,
) -> tuple[RuntimeHost, LocalFilesystemStore]:
    store = LocalFilesystemStore(tmp_path, "https://example.test")
    host = RuntimeHost(
        store,
        coach,
        budgets=budgets or _ZERO_DWELL_BUDGETS,
        phase_timeout_s=phase_timeout_s,
    )
    return host, store


async def _drain_until_control(
    endpoint: Any,
    *,
    timeout_s: float = 5.0,
    max_events: int = 20,
) -> TransportEvent | None:
    """Receive events until a control event arrives, or we hit the cap."""
    for _ in range(max_events):
        evt = await asyncio.wait_for(endpoint.receive(), timeout=timeout_s)
        if evt.kind == "control":
            return evt
    return None


# ---------------------------------------------------------------------------
# Happy path — all five artifacts produced, all three phases visited
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_all_artifacts_written(tmp_path: Path) -> None:
    """Three-phase scripted session produces all five artifact files."""
    coach = ScriptedCoach(
        [
            "Tell me more about that.",           # intake turn 1 reply
            "I see. What's your goal?",            # intake turn 2 reply
            "Great. Let's start practicing.",      # intake cue turn reply
            "Good try! Keep going.",               # practice turn reply
            "Here's what I noticed...",            # feedback cue reply
        ]
    )
    host, store = _make_host(tmp_path, coach)
    transport = InMemoryDuplexTransport()
    session_id = "sess-happy-001"

    async def _customer() -> None:
        c = transport.customer
        # --- INTAKE ---
        # Send turns without waiting for individual text replies: _watch_signals
        # may send the phase_transition control event before RuntimeHost sends
        # the text reply (both are sent while the coach sleeps).
        await c.send("text", payload={"text": "I need help negotiating salary with my manager."})
        await c.send("text", payload={"text": "My goal is a 20% raise."})
        # Cue triggers INTAKE→PRACTICE once IntakeComplete is received (after turn 2).
        await c.send("text", payload={"text": "I'm ready to practice this."})

        # Drain: skip text replies; stop when we get the PRACTICE control event.
        ctrl = await _drain_until_control(c)
        assert ctrl is not None
        assert ctrl.payload.get("event") == "phase_transition"
        assert ctrl.payload.get("phase") == "PRACTICE"

        # --- PRACTICE ---
        await c.send("text", payload={"text": "Hi, I wanted to discuss my compensation."})
        # Feedback cue triggers PRACTICE→FEEDBACK.
        await c.send("text", payload={"text": "Give me feedback on how I did."})

        ctrl2 = await _drain_until_control(c)
        assert ctrl2 is not None
        assert ctrl2.payload.get("phase") == "FEEDBACK"

        # --- end ---
        await c.send("control", payload={"event": "customer_done"})

    artifacts, _ = await asyncio.gather(
        host.run(session_id=session_id, transport=transport.runtime),
        _customer(),
    )

    assert isinstance(artifacts, SessionArtifacts)
    assert artifacts.session_json.exists()
    assert artifacts.transcript_jsonl.exists()
    assert artifacts.phase_timing_json.exists()
    assert artifacts.intake_json is not None and artifacts.intake_json.exists()
    assert artifacts.persona_json is not None and artifacts.persona_json.exists()

    timing = json.loads(artifacts.phase_timing_json.read_text())
    assert [row["phase"] for row in timing] == ["intake", "practice", "feedback"]


# ---------------------------------------------------------------------------
# Phase timeout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_phase_timeout_raises_and_cleans_up(tmp_path: Path) -> None:
    """A customer that sends nothing triggers RuntimePhaseTimeoutError."""
    coach = ScriptedCoach([])
    host, _ = _make_host(tmp_path, coach, phase_timeout_s=0.05)
    transport = InMemoryDuplexTransport()

    with pytest.raises(RuntimePhaseTimeoutError):
        await host.run(session_id="sess-timeout", transport=transport.runtime)

    # No coroutines should be left hanging — the test itself completing proves that.


# ---------------------------------------------------------------------------
# IntakeComplete happens-before phase_transition control event
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_intake_complete_before_phase_transition_control(tmp_path: Path) -> None:
    """session.json contains intake data before the PRACTICE control event arrives."""
    coach = ScriptedCoach(
        [
            "Tell me more.",
            "What's your goal?",
            "Let's practice!",
        ]
    )
    host, store = _make_host(tmp_path, coach)
    transport = InMemoryDuplexTransport()
    session_id = "sess-happens-before"

    intake_present_at_control: bool = False

    async def _customer() -> None:
        nonlocal intake_present_at_control
        c = transport.customer

        await c.send("text", payload={"text": "I need help with a performance review."})
        await c.send("text", payload={"text": "My manager keeps dismissing my contributions."})
        await c.send("text", payload={"text": "I'm ready to practice what I'll say."})

        ctrl = await _drain_until_control(c)
        assert ctrl is not None and ctrl.payload.get("phase") == "PRACTICE"

        # At the moment the control event arrives, check session.json for intake data.
        session_path = store.session_dir(session_id) / "session.json"
        session_data = json.loads(session_path.read_text())
        intake_present_at_control = session_data.get("intake") is not None

        await c.send("control", payload={"event": "customer_done"})

    await asyncio.gather(
        host.run(session_id=session_id, transport=transport.runtime),
        _customer(),
    )

    assert intake_present_at_control, (
        "session.json must contain intake data by the time the PRACTICE "
        "phase_transition control event reaches the customer"
    )


# ---------------------------------------------------------------------------
# current_phase property tracks transitions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_current_phase_tracks_transitions(tmp_path: Path) -> None:
    """host.current_phase reflects the live phase during and after the session."""
    coach = ScriptedCoach(
        ["Tell me more.", "I see.", "Let's start!", "Nice try.", "Good work!"]
    )
    host, _ = _make_host(tmp_path, coach)
    transport = InMemoryDuplexTransport()
    session_id = "sess-phase-prop"

    observed_at_start: Phase | None = None
    observed_at_practice: Phase | None = None

    async def _customer() -> None:
        nonlocal observed_at_start, observed_at_practice
        c = transport.customer

        observed_at_start = host.current_phase

        await c.send("text", payload={"text": "I want to negotiate a job offer."})
        await c.send("text", payload={"text": "The salary is too low."})
        await c.send("text", payload={"text": "I'm ready to practice."})

        ctrl = await _drain_until_control(c)
        assert ctrl is not None

        observed_at_practice = host.current_phase

        await c.send("control", payload={"event": "customer_done"})

    await asyncio.gather(
        host.run(session_id=session_id, transport=transport.runtime),
        _customer(),
    )

    assert observed_at_start == Phase.INTAKE
    assert observed_at_practice == Phase.PRACTICE
    assert host.current_phase == Phase.PRACTICE


# ---------------------------------------------------------------------------
# Mid-run crash → session.json present, intake.json absent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mid_run_crash_leaves_partial_artifacts(tmp_path: Path) -> None:
    """A coach crash after the first turn leaves session.json but no intake.json."""
    coach = CrashingCoach(crash_on=2)
    host, store = _make_host(tmp_path, coach)
    transport = InMemoryDuplexTransport()
    session_id = "sess-crash"

    async def _customer() -> None:
        c = transport.customer
        await c.send("text", payload={"text": "I need coaching help."})
        await c.receive()  # first reply succeeds
        await c.send("text", payload={"text": "Tell me more about the next step."})
        # coach crashes on this call — host.run() will raise

    with pytest.raises(RuntimeError, match="scripted coach failure"):
        await asyncio.gather(
            host.run(session_id=session_id, transport=transport.runtime),
            _customer(),
        )

    session_path = store.session_dir(session_id) / "session.json"
    intake_path = store.session_dir(session_id) / "intake.json"

    assert session_path.exists(), "session.json must be written at session start"
    assert not intake_path.exists(), "intake.json must not exist when run() raised"


# ---------------------------------------------------------------------------
# Transport closed mid-call (early customer_done) → clean exit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_early_hangup_exits_cleanly(tmp_path: Path) -> None:
    """Customer sending customer_done immediately produces a valid partial session."""
    coach = ScriptedCoach([])
    host, store = _make_host(tmp_path, coach)
    transport = InMemoryDuplexTransport()
    session_id = "sess-early-done"

    async def _customer() -> None:
        await transport.customer.send("control", payload={"event": "customer_done"})

    artifacts, _ = await asyncio.gather(
        host.run(session_id=session_id, transport=transport.runtime),
        _customer(),
    )

    assert artifacts.session_json.exists()
    assert artifacts.phase_timing_json.exists()
    assert artifacts.intake_json is None
