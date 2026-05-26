"""Tests for MeetingPhaseProcessor — LLM-driven phase transitions.

All tests follow RED-GREEN-REFACTOR. The LLM client is stubbed so tests run
offline and deterministically. The bus, store, and manifest are real.
"""

from __future__ import annotations

import asyncio
import json
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from rehearse.bus import FrameBus
from rehearse.frames import PhaseSignal, TranscriptDelta
from rehearse.phases.phases import PhaseBudgets
from rehearse.phases.phases_llm import MeetingPhaseProcessor
from rehearse.storage import LocalFilesystemStore
from rehearse.types import ConsentState, Phase, Session, Speaker


# ---------------------------------------------------------------------------
# Helpers — fake clock and fake LLM client
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _make_text_response(text: str = "No transition needed.") -> Any:
    """Fake Anthropic response with a text block (no tool call)."""
    block = types.SimpleNamespace(type="text", text=text)
    return types.SimpleNamespace(content=[block])


def _make_tool_response(to: str, reason: str) -> Any:
    """Fake Anthropic response with a transition_phase tool call."""
    block = types.SimpleNamespace(
        type="tool_use",
        name="transition_phase",
        input={"to": to, "reason": reason},
    )
    return types.SimpleNamespace(content=[block])


class StubMessages:
    """Fake AsyncAnthropic().messages — records calls and returns scripted responses."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._responses:
            return self._responses.pop(0)
        return _make_text_response()


class StubLLMClient:
    def __init__(self, responses: list[Any] | None = None) -> None:
        self.messages = StubMessages(responses or [])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def phase_store(tmp_path: Path) -> tuple[LocalFilesystemStore, str]:
    store = LocalFilesystemStore(tmp_path, "https://example.test")
    session = Session(created_at=datetime.now(UTC), consent=ConsentState.GRANTED)
    (store.session_dir(session.id) / "session.json").write_text(
        session.model_dump_json(indent=2)
    )
    return store, session.id


def _make_processor(
    store: LocalFilesystemStore,
    session_id: str,
    bus: FrameBus,
    clock: FakeClock,
    llm_client: StubLLMClient,
    *,
    intake_min_dwell: int = 30,
    practice_min_dwell: int = 90,
    context_turns: int = 10,
) -> MeetingPhaseProcessor:
    return MeetingPhaseProcessor(
        session_id,
        store,
        bus,
        budgets=PhaseBudgets(
            intake_min_dwell_seconds=intake_min_dwell,
            practice_min_dwell_seconds=practice_min_dwell,
        ),
        clock=clock,
        llm_client=llm_client,
        context_turns=context_turns,
    )


def _user_turn(session_id: str, uid: str, text: str) -> TranscriptDelta:
    return TranscriptDelta(
        session_id=session_id,
        utterance_id=uid,
        speaker=Speaker.USER,
        text=text,
        is_final=True,
        ts_start=0.0,
        ts_end=1.0,
    )


# ---------------------------------------------------------------------------
# RED tests — all should fail before implementation exists
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_llm_call_before_min_dwell(
    phase_store: tuple[LocalFilesystemStore, str],
) -> None:
    """LLM must not be called until the min dwell floor has elapsed."""
    store, session_id = phase_store
    bus = FrameBus(session_id)
    clock = FakeClock(datetime(2026, 5, 18, 12, 0, tzinfo=UTC))
    client = StubLLMClient(responses=[_make_tool_response("practice", "ready")])

    proc = _make_processor(store, session_id, bus, clock, client, intake_min_dwell=30)
    await proc.bootstrap()
    task = asyncio.create_task(proc.run(bus.subscribe()))
    await asyncio.sleep(0)

    # Send a user turn within the dwell floor (only 5s elapsed).
    clock.now += timedelta(seconds=5)
    await bus.publish(_user_turn(session_id, "u1", "Let's practice."))
    await asyncio.sleep(0)

    assert len(client.messages.calls) == 0, "LLM should not be called before min dwell"
    assert proc.current_phase == Phase.INTAKE

    await bus.aclose()
    await task


@pytest.mark.asyncio
async def test_no_transition_when_llm_returns_text(
    phase_store: tuple[LocalFilesystemStore, str],
) -> None:
    """Text-only LLM response → phase unchanged, no PhaseSignal emitted."""
    store, session_id = phase_store
    bus = FrameBus(session_id)
    clock = FakeClock(datetime(2026, 5, 18, 12, 0, tzinfo=UTC))
    client = StubLLMClient(responses=[_make_text_response("Still in intake.")])

    proc = _make_processor(store, session_id, bus, clock, client, intake_min_dwell=5)
    await proc.bootstrap()

    signals: list[PhaseSignal] = []
    async def _collect(frames):
        async for f in frames:
            if isinstance(f, PhaseSignal):
                signals.append(f)
    collector = asyncio.create_task(_collect(bus.subscribe()))

    task = asyncio.create_task(proc.run(bus.subscribe()))
    await asyncio.sleep(0)

    clock.now += timedelta(seconds=10)
    await bus.publish(_user_turn(session_id, "u1", "Tell me about yourself."))
    await asyncio.sleep(0.05)

    assert len(client.messages.calls) == 1, "LLM should have been called once"
    assert proc.current_phase == Phase.INTAKE
    assert signals == [], "No PhaseSignal should be emitted"

    await bus.aclose()
    await task
    collector.cancel()


@pytest.mark.asyncio
async def test_transition_when_tool_called(
    phase_store: tuple[LocalFilesystemStore, str],
) -> None:
    """Tool call → phase advances, PhaseSignal emitted, manifest updated."""
    store, session_id = phase_store
    bus = FrameBus(session_id)
    clock = FakeClock(datetime(2026, 5, 18, 12, 0, tzinfo=UTC))
    client = StubLLMClient(
        responses=[_make_tool_response("practice", "User named scenario and is ready.")]
    )

    proc = _make_processor(store, session_id, bus, clock, client, intake_min_dwell=5)
    await proc.bootstrap()

    signals: list[PhaseSignal] = []
    async def _collect(frames):
        async for f in frames:
            if isinstance(f, PhaseSignal):
                signals.append(f)
    collector = asyncio.create_task(_collect(bus.subscribe()))

    task = asyncio.create_task(proc.run(bus.subscribe()))
    await asyncio.sleep(0)

    clock.now += timedelta(seconds=10)
    await bus.publish(_user_turn(session_id, "u1", "Let's roleplay the call."))
    await asyncio.sleep(0.05)

    assert proc.current_phase == Phase.PRACTICE
    assert len(signals) == 1
    assert signals[0].from_phase == Phase.INTAKE
    assert signals[0].to_phase == Phase.PRACTICE

    manifest = json.loads(
        (store.session_dir(session_id) / "session.json").read_text()
    )
    phases = [row["phase"] for row in manifest["phase_timings"]]
    assert phases == ["intake", "practice"]

    await bus.aclose()
    await task
    collector.cancel()


@pytest.mark.asyncio
async def test_reason_written_to_telemetry(
    phase_store: tuple[LocalFilesystemStore, str],
) -> None:
    """Transition reason must be appended to telemetry.jsonl."""
    store, session_id = phase_store
    bus = FrameBus(session_id)
    clock = FakeClock(datetime(2026, 5, 18, 12, 0, tzinfo=UTC))
    reason = "User explicitly said they are ready to practice."
    client = StubLLMClient(
        responses=[_make_tool_response("practice", reason)]
    )

    proc = _make_processor(store, session_id, bus, clock, client, intake_min_dwell=5)
    await proc.bootstrap()
    task = asyncio.create_task(proc.run(bus.subscribe()))
    await asyncio.sleep(0)

    clock.now += timedelta(seconds=10)
    await bus.publish(_user_turn(session_id, "u1", "Let's go."))
    await asyncio.sleep(0.05)

    await bus.aclose()
    await task

    telemetry_path = store.session_dir(session_id) / "telemetry.jsonl"
    assert telemetry_path.exists(), "telemetry.jsonl should be created"
    lines = [json.loads(l) for l in telemetry_path.read_text().splitlines() if l.strip()]
    transition_events = [l for l in lines if l.get("event") == "phase_transition"]
    assert len(transition_events) == 1
    assert transition_events[0]["reason"] == reason
    assert transition_events[0]["from"] == "intake"
    assert transition_events[0]["to"] == "practice"


@pytest.mark.asyncio
async def test_rolling_window_limits_context_sent_to_llm(
    phase_store: tuple[LocalFilesystemStore, str],
) -> None:
    """Only the last context_turns turns are passed to the LLM, not the full history."""
    store, session_id = phase_store
    bus = FrameBus(session_id)
    clock = FakeClock(datetime(2026, 5, 18, 12, 0, tzinfo=UTC))
    # context_turns=2 — only last 2 turns should appear in messages
    client = StubLLMClient(responses=[_make_text_response()] * 10)

    proc = _make_processor(
        store, session_id, bus, clock, client, intake_min_dwell=0, context_turns=2
    )
    await proc.bootstrap()
    task = asyncio.create_task(proc.run(bus.subscribe()))
    await asyncio.sleep(0)

    for i in range(4):
        await bus.publish(_user_turn(session_id, f"u{i}", f"Turn {i}"))
        await asyncio.sleep(0.02)

    await bus.aclose()
    await task

    assert len(client.messages.calls) == 4
    # The last call should contain at most 2 user turns in messages
    last_call_messages = client.messages.calls[-1]["messages"]
    user_messages = [m for m in last_call_messages if m["role"] == "user" and "Turn" in str(m.get("content", ""))]
    assert len(user_messages) <= 2, (
        f"Expected at most 2 user turns in window, got {len(user_messages)}"
    )


@pytest.mark.asyncio
async def test_llm_call_includes_current_phase_in_system(
    phase_store: tuple[LocalFilesystemStore, str],
) -> None:
    """System prompt sent to LLM must reference the current phase name."""
    store, session_id = phase_store
    bus = FrameBus(session_id)
    clock = FakeClock(datetime(2026, 5, 18, 12, 0, tzinfo=UTC))
    client = StubLLMClient(responses=[_make_text_response()])

    proc = _make_processor(store, session_id, bus, clock, client, intake_min_dwell=0)
    await proc.bootstrap()
    task = asyncio.create_task(proc.run(bus.subscribe()))
    await asyncio.sleep(0)

    await bus.publish(_user_turn(session_id, "u1", "Hello."))
    await asyncio.sleep(0.05)

    await bus.aclose()
    await task

    assert client.messages.calls, "LLM should have been called"
    system = client.messages.calls[0].get("system", "")
    assert "intake" in system.lower(), "System prompt should mention current phase"


@pytest.mark.asyncio
async def test_processor_closes_phase_timing_on_exit(
    phase_store: tuple[LocalFilesystemStore, str],
) -> None:
    """Phase timing row must be closed (ended_at set) when the bus closes."""
    store, session_id = phase_store
    bus = FrameBus(session_id)
    clock = FakeClock(datetime(2026, 5, 18, 12, 0, tzinfo=UTC))
    client = StubLLMClient()

    proc = _make_processor(store, session_id, bus, clock, client)
    await proc.bootstrap()
    task = asyncio.create_task(proc.run(bus.subscribe()))
    await asyncio.sleep(0)

    await bus.aclose()
    await task

    manifest = json.loads(
        (store.session_dir(session_id) / "session.json").read_text()
    )
    assert manifest["phase_timings"][-1]["ended_at"] is not None
