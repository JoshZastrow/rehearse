"""Test the runtime's live phase controller and manifest timing updates."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rehearse.bus import FrameBus
from rehearse.frames import TranscriptDelta
from rehearse.phases import PhaseBudgets, PhaseProcessor
from rehearse.storage import LocalFilesystemStore
from rehearse.types import ConsentState, Phase, Session, Speaker


class FakeClock:
    """Provide deterministic wall-clock timestamps to the phase controller."""

    def __init__(self, now: datetime) -> None:
        """Store the current fake time."""
        self.now = now

    def __call__(self) -> datetime:
        """Return the current fake timestamp."""
        return self.now


@pytest.fixture
def phase_store(tmp_path: Path) -> tuple[LocalFilesystemStore, str]:
    """Create a session manifest for phase-controller tests."""
    store = LocalFilesystemStore(tmp_path, "https://example.test")
    session = Session(created_at=datetime.now(UTC), consent=ConsentState.PENDING)
    (store.session_dir(session.id) / "session.json").write_text(session.model_dump_json(indent=2))
    return store, session.id


@pytest.mark.asyncio
async def test_phase_processor_bootstraps_manifest_and_transitions_on_cues(
    phase_store: tuple[LocalFilesystemStore, str],
) -> None:
    store, session_id = phase_store
    bus = FrameBus(session_id)
    clock = FakeClock(datetime(2026, 4, 28, 12, 0, tzinfo=UTC))
    processor = PhaseProcessor(
        session_id,
        store,
        bus,
        budgets=PhaseBudgets(intake_seconds=60, practice_seconds=180, feedback_seconds=60),
        clock=clock,
    )

    await processor.bootstrap()
    run_task = asyncio.create_task(processor.run(bus.subscribe()))
    await asyncio.sleep(0)
    await bus.publish(
        TranscriptDelta(
            session_id=session_id,
            utterance_id="u1",
            speaker=Speaker.USER,
            text="I need help with a compensation conversation.",
            is_final=True,
            ts_start=0.0,
            ts_end=0.1,
        )
    )
    await asyncio.sleep(0)
    await bus.publish(
        TranscriptDelta(
            session_id=session_id,
            utterance_id="u2",
            speaker=Speaker.USER,
            text="Let's practice what I should actually say.",
            is_final=True,
            ts_start=0.2,
            ts_end=0.3,
        )
    )
    await asyncio.sleep(0)
    await bus.publish(
        TranscriptDelta(
            session_id=session_id,
            utterance_id="u3",
            speaker=Speaker.USER,
            text="How did that go? Give me feedback.",
            is_final=True,
            ts_start=0.4,
            ts_end=0.5,
        )
    )
    await asyncio.sleep(0)
    await bus.aclose()
    await run_task

    manifest = json.loads((store.session_dir(session_id) / "session.json").read_text())
    assert [row["phase"] for row in manifest["phase_timings"]] == [
        "intake",
        "practice",
        "feedback",
    ]
    assert processor.current_phase == Phase.FEEDBACK


@pytest.mark.asyncio
async def test_phase_processor_transitions_on_budget_expiry(
    phase_store: tuple[LocalFilesystemStore, str],
) -> None:
    store, session_id = phase_store
    bus = FrameBus(session_id)
    clock = FakeClock(datetime(2026, 4, 28, 12, 0, tzinfo=UTC))
    processor = PhaseProcessor(
        session_id,
        store,
        bus,
        budgets=PhaseBudgets(intake_seconds=1, practice_seconds=1, feedback_seconds=1),
        clock=clock,
    )

    await processor.bootstrap()
    run_task = asyncio.create_task(processor.run(bus.subscribe()))
    await asyncio.sleep(0)
    clock.now += timedelta(seconds=2)
    await bus.publish(
        TranscriptDelta(
            session_id=session_id,
            utterance_id="u1",
            speaker=Speaker.USER,
            text="Still talking.",
            is_final=True,
            ts_start=0.0,
            ts_end=0.1,
        )
    )
    await asyncio.sleep(0)
    clock.now += timedelta(seconds=2)
    await bus.publish(
        TranscriptDelta(
            session_id=session_id,
            utterance_id="u2",
            speaker=Speaker.USER,
            text="Still going.",
            is_final=True,
            ts_start=0.2,
            ts_end=0.3,
        )
    )
    await asyncio.sleep(0)
    await bus.aclose()
    await run_task

    manifest = json.loads((store.session_dir(session_id) / "session.json").read_text())
    assert [row["phase"] for row in manifest["phase_timings"]] == [
        "intake",
        "practice",
        "feedback",
    ]
    assert manifest["phase_timings"][0]["overran"] is True
    assert manifest["phase_timings"][1]["overran"] is True
