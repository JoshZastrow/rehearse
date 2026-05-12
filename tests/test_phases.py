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
        budgets=PhaseBudgets(
            intake_seconds=60,
            practice_seconds=180,
            feedback_seconds=60,
            intake_min_dwell_seconds=30,
            practice_min_dwell_seconds=90,
        ),
        clock=clock,
        consent_getter=lambda: ConsentState.GRANTED,
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
    # Advance past the intake min-dwell floor so a cue transition is allowed.
    clock.now += timedelta(seconds=35)
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
    # Advance past the practice min-dwell floor.
    clock.now += timedelta(seconds=95)
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


@pytest.mark.asyncio
async def test_consent_grant_does_not_count_toward_intake_turns(
    phase_store: tuple[LocalFilesystemStore, str],
) -> None:
    """Regression for session ca4299ff: 'Yup' (consent) plus first answer
    should not flip intake -> practice before any intake content lands.
    """
    store, session_id = phase_store
    bus = FrameBus(session_id)
    clock = FakeClock(datetime(2026, 5, 6, 23, 2, 17, tzinfo=UTC))
    consent = {"state": ConsentState.PENDING}
    processor = PhaseProcessor(
        session_id,
        store,
        bus,
        budgets=PhaseBudgets(intake_min_dwell_seconds=30, practice_min_dwell_seconds=90),
        clock=clock,
        consent_getter=lambda: consent["state"],
    )

    await processor.bootstrap()
    run_task = asyncio.create_task(processor.run(bus.subscribe()))
    await asyncio.sleep(0)

    # User answers consent prompt — phase still INTAKE, consent moves to GRANTED.
    await bus.publish(
        TranscriptDelta(
            session_id=session_id,
            utterance_id="user-2",
            speaker=Speaker.USER,
            text="Yup.",
            is_final=True,
            ts_start=11.6,
            ts_end=13.0,
        )
    )
    await asyncio.sleep(0)
    consent["state"] = ConsentState.GRANTED

    # First post-consent reply. With the bug this was the 2nd "user turn" and
    # tripped the n>=2 cue. Fixed code only counts post-consent turns.
    clock.now += timedelta(seconds=4)
    await bus.publish(
        TranscriptDelta(
            session_id=session_id,
            utterance_id="user-4",
            speaker=Speaker.USER,
            text="Yeah, let's do it.",
            is_final=True,
            ts_start=15.1,
            ts_end=16.6,
        )
    )
    await asyncio.sleep(0)
    await bus.aclose()
    await run_task

    assert processor.current_phase == Phase.INTAKE
    manifest = json.loads((store.session_dir(session_id) / "session.json").read_text())
    assert [row["phase"] for row in manifest["phase_timings"]] == ["intake"]


@pytest.mark.asyncio
async def test_word_feedback_in_user_situation_does_not_trigger_transition(
    phase_store: tuple[LocalFilesystemStore, str],
) -> None:
    """Regression for session ca4299ff: the word 'feedback' inside a user's
    description of their situation should not end the practice phase.
    """
    store, session_id = phase_store
    bus = FrameBus(session_id)
    clock = FakeClock(datetime(2026, 5, 6, 23, 2, 17, tzinfo=UTC))
    processor = PhaseProcessor(
        session_id,
        store,
        bus,
        budgets=PhaseBudgets(intake_min_dwell_seconds=30, practice_min_dwell_seconds=90),
        clock=clock,
        consent_getter=lambda: ConsentState.GRANTED,
    )

    await processor.bootstrap()
    run_task = asyncio.create_task(processor.run(bus.subscribe()))
    await asyncio.sleep(0)

    # Drive intake -> practice via dwell + cue.
    clock.now += timedelta(seconds=35)
    await bus.publish(
        TranscriptDelta(
            session_id=session_id,
            utterance_id="u1",
            speaker=Speaker.USER,
            text="I have a comp negotiation with my CEO.",
            is_final=True,
            ts_start=0.0,
            ts_end=2.0,
        )
    )
    await asyncio.sleep(0)
    await bus.publish(
        TranscriptDelta(
            session_id=session_id,
            utterance_id="u2",
            speaker=Speaker.USER,
            text="Let's practice the call.",
            is_final=True,
            ts_start=2.0,
            ts_end=3.0,
        )
    )
    await asyncio.sleep(0.05)
    assert processor.current_phase == Phase.PRACTICE

    # User mentions "feedback" describing their situation — should NOT trip a
    # transition out of practice. Old greedy cue list contained bare "feedback".
    clock.now += timedelta(seconds=95)  # past practice min dwell
    await bus.publish(
        TranscriptDelta(
            session_id=session_id,
            utterance_id="u3",
            speaker=Speaker.USER,
            text=(
                "I'm a career coach focused on providing feedback to a "
                "company that sent me an offer."
            ),
            is_final=True,
            ts_start=10.0,
            ts_end=15.0,
        )
    )
    await asyncio.sleep(0.05)
    assert processor.current_phase == Phase.PRACTICE

    await bus.aclose()
    await run_task


@pytest.mark.asyncio
@pytest.mark.parametrize("phrase", [
    "great, thanks",
    "great thanks",
    "okay, thanks",
    "that was helpful",
    "that's helpful",
    "that was great",
    "i think i'm done",
    "let's stop there",
    "we can stop",
    "how did i do",
    "stop the scene",
])
async def test_natural_conversation_ender_triggers_feedback(
    phase_store: tuple[LocalFilesystemStore, str],
    phrase: str,
) -> None:
    """Natural closing phrases spoken during PRACTICE must transition to FEEDBACK."""
    store, session_id = phase_store
    bus = FrameBus(session_id)
    clock = FakeClock(datetime(2026, 5, 6, 23, 0, tzinfo=UTC))
    processor = PhaseProcessor(
        session_id,
        store,
        bus,
        budgets=PhaseBudgets(intake_min_dwell_seconds=5, practice_min_dwell_seconds=10),
        clock=clock,
        consent_getter=lambda: ConsentState.GRANTED,
    )
    await processor.bootstrap()
    run_task = asyncio.create_task(processor.run(bus.subscribe()))
    await asyncio.sleep(0)

    # Drive intake → practice: need >=2 user turns + min dwell elapsed
    clock.now += timedelta(seconds=10)
    await bus.publish(TranscriptDelta(
        session_id=session_id, utterance_id="u1", speaker=Speaker.USER,
        text="I need to rehearse a difficult conversation.", is_final=True,
        ts_start=0.0, ts_end=1.0,
    ))
    await asyncio.sleep(0)
    await bus.publish(TranscriptDelta(
        session_id=session_id, utterance_id="u2", speaker=Speaker.USER,
        text="let's practice", is_final=True, ts_start=1.0, ts_end=2.0,
    ))
    await asyncio.sleep(0.05)
    assert processor.current_phase == Phase.PRACTICE

    # User says the natural ender after practice min dwell
    clock.now += timedelta(seconds=15)
    await bus.publish(TranscriptDelta(
        session_id=session_id, utterance_id="u3", speaker=Speaker.USER,
        text=phrase, is_final=True, ts_start=15.0, ts_end=16.0,
    ))
    await asyncio.sleep(0.05)

    assert processor.current_phase == Phase.FEEDBACK, (
        f"Expected FEEDBACK after '{phrase}' but stayed in PRACTICE"
    )

    await bus.aclose()
    await run_task


@pytest.mark.asyncio
async def test_cue_blocked_before_min_dwell_elapsed(
    phase_store: tuple[LocalFilesystemStore, str],
) -> None:
    """A clear practice cue should not fire if intake just started."""
    store, session_id = phase_store
    bus = FrameBus(session_id)
    clock = FakeClock(datetime(2026, 5, 6, 23, 0, tzinfo=UTC))
    processor = PhaseProcessor(
        session_id,
        store,
        bus,
        budgets=PhaseBudgets(intake_min_dwell_seconds=30, practice_min_dwell_seconds=90),
        clock=clock,
        consent_getter=lambda: ConsentState.GRANTED,
    )
    await processor.bootstrap()
    run_task = asyncio.create_task(processor.run(bus.subscribe()))
    await asyncio.sleep(0)

    # Two user turns within 5 seconds, including an explicit ready cue.
    await bus.publish(
        TranscriptDelta(
            session_id=session_id,
            utterance_id="u1",
            speaker=Speaker.USER,
            text="I have a quick situation.",
            is_final=True,
            ts_start=0.0,
            ts_end=1.0,
        )
    )
    clock.now += timedelta(seconds=5)
    await bus.publish(
        TranscriptDelta(
            session_id=session_id,
            utterance_id="u2",
            speaker=Speaker.USER,
            text="I'm ready, let's practice.",
            is_final=True,
            ts_start=2.0,
            ts_end=3.0,
        )
    )
    await asyncio.sleep(0)
    assert processor.current_phase == Phase.INTAKE

    await bus.aclose()
    await run_task
