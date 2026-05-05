"""Test deterministic outcome classification, label building, and probe lifecycle."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rehearse.bus import FrameBus
from rehearse.frames import EndOfCall, PhaseSignal, TranscriptDelta
from rehearse.outcome import (
    OUTCOME_PROMPT,
    OutcomeProbe,
    OutcomeProbeConfig,
    build_label,
    classify_outcome,
)
from rehearse.storage import LocalFilesystemStore
from rehearse.types import ConsentState, Phase, Session, Speaker

# ---------------------------------------------------------------------------
# Pure classifier and label-builder tests (spec §6.2 / §6.3)
# ---------------------------------------------------------------------------

POSITIVE_FIXTURE = [
    "yes",
    "Yeah.",
    "yep",
    "absolutely",
    "definitely, that helped",
    "yes, the part about pacing was useful",
    "it did help me see it differently",
    "useful, especially the framing",
    "great",
    "for sure, I think so",
]

NEGATIVE_FIXTURE = [
    "no",
    "Nope.",
    "nah",
    "not really",
    "not useful",
    "no, it didn't land for me",
    "didn't help",
    "did not help, sorry",
]

UNCLEAR_FIXTURE = [
    "maybe",
    "I don't know",
    "kind of, but also kind of not",
    "let me think about it",
    "",
    "   ",
]


@pytest.mark.parametrize("text", POSITIVE_FIXTURE)
def test_classify_outcome_positive(text: str) -> None:
    assert classify_outcome(text) == "positive"


@pytest.mark.parametrize("text", NEGATIVE_FIXTURE)
def test_classify_outcome_negative(text: str) -> None:
    assert classify_outcome(text) == "negative"


@pytest.mark.parametrize("text", UNCLEAR_FIXTURE)
def test_classify_outcome_unclear(text: str) -> None:
    assert classify_outcome(text) == "unclear"


def test_build_label_bare_yes_omits_notes() -> None:
    label = build_label("positive", "yes", captured_at=datetime.now(UTC))
    assert label is not None
    assert label.did_it_help is True
    assert label.notes is None


def test_build_label_bare_no_omits_notes() -> None:
    label = build_label("negative", "No.", captured_at=datetime.now(UTC))
    assert label is not None
    assert label.did_it_help is False
    assert label.notes is None


def test_build_label_keeps_verbatim_notes() -> None:
    label = build_label(
        "positive",
        "yes, the part about pacing was useful",
        captured_at=datetime.now(UTC),
    )
    assert label is not None
    assert label.did_it_help is True
    assert label.notes == "yes, the part about pacing was useful"


def test_build_label_unclear_returns_none() -> None:
    assert build_label("unclear", "maybe", captured_at=datetime.now(UTC)) is None


# ---------------------------------------------------------------------------
# OutcomeProbe lifecycle tests
# ---------------------------------------------------------------------------


@pytest.fixture
def probe_store(tmp_path: Path) -> tuple[LocalFilesystemStore, str]:
    store = LocalFilesystemStore(tmp_path, "https://example.test")
    session = Session(created_at=datetime.now(UTC), consent=ConsentState.GRANTED)
    (store.session_dir(session.id) / "session.json").write_text(
        session.model_dump_json(indent=2)
    )
    return store, session.id


def _read_manifest(store: LocalFilesystemStore, session_id: str) -> dict:
    path = store.session_dir(session_id) / "session.json"
    return json.loads(path.read_text())


def _user_frame(session_id: str, text: str, *, ts: float = 1.0) -> TranscriptDelta:
    return TranscriptDelta(
        session_id=session_id,
        utterance_id=f"u-{ts}",
        speaker=Speaker.USER,
        text=text,
        is_final=True,
        ts_start=ts,
        ts_end=ts + 0.1,
    )


@pytest.mark.asyncio
async def test_probe_captures_positive_label(
    probe_store: tuple[LocalFilesystemStore, str],
) -> None:
    store, session_id = probe_store
    bus = FrameBus(session_id)
    spoken: list[str] = []

    async def speak(text: str) -> None:
        spoken.append(text)

    probe = OutcomeProbe(
        session_id,
        store,
        speak=speak,
        config=OutcomeProbeConfig(
            response_timeout_seconds=2,
            reprompt_limit=1,
            prompt_lead_seconds=10,
            feedback_budget_seconds=10,  # delay = 0
        ),
    )
    task = asyncio.create_task(probe.run(bus.subscribe()))
    await asyncio.sleep(0)
    await bus.publish(
        PhaseSignal(
            session_id=session_id,
            from_phase=Phase.PRACTICE,
            to_phase=Phase.FEEDBACK,
            reason="cue",
            ts=0.0,
        )
    )
    # Give the delay-task a tick to fire (delay = 0)
    for _ in range(5):
        await asyncio.sleep(0)
    assert spoken == [OUTCOME_PROMPT]
    await bus.publish(_user_frame(session_id, "yes, the pacing tip was useful"))
    await asyncio.sleep(0)
    await bus.aclose()
    await task

    manifest = _read_manifest(store, session_id)
    assert manifest["outcome_probe_status"] == "captured"
    assert manifest["outcome_label"]["did_it_help"] is True
    assert manifest["outcome_label"]["notes"] == "yes, the pacing tip was useful"


@pytest.mark.asyncio
async def test_probe_skips_on_timeout(
    probe_store: tuple[LocalFilesystemStore, str],
) -> None:
    store, session_id = probe_store
    bus = FrameBus(session_id)

    async def speak(_text: str) -> None:
        return None

    probe = OutcomeProbe(
        session_id,
        store,
        speak=speak,
        config=OutcomeProbeConfig(
            response_timeout_seconds=0,  # instant timeout
            reprompt_limit=1,
            prompt_lead_seconds=10,
            feedback_budget_seconds=10,
        ),
    )
    task = asyncio.create_task(probe.run(bus.subscribe()))
    await asyncio.sleep(0)
    await bus.publish(
        PhaseSignal(
            session_id=session_id,
            from_phase=Phase.PRACTICE,
            to_phase=Phase.FEEDBACK,
            reason="cue",
            ts=0.0,
        )
    )
    # Let the prompt fire and the zero-second timeout elapse
    for _ in range(10):
        await asyncio.sleep(0)
    await bus.aclose()
    await task

    manifest = _read_manifest(store, session_id)
    assert manifest["outcome_probe_status"] == "skipped"
    assert manifest["outcome_label"] is None


@pytest.mark.asyncio
async def test_probe_reprompts_then_skips_on_unclear(
    probe_store: tuple[LocalFilesystemStore, str],
) -> None:
    store, session_id = probe_store
    bus = FrameBus(session_id)
    spoken: list[str] = []

    async def speak(text: str) -> None:
        spoken.append(text)

    probe = OutcomeProbe(
        session_id,
        store,
        speak=speak,
        config=OutcomeProbeConfig(
            response_timeout_seconds=5,
            reprompt_limit=1,
            prompt_lead_seconds=10,
            feedback_budget_seconds=10,
        ),
    )
    task = asyncio.create_task(probe.run(bus.subscribe()))
    await asyncio.sleep(0)
    await bus.publish(
        PhaseSignal(
            session_id=session_id,
            from_phase=Phase.PRACTICE,
            to_phase=Phase.FEEDBACK,
            reason="cue",
            ts=0.0,
        )
    )
    for _ in range(5):
        await asyncio.sleep(0)
    await bus.publish(_user_frame(session_id, "maybe", ts=1.0))
    await asyncio.sleep(0)
    await bus.publish(_user_frame(session_id, "I don't know", ts=2.0))
    await asyncio.sleep(0)
    await bus.aclose()
    await task

    assert len(spoken) == 2  # initial prompt + reprompt
    manifest = _read_manifest(store, session_id)
    assert manifest["outcome_probe_status"] == "skipped"
    assert manifest["outcome_label"] is None


@pytest.mark.asyncio
async def test_probe_is_idempotent_after_capture(
    probe_store: tuple[LocalFilesystemStore, str],
) -> None:
    store, session_id = probe_store
    bus = FrameBus(session_id)

    async def speak(_text: str) -> None:
        return None

    probe = OutcomeProbe(
        session_id,
        store,
        speak=speak,
        config=OutcomeProbeConfig(
            response_timeout_seconds=5,
            reprompt_limit=1,
            prompt_lead_seconds=10,
            feedback_budget_seconds=10,
        ),
    )
    task = asyncio.create_task(probe.run(bus.subscribe()))
    await asyncio.sleep(0)
    await bus.publish(
        PhaseSignal(
            session_id=session_id,
            from_phase=Phase.PRACTICE,
            to_phase=Phase.FEEDBACK,
            reason="cue",
            ts=0.0,
        )
    )
    for _ in range(5):
        await asyncio.sleep(0)
    await bus.publish(_user_frame(session_id, "yes", ts=1.0))
    await asyncio.sleep(0)
    await bus.publish(_user_frame(session_id, "no, actually", ts=2.0))
    await asyncio.sleep(0)
    await bus.aclose()
    await task

    manifest = _read_manifest(store, session_id)
    assert manifest["outcome_probe_status"] == "captured"
    assert manifest["outcome_label"]["did_it_help"] is True


@pytest.mark.asyncio
async def test_probe_marks_skipped_on_hangup_after_prompt(
    probe_store: tuple[LocalFilesystemStore, str],
) -> None:
    store, session_id = probe_store
    bus = FrameBus(session_id)

    async def speak(_text: str) -> None:
        return None

    probe = OutcomeProbe(
        session_id,
        store,
        speak=speak,
        config=OutcomeProbeConfig(
            response_timeout_seconds=30,
            reprompt_limit=1,
            prompt_lead_seconds=10,
            feedback_budget_seconds=10,
        ),
    )
    task = asyncio.create_task(probe.run(bus.subscribe()))
    await asyncio.sleep(0)
    await bus.publish(
        PhaseSignal(
            session_id=session_id,
            from_phase=Phase.PRACTICE,
            to_phase=Phase.FEEDBACK,
            reason="cue",
            ts=0.0,
        )
    )
    for _ in range(5):
        await asyncio.sleep(0)
    await bus.publish(EndOfCall(session_id=session_id, reason="hangup", ts=1.0))
    await bus.aclose()
    await task

    manifest = _read_manifest(store, session_id)
    assert manifest["outcome_probe_status"] == "skipped"
    assert manifest["outcome_label"] is None
