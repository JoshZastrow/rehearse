"""Tests for natural gender preference elicitation in intake."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rehearse.phases.intake import IntakeProcessor, classify_gender_preference
from rehearse.personas import COACH_PROMPT


# ---------------------------------------------------------------------------
# Prompt contains natural elicitation guidance
# ---------------------------------------------------------------------------


def test_coach_prompt_mentions_gender_elicitation() -> None:
    assert "prefer" in COACH_PROMPT.lower() or "gender" in COACH_PROMPT.lower() or "man or woman" in COACH_PROMPT.lower()


def test_coach_prompt_does_not_use_robotic_phrasing() -> None:
    robotic = ["select either", "choose between", "option 1", "option 2", "input your"]
    for phrase in robotic:
        assert phrase not in COACH_PROMPT.lower(), f"Found robotic phrasing: {phrase!r}"


# ---------------------------------------------------------------------------
# Gender classifier
# ---------------------------------------------------------------------------


def test_classifier_detects_female_preference_explicit() -> None:
    assert classify_gender_preference("I prefer talking to women about these things") == "female"


def test_classifier_detects_female_preference_natural() -> None:
    assert classify_gender_preference("honestly I find it easier with women") == "female"


def test_classifier_detects_female_preference_short() -> None:
    assert classify_gender_preference("a woman") == "female"


def test_classifier_detects_male_preference_explicit() -> None:
    assert classify_gender_preference("I tend to connect better with men on work stuff") == "male"


def test_classifier_detects_male_preference_natural() -> None:
    assert classify_gender_preference("probably a man, it feels more realistic for this") == "male"


def test_classifier_detects_male_preference_short() -> None:
    assert classify_gender_preference("male please") == "male"


def test_classifier_returns_none_for_no_preference() -> None:
    assert classify_gender_preference("I don't really care either way") is None


def test_classifier_returns_none_for_ambiguous_sentence() -> None:
    # "his" is used about a third party, not expressing preference
    assert classify_gender_preference("my manager said his decision was final") is None


def test_classifier_returns_none_for_empty() -> None:
    assert classify_gender_preference("") is None


def test_classifier_returns_none_when_both_genders_mentioned_no_preference() -> None:
    # Ambiguous — mentions both, no clear preference
    assert classify_gender_preference("could be a man or a woman, either is fine") is None


def test_classifier_handles_gender_nonbinary_gracefully() -> None:
    # Should not crash; returns None if not clearly male/female
    result = classify_gender_preference("I'd prefer non-binary actually")
    assert result is None


# ---------------------------------------------------------------------------
# IntakeProcessor captures gender from transcript
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_intake_processor_captures_female_preference(tmp_path) -> None:
    """When a user says 'I prefer women' in intake, IntakeRecord.gender_preference == 'female'."""
    from rehearse.bus import FrameBus
    from rehearse.frames import TranscriptDelta
    from rehearse.storage import LocalFilesystemStore
    from rehearse.types import ConsentState, Phase, Session, Speaker

    session_id = "test-gender-intake"
    store = LocalFilesystemStore(root=tmp_path, public_base_url="http://test")
    session = Session(created_at=datetime(2026, 1, 1, tzinfo=UTC), consent=ConsentState.GRANTED)
    await store.write(session_id, "session.json", session.model_dump_json())

    bus = FrameBus(session_id)
    processor = IntakeProcessor(
        session_id,
        store,
        phase_getter=lambda: Phase.INTAKE,
        bus=bus,
    )

    import asyncio
    task = asyncio.create_task(processor.run(bus.subscribe()))
    await asyncio.sleep(0)  # yield so the task registers its subscription

    ts = datetime(2026, 1, 1, tzinfo=UTC).timestamp()

    # User explains the situation
    await bus.publish(TranscriptDelta(
        session_id=session_id, utterance_id="u1",
        ts_start=ts, ts_end=ts, speaker=Speaker.USER,
        text="I need to talk to my manager about a raise.", is_final=True,
    ))

    # Coach asks about gender preference (natural phrasing)
    await bus.publish(TranscriptDelta(
        session_id=session_id, utterance_id="c1",
        ts_start=ts, ts_end=ts, speaker=Speaker.GUIDE,
        text="Do you tend to find it easier talking through this kind of thing with men or women?",
        is_final=True,
    ))

    # User expresses preference
    await bus.publish(TranscriptDelta(
        session_id=session_id, utterance_id="u2",
        ts_start=ts, ts_end=ts, speaker=Speaker.USER,
        text="Honestly I feel more comfortable with women.", is_final=True,
    ))

    await bus.aclose()
    await task

    updated = await store.read(session_id, "session.json")
    from rehearse.types import Session as S
    loaded = S.model_validate_json(updated)
    assert loaded.intake is not None
    assert loaded.intake.gender_preference == "female"


@pytest.mark.asyncio
async def test_intake_processor_no_gender_preference_when_not_asked(tmp_path) -> None:
    """When no gender question is asked, gender_preference stays None."""
    from rehearse.bus import FrameBus
    from rehearse.frames import TranscriptDelta
    from rehearse.storage import LocalFilesystemStore
    from rehearse.types import ConsentState, Phase, Session, Speaker

    session_id = "test-no-gender"
    store = LocalFilesystemStore(root=tmp_path, public_base_url="http://test")
    session = Session(created_at=datetime(2026, 1, 1, tzinfo=UTC), consent=ConsentState.GRANTED)
    await store.write(session_id, "session.json", session.model_dump_json())

    bus = FrameBus(session_id)
    processor = IntakeProcessor(session_id, store, phase_getter=lambda: Phase.INTAKE, bus=bus)

    import asyncio
    task = asyncio.create_task(processor.run(bus.subscribe()))
    await asyncio.sleep(0)  # yield so the task registers its subscription
    ts = datetime(2026, 1, 1, tzinfo=UTC).timestamp()

    await bus.publish(TranscriptDelta(
        session_id=session_id, utterance_id="u1", ts_start=ts, ts_end=ts,
        speaker=Speaker.USER, text="I need to talk to my manager.", is_final=True,
    ))
    await bus.publish(TranscriptDelta(
        session_id=session_id, utterance_id="u2", ts_start=ts, ts_end=ts,
        speaker=Speaker.USER, text="He keeps dismissing my ideas.", is_final=True,
    ))

    await bus.aclose()
    await task

    updated = await store.read(session_id, "session.json")
    from rehearse.types import Session as S
    loaded = S.model_validate_json(updated)
    # "He" in a non-preference context should not trigger gender detection
    assert loaded.intake is None or loaded.intake.gender_preference is None
