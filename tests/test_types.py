"""Round-trip tests for pydantic contracts.

Establishes that every domain, eval, training, and telemetry type serializes
and deserializes without loss. These are the cheapest tests that assert the
schema exists and is self-consistent; they will run before any application
logic lands.
"""

from datetime import UTC, datetime

from rehearse.types import Session


def test_session_default_persona_key():
    session = Session(created_at=datetime.now(UTC))
    assert session.persona_key == "default"


def test_session_accepts_explicit_persona_key():
    session = Session(created_at=datetime.now(UTC), persona_key="relationship_coach")
    assert session.persona_key == "relationship_coach"


def test_prosody_source_has_local_classifier():
    from rehearse.types import ProsodySource
    assert ProsodySource.LOCAL_CLASSIFIER == "local_classifier"


def test_session_has_backend_fields():
    s = Session(created_at=datetime.now(UTC))
    assert s.backend_type is None
    assert s.speech_services is None
    assert s.prosody_source == "none"


def test_speaker_provider_caller_are_aliases_of_coach_user():
    from rehearse.types import Speaker

    assert Speaker.PROVIDER is Speaker.COACH
    assert Speaker.CALLER is Speaker.USER
    # canonical serialized values are unchanged (backward compatible on disk)
    assert Speaker.PROVIDER.value == "coach"
    assert Speaker.CALLER.value == "user"
    # aliases are excluded from iteration
    assert list(Speaker) == [Speaker.USER, Speaker.COACH, Speaker.CHARACTER]


def test_speaker_parses_both_legacy_and_current_labels():
    from rehearse.types import Speaker

    assert Speaker("user") is Speaker.USER
    assert Speaker("coach") is Speaker.COACH
    assert Speaker("caller") is Speaker.USER
    assert Speaker("provider") is Speaker.COACH
    assert Speaker("PROVIDER") is Speaker.COACH  # current labels are case-insensitive


def test_speaker_round_trips_through_pydantic_for_current_labels():
    from rehearse.types import Phase, Speaker, TranscriptFrame

    frame = TranscriptFrame(
        session_id="s",
        utterance_id="u",
        ts_start=0.0,
        ts_end=0.1,
        speaker="provider",  # current label accepted via _missing_
        phase=Phase.INTAKE,
        text="hi",
        is_interim=False,
    )
    assert frame.speaker is Speaker.COACH
    # serializes back to the canonical "coach"
    assert '"speaker":"coach"' in frame.model_dump_json()
