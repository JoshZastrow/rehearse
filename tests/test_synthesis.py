"""Post-call synthesis replayability tests.

Given a frozen session fixture, `SessionSynthesizer.synthesize` must produce
markdown artifacts whose evidence lines all resolve back to real utterances in
the fixture's transcript. Model calls are faked; these tests assert structure
and citation resolution, not generated content.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from rehearse.storage import LocalFilesystemStore
from rehearse.synthesis import SessionSynthesizer, persist_synthesis
from rehearse.types import (
    Phase,
    ProsodyFrame,
    ProsodyScores,
    ProsodySource,
    Session,
    Speaker,
    TranscriptFrame,
)


@pytest.fixture
def store(tmp_path: Path) -> LocalFilesystemStore:
    return LocalFilesystemStore(root=tmp_path, public_base_url="https://example.test")


async def _freeze_fixture(
    store: LocalFilesystemStore,
) -> tuple[Session, list[TranscriptFrame], list[ProsodyFrame]]:
    """Write a small frozen session to disk and return the in-memory copy."""
    session = Session(
        created_at=datetime(2026, 4, 28, tzinfo=UTC),
        completion_status="complete",
        artifact_paths={"transcript": "transcript.jsonl", "prosody": "prosody.jsonl"},
    )
    transcript = [
        TranscriptFrame(
            session_id=session.id,
            utterance_id="u1",
            ts_start=0.0,
            ts_end=1.5,
            speaker=Speaker.USER,
            phase=Phase.INTAKE,
            text="I want to ask my manager for a raise.",
        ),
        TranscriptFrame(
            session_id=session.id,
            utterance_id="u2",
            ts_start=1.6,
            ts_end=3.0,
            speaker=Speaker.COACH,
            phase=Phase.PRACTICE,
            text="Open with the impact you've had this quarter.",
        ),
        TranscriptFrame(
            session_id=session.id,
            utterance_id="u3",
            ts_start=3.1,
            ts_end=4.4,
            speaker=Speaker.USER,
            phase=Phase.PRACTICE,
            text="Last quarter I shipped two launches on time.",
        ),
    ]
    prosody = [
        ProsodyFrame(
            session_id=session.id,
            utterance_id="u1",
            ts_start=0.0,
            ts_end=1.5,
            speaker=Speaker.USER,
            source=ProsodySource.HUME_LIVE,
            scores=ProsodyScores(arousal=0.6, valence=0.1, emotions={"anxiety": 0.4}),
        ),
        ProsodyFrame(
            session_id=session.id,
            utterance_id="u3",
            ts_start=3.1,
            ts_end=4.4,
            speaker=Speaker.USER,
            source=ProsodySource.HUME_LIVE,
            scores=ProsodyScores(arousal=0.4, valence=0.3, emotions={"confidence": 0.5}),
        ),
    ]
    await store.write(session.id, "session.json", session.model_dump_json(indent=2))
    await store.write(
        session.id,
        "transcript.jsonl",
        "\n".join(frame.model_dump_json() for frame in transcript) + "\n",
    )
    await store.write(
        session.id,
        "prosody.jsonl",
        "\n".join(frame.model_dump_json() for frame in prosody) + "\n",
    )
    return session, transcript, prosody


@pytest.mark.asyncio
async def test_fallback_synthesis_produces_structured_artifacts(
    store: LocalFilesystemStore,
) -> None:
    session, _, _ = await _freeze_fixture(store)
    synthesizer = SessionSynthesizer()

    story, feedback = await synthesizer.synthesize(store, session)

    assert story.startswith("# Story")
    assert f"`{session.id}`" in story
    assert feedback.startswith("# Feedback")
    assert "## Evidence" in feedback
    assert "## Coaching take" in feedback
    assert "## Next line to try" in feedback


@pytest.mark.asyncio
async def test_fallback_feedback_citations_resolve_to_transcript(
    store: LocalFilesystemStore,
) -> None:
    session, transcript, prosody = await _freeze_fixture(store)
    synthesizer = SessionSynthesizer()

    _, feedback = await synthesizer.synthesize(store, session)

    transcript_texts = {frame.text for frame in transcript}
    user_evidence = next(frame.text for frame in transcript if frame.speaker == Speaker.USER)
    coach_next_step = next(
        frame.text for frame in reversed(transcript) if frame.speaker != Speaker.USER
    )
    assert user_evidence in feedback
    assert coach_next_step in feedback
    for line in feedback.splitlines():
        if line.startswith("- Transcript evidence: "):
            quoted = line.removeprefix("- Transcript evidence: ").strip()
            assert quoted in transcript_texts
    expected_arousal = sum(frame.scores.arousal for frame in prosody) / len(prosody)
    assert f"{expected_arousal:.2f}" in feedback


@pytest.mark.asyncio
async def test_synthesis_is_replayable(store: LocalFilesystemStore) -> None:
    session, _, _ = await _freeze_fixture(store)
    synthesizer = SessionSynthesizer()

    first = await synthesizer.synthesize(store, session)
    second = await synthesizer.synthesize(store, session)

    assert first == second


@pytest.mark.asyncio
async def test_persist_synthesis_writes_files_and_updates_manifest(
    store: LocalFilesystemStore,
) -> None:
    session, _, _ = await _freeze_fixture(store)
    synthesizer = SessionSynthesizer()

    updated = await persist_synthesis(store, session, synthesizer)

    story = (await store.read(session.id, "story.md")).decode("utf-8")
    feedback = (await store.read(session.id, "feedback.md")).decode("utf-8")
    assert story.startswith("# Story")
    assert feedback.startswith("# Feedback")
    assert updated.artifact_paths["story"] == "story.md"
    assert updated.artifact_paths["feedback"] == "feedback.md"


@pytest.mark.asyncio
async def test_synthesis_handles_missing_artifact_paths(store: LocalFilesystemStore) -> None:
    session = Session(
        created_at=datetime(2026, 4, 28, tzinfo=UTC),
        completion_status="partial",
    )
    await store.write(session.id, "session.json", session.model_dump_json(indent=2))
    synthesizer = SessionSynthesizer()

    story, feedback = await synthesizer.synthesize(store, session)

    assert "# Story" in story
    assert "# Feedback" in feedback
    assert "0.00" in feedback


@pytest.mark.asyncio
async def test_anthropic_path_uses_injected_client(store: LocalFilesystemStore) -> None:
    session, _, _ = await _freeze_fixture(store)

    class _Block:
        def __init__(self, text: str) -> None:
            self.type = "text"
            self.text = text

    class _Response:
        def __init__(self, text: str) -> None:
            self.content = [_Block(text)]

    class _Messages:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            label = "story" if "story" in kwargs["system"] else "feedback"
            return _Response(f"# {label.title()}\n\nFAKE-{label.upper()}")

    class _Client:
        def __init__(self) -> None:
            self.messages = _Messages()

    fake_client = _Client()
    synthesizer = SessionSynthesizer(anthropic_api_key="test-key")
    synthesizer._client = fake_client  # type: ignore[attr-defined]

    story, feedback = await synthesizer.synthesize(store, session)

    assert "FAKE-STORY" in story
    assert "FAKE-FEEDBACK" in feedback
    assert len(fake_client.messages.calls) == 2
    feedback_call = next(c for c in fake_client.messages.calls if "feedback" in c["system"])
    user_prompt = feedback_call["messages"][0]["content"]
    assert "I want to ask my manager for a raise." in user_prompt
    assert '"arousal": 0.6' in user_prompt
