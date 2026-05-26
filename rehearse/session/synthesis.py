"""Generate post-call artifacts from the frozen session directory.

This file keeps post-call synthesis replayable. Given a stored session and its
artifacts, it writes `story.md` and `feedback.md` without depending on the live
Twilio or Hume call path.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from anthropic import AsyncAnthropic

from rehearse.storage import LocalFilesystemStore
from rehearse.types import ProsodyFrame, Session, TranscriptFrame


class SessionSynthesizer:
    """Create post-call story and feedback artifacts for one session."""

    def __init__(
        self,
        *,
        anthropic_api_key: str | None = None,
        anthropic_model: str | None = None,
    ) -> None:
        """Store the optional Anthropic settings used for richer synthesis."""
        self._anthropic_api_key = anthropic_api_key
        self._anthropic_model = anthropic_model or "claude-sonnet-4-6"
        self._client: AsyncAnthropic | None = None

    async def synthesize(self, store: LocalFilesystemStore, session: Session) -> tuple[str, str]:
        """Return story and feedback markdown built from the frozen artifacts."""
        transcript = await _load_transcript(store, session)
        prosody = await _load_prosody(store, session)
        if self._anthropic_api_key:
            try:
                return await self._synthesize_with_anthropic(session, transcript, prosody)
            except Exception:
                pass
        return (
            _fallback_story(session, transcript),
            _fallback_feedback(session, transcript, prosody),
        )

    def _client_lazy(self) -> AsyncAnthropic:
        """Create the Anthropic client the first time synthesis needs it."""
        if self._client is None:
            self._client = AsyncAnthropic(api_key=self._anthropic_api_key)
        return self._client

    async def _synthesize_with_anthropic(
        self,
        session: Session,
        transcript: Sequence[TranscriptFrame],
        prosody: Sequence[ProsodyFrame],
    ) -> tuple[str, str]:
        """Use Anthropic to generate both markdown artifacts in parallel."""
        client = self._client_lazy()
        story_prompt = _story_prompt(session, transcript)
        feedback_prompt = _feedback_prompt(session, transcript, prosody)
        story_resp = await client.messages.create(
            model=self._anthropic_model,
            max_tokens=700,
            temperature=0.2,
            system="Write concise markdown for a session story artifact.",
            messages=[{"role": "user", "content": story_prompt}],
        )
        feedback_resp = await client.messages.create(
            model=self._anthropic_model,
            max_tokens=900,
            temperature=0.2,
            system="Write grounded coaching feedback in markdown using only the supplied evidence.",
            messages=[{"role": "user", "content": feedback_prompt}],
        )
        return _response_text(story_resp), _response_text(feedback_resp)


async def persist_synthesis(
    store: LocalFilesystemStore,
    session: Session,
    synthesizer: SessionSynthesizer,
) -> Session:
    """Write `story.md` and `feedback.md`, then update the session manifest."""
    story, feedback = await synthesizer.synthesize(store, session)
    await store.write(session.id, "story.md", story)
    await store.write(session.id, "feedback.md", feedback)
    return await store.update_session(
        session.id,
        lambda current: _attach_synthesis_artifacts(
            current,
            story_name="story.md",
            feedback_name="feedback.md",
        ),
    )


def _attach_synthesis_artifacts(
    session: Session,
    *,
    story_name: str,
    feedback_name: str,
) -> Session:
    """Record the synthesized artifact filenames on the session manifest."""
    session.artifact_paths["story"] = story_name
    session.artifact_paths["feedback"] = feedback_name
    return session


async def _load_transcript(
    store: LocalFilesystemStore,
    session: Session,
) -> list[TranscriptFrame]:
    """Read transcript frames from disk, or return an empty list if absent."""
    path = session.artifact_paths.get("transcript")
    if not path:
        return []
    raw = (await store.read(session.id, path)).decode("utf-8").strip()
    if not raw:
        return []
    return [TranscriptFrame.model_validate_json(line) for line in raw.splitlines()]


async def _load_prosody(
    store: LocalFilesystemStore,
    session: Session,
) -> list[ProsodyFrame]:
    """Read prosody frames from disk, or return an empty list if absent."""
    path = session.artifact_paths.get("prosody")
    if not path:
        return []
    raw = (await store.read(session.id, path)).decode("utf-8").strip()
    if not raw:
        return []
    return [ProsodyFrame.model_validate_json(line) for line in raw.splitlines()]


def _fallback_story(session: Session, transcript: Sequence[TranscriptFrame]) -> str:
    """Return a deterministic story artifact when no LLM call is available."""
    opener = transcript[0].text if transcript else "The user completed a rehearsal call."
    closer = transcript[-1].text if transcript else "No closing transcript was captured."
    return "\n".join(
        [
            "# Story",
            "",
            f"Session `{session.id}` captured a live rehearsal conversation.",
            "",
            f"Opening moment: {opener}",
            f"Closing moment: {closer}",
        ]
    )


def _fallback_feedback(
    session: Session,
    transcript: Sequence[TranscriptFrame],
    prosody: Sequence[ProsodyFrame],
) -> str:
    """Return deterministic feedback that cites captured transcript evidence."""
    user_turns = [frame for frame in transcript if frame.speaker.value == "user"]
    coach_turns = [frame for frame in transcript if frame.speaker.value != "user"]
    average_arousal = 0.0
    if prosody:
        average_arousal = sum(frame.scores.arousal for frame in prosody) / len(prosody)
    evidence_line = user_turns[0].text if user_turns else "No user transcript was captured."
    next_step = coach_turns[-1].text if coach_turns else "No coaching reply was captured."
    return "\n".join(
        [
            "# Feedback",
            "",
            f"Session `{session.id}` ended with status `{session.completion_status}`.",
            "",
            "## Evidence",
            f"- Transcript evidence: {evidence_line}",
            f"- Average measured arousal: {average_arousal:.2f}",
            "",
            "## Coaching take",
            "You are strongest when you state the point directly and keep the ask concrete.",
            "",
            "## Next line to try",
            next_step,
        ]
    )


def _story_prompt(session: Session, transcript: Sequence[TranscriptFrame]) -> str:
    """Build the user prompt used for LLM-backed story synthesis."""
    transcript_block = "\n".join(
        f"{frame.speaker.value}: {frame.text}" for frame in transcript[:12]
    )
    return (
        "Write a short markdown story artifact for this rehearsal session.\n\n"
        f"Session ID: {session.id}\n"
        f"Completion status: {session.completion_status}\n"
        "Transcript excerpt:\n"
        f"{transcript_block or 'No transcript captured.'}"
    )


def _feedback_prompt(
    session: Session,
    transcript: Sequence[TranscriptFrame],
    prosody: Sequence[ProsodyFrame],
) -> str:
    """Build the user prompt used for LLM-backed feedback synthesis."""
    transcript_block = "\n".join(
        f"{frame.speaker.value}: {frame.text}" for frame in transcript[:20]
    )
    prosody_block = "\n".join(
        json.dumps(
            {
                "speaker": frame.speaker.value,
                "utterance_id": frame.utterance_id,
                "arousal": frame.scores.arousal,
                "valence": frame.scores.valence,
                "emotions": frame.scores.emotions,
            }
        )
        for frame in prosody[:10]
    )
    return (
        "Write grounded markdown coaching feedback using only the evidence below.\n\n"
        f"Session ID: {session.id}\n"
        f"Completion status: {session.completion_status}\n"
        "Transcript evidence:\n"
        f"{transcript_block or 'No transcript captured.'}\n\n"
        "Prosody evidence:\n"
        f"{prosody_block or 'No prosody captured.'}"
    )


def _response_text(response: object) -> str:
    """Extract concatenated text blocks from an Anthropic response object."""
    return "".join(
        block.text
        for block in getattr(response, "content", [])
        if getattr(block, "type", None) == "text"
    )
