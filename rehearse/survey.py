"""Post-call survey agent for the SURVEY phase.

Runs after FEEDBACK, asks the caller how the call went, captures the response,
writes survey.json, and back-fills OutcomeLabel so the training pipeline is
unaffected.

Spec: docs/specs/v2026-05-13-survey-agent.md §7
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import structlog

from rehearse.frames import EndOfCall, Frame, PhaseSignal, TranscriptDelta
from rehearse.outcome import build_label, classify_outcome
from rehearse.participants import SpeakRequest, VoiceSpeaker
from rehearse.session import utcnow
from rehearse.storage import LocalFilesystemStore
from rehearse.types import (
    OutcomeLabel,
    Phase,
    RubricDimension,
    Session,
    Speaker,
    SurveyQuestion,
    SurveyRecord,
    SurveyResponse,
)

log = structlog.get_logger(__name__)

SURVEY_QUESTION = (
    "Before we wrap up — how did the rehearsal feel overall? "
    "Was it useful?"
)

SURVEY_CLOSING_LINE = "Great — enjoy the conversation. Good luck!"

_OUTCOME_QUESTION = SurveyQuestion(
    text=SURVEY_QUESTION,
    response_type="binary",
    rubric_dimension=RubricDimension.USEFULNESS_HOLISTIC,
)

# ── scale word map ──────────────────────────────────────────────────────────
_SCALE_WORDS: dict[str, int] = {
    "one": 1, "terrible": 1, "bad": 1, "awful": 1,
    "two": 2, "poor": 2,
    "three": 3, "okay": 3, "ok": 3, "alright": 3, "average": 3,
    "four": 4, "good": 4, "well": 4,
    "five": 5, "great": 5, "excellent": 5, "amazing": 5,
    "perfect": 5, "fantastic": 5, "wonderful": 5,
}

_DIGIT_RE = re.compile(r"\b([1-5])\b")


def classify_scale(text: str) -> int | None:
    """Return 1-5 from a scale response, or None if unparseable."""
    norm = text.strip().lower().rstrip(".!?,")
    if not norm:
        return None
    m = _DIGIT_RE.search(norm)
    if m:
        return int(m.group(1))
    for word, value in _SCALE_WORDS.items():
        if word in norm.split() or norm == word:
            return value
    return None


# ── config ──────────────────────────────────────────────────────────────────

@dataclass
class SurveyAgentConfig:
    response_timeout_seconds: int = 12
    reprompt_limit: int = 1
    reprompt_text: str = "Sorry — was that a yes or a no?"


# ── agent ───────────────────────────────────────────────────────────────────

class SurveyAgent:
    """Ask the post-call survey question, capture the response, persist results.

    Lifecycle (spec §7.2):
      1. Idle until PhaseSignal(to_phase=SURVEY) arrives.
      2. Speak the handoff / survey question.
      3. Classify the next final USER transcript frame.
      4. positive/negative → persist SurveyRecord + OutcomeLabel.
      5. unclear → reprompt up to reprompt_limit, then mark partial.
      6. timeout / hangup → mark partial/skipped.
    """

    def __init__(
        self,
        session_id: str,
        store: LocalFilesystemStore,
        *,
        speaker: VoiceSpeaker,
        config: SurveyAgentConfig | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._session_id = session_id
        self._store = store
        self._speaker = speaker
        self._config = config or SurveyAgentConfig()
        self._clock = clock

        self._survey_started = False
        self._asked_at: datetime | None = None
        self._captured = False
        self._reprompts = 0
        self._responses: list[SurveyResponse] = []
        self._started_at: datetime | None = None
        self._timeout_task: asyncio.Task[None] | None = None

    async def run(self, frames: AsyncIterator[Frame]) -> None:
        """Drive the survey lifecycle from a frame-bus subscription."""
        try:
            async for frame in frames:
                if (
                    isinstance(frame, PhaseSignal)
                    and frame.to_phase == Phase.SURVEY
                    and not self._survey_started
                ):
                    self._survey_started = True
                    self._started_at = self._clock()
                    await self._begin_survey()
                elif (
                    isinstance(frame, TranscriptDelta)
                    and frame.speaker == Speaker.USER
                    and frame.is_final
                    and self._asked_at is not None
                    and not self._captured
                ):
                    await self._handle_response(frame.text)
                elif isinstance(frame, EndOfCall):
                    break
        finally:
            await self._shutdown()

    async def _begin_survey(self) -> None:
        """Set survey status and ask the main question."""
        await self._store.update_session(
            self._session_id,
            lambda s: _set_survey_status(s, "in_progress"),
        )
        log.info("survey.phase.started", session_id=self._session_id)
        await self._ask(_OUTCOME_QUESTION.text)

    async def _ask(self, text: str) -> None:
        """Speak a question, record asked_at, and arm the response timeout."""
        if self._captured:
            return
        await self._speaker.say(SpeakRequest(text=text))
        self._asked_at = self._clock()
        log.info(
            "survey.question.asked",
            session_id=self._session_id,
            rubric_dimension=_OUTCOME_QUESTION.rubric_dimension,
            response_type=_OUTCOME_QUESTION.response_type,
        )
        if self._timeout_task is not None:
            self._timeout_task.cancel()
        self._timeout_task = asyncio.create_task(self._fire_timeout())

    async def _fire_timeout(self) -> None:
        try:
            await asyncio.sleep(self._config.response_timeout_seconds)
        except asyncio.CancelledError:
            return
        if not self._captured:
            await self._finish(captured=False, verbatim=None)

    async def _handle_response(self, text: str) -> None:
        classification = classify_outcome(text)
        log.info(
            "survey.response.classified",
            session_id=self._session_id,
            response_type="binary",
            captured=(classification != "unclear"),
        )
        if classification == "unclear":
            if self._reprompts < self._config.reprompt_limit:
                self._reprompts += 1
                log.info("survey.response.unclear", session_id=self._session_id, attempt=self._reprompts)
                await self._ask(self._config.reprompt_text)
                return
            await self._finish(captured=False, verbatim=text)
            return
        await self._finish(captured=True, verbatim=text, classification=classification)

    async def _finish(
        self,
        *,
        captured: bool,
        verbatim: str | None,
        classification: Literal["positive", "negative", "unclear"] | None = None,
    ) -> None:
        """Persist survey results and back-fill OutcomeLabel."""
        if self._timeout_task is not None:
            self._timeout_task.cancel()

        self._captured = True
        now = self._clock()
        value: bool | None = None
        if captured and classification in ("positive", "negative"):
            value = classification == "positive"

        response = SurveyResponse(
            question_text=_OUTCOME_QUESTION.text,
            response_type="binary",
            rubric_dimension=_OUTCOME_QUESTION.rubric_dimension,
            captured=captured,
            value=value,
            verbatim=verbatim,
            captured_at=now if captured else None,
        )
        self._responses.append(response)

        record = SurveyRecord(
            session_id=self._session_id,
            generation_method="fallback",
            questions=[_OUTCOME_QUESTION],
            responses=self._responses,
            started_at=self._started_at or now,
            completed_at=now,
        )
        await self._write_survey_json(record)

        outcome_label: OutcomeLabel | None = None
        if captured and value is not None:
            outcome_label = build_label(
                "positive" if value else "negative",
                verbatim or "",
                captured_at=now,
            )

        final_status = "complete" if captured else "partial"
        await self._store.update_session(
            self._session_id,
            lambda s: _apply_survey_result(s, outcome_label, final_status),
        )
        log.info(
            "survey.complete" if captured else "survey.partial",
            session_id=self._session_id,
            captured=captured,
        )

        if captured:
            await self._speaker.say(SpeakRequest(text=SURVEY_CLOSING_LINE))

    async def _write_survey_json(self, record: SurveyRecord) -> None:
        """Write survey.json and register its path in the session manifest."""
        content = record.model_dump_json(indent=2)
        await self._store.write(self._session_id, "survey.json", content)
        await self._store.update_session(
            self._session_id,
            lambda s: _register_artifact(s, "survey", "survey.json"),
        )

    async def _shutdown(self) -> None:
        """Cancel pending timers; write a partial record if never captured."""
        if self._timeout_task is not None and not self._timeout_task.done():
            self._timeout_task.cancel()

        if not self._survey_started:
            # SURVEY phase never began (e.g. consent declined)
            return

        if not self._captured:
            # Hangup before we finished — write whatever we have
            now = self._clock()
            record = SurveyRecord(
                session_id=self._session_id,
                generation_method="fallback",
                questions=[_OUTCOME_QUESTION],
                responses=self._responses,
                started_at=self._started_at or now,
                completed_at=now,
            )
            await self._write_survey_json(record)
            await self._store.update_session(
                self._session_id,
                lambda s: _apply_survey_result(s, None, "partial"),
            )
            log.info("survey.partial", session_id=self._session_id, reason="hangup")


# ── session mutators ─────────────────────────────────────────────────────────

def _set_survey_status(
    session: Session,
    status: Literal["generating", "in_progress", "complete", "partial", "skipped"],
) -> Session:
    session.survey_status = status
    return session


def _apply_survey_result(
    session: Session,
    outcome_label: OutcomeLabel | None,
    status: Literal["complete", "partial", "skipped"],
) -> Session:
    session.survey_status = status
    if outcome_label is not None and session.outcome_label is None:
        session.outcome_label = outcome_label
        session.outcome_probe_status = "captured"
    elif outcome_label is None and session.outcome_probe_status not in ("captured",):
        session.outcome_probe_status = "skipped"
    return session


def _register_artifact(session: Session, key: str, relative_path: str) -> Session:
    session.artifact_paths[key] = relative_path
    return session


__all__ = [
    "SURVEY_CLOSING_LINE",
    "SURVEY_QUESTION",
    "SurveyAgent",
    "SurveyAgentConfig",
    "classify_scale",
]
