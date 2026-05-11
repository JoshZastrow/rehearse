"""Capture the call-quality outcome inline at the end of the feedback phase.

The runtime owns one new component, `OutcomeProbe`, that asks the user a
single yes/no question on the line before hangup, classifies the response
with a deterministic parser (mirroring consent classification), and writes
an `OutcomeLabel` onto the session manifest.

Coach speaker channel: the probe is given an injected speaker so production
wiring can route the prompt through the active voice participant while tests
can pass a fake. The probe never depends on the LLM path.

Spec: docs/specs/v2026-05-01-consent-and-outcome-capture.md
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import structlog

from rehearse.frames import EndOfCall, Frame, PhaseSignal, TranscriptDelta
from rehearse.participants import SpeakRequest, VoiceSpeaker
from rehearse.session import utcnow
from rehearse.storage import LocalFilesystemStore
from rehearse.types import OutcomeLabel, Phase, Session, Speaker

log = structlog.get_logger(__name__)

OutcomeClassification = Literal["positive", "negative", "unclear"]

OUTCOME_PROMPT = (
    "Before we wrap up — did this rehearsal feel useful? "
    "You can say yes or no, or share a sentence."
)

OUTCOME_REPROMPT = "Sorry — was that a yes or a no?"

POSITIVE_PHRASES: frozenset[str] = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "yup",
        "sure",
        "absolutely",
        "definitely",
        "useful",
        "helpful",
        "great",
        "it did",
        "i think so",
        "totally",
        "for sure",
    }
)

NEGATIVE_PHRASES: frozenset[str] = frozenset(
    {
        "no",
        "nope",
        "nah",
        "not really",
        "not useful",
        "not helpful",
        "wasn't helpful",
        "didn't help",
        "did not help",
        "not at all",
    }
)


@dataclass
class OutcomeProbeConfig:
    """Configuration for the inline outcome probe."""

    response_timeout_seconds: int = 15
    reprompt_limit: int = 1
    prompt_lead_seconds: int = 15
    feedback_budget_seconds: int = 60


class OutcomeProbe:
    """Ask the outcome question, classify the answer, persist the label.

    Lifecycle (see spec §7.2):
      1. Wait for `PhaseSignal(to_phase=Phase.FEEDBACK)` on the bus.
      2. Sleep `feedback_budget - prompt_lead` seconds, then ask via the
         injected speaker.
      3. Classify the next final user transcript frame with
         `classify_outcome`.
      4. positive/negative -> persist `OutcomeLabel`, mark `"captured"`.
      5. unclear -> reprompt up to `reprompt_limit`, then mark `"skipped"`.
      6. timeout / hangup -> mark `"skipped"` without a label.
    """

    def __init__(
        self,
        session_id: str,
        store: LocalFilesystemStore,
        *,
        speaker: VoiceSpeaker,
        config: OutcomeProbeConfig | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        """Store identity, persistence, speaker channel, and timing knobs."""
        self._session_id = session_id
        self._store = store
        self._speaker = speaker
        self._config = config or OutcomeProbeConfig()
        self._clock = clock
        self._asked_at: datetime | None = None
        self._captured = False
        self._reprompts = 0
        self._delay_task: asyncio.Task[None] | None = None
        self._timeout_task: asyncio.Task[None] | None = None

    async def run(self, frames: AsyncIterator[Frame]) -> None:
        """Drive the probe lifecycle from a frame-bus subscription."""
        try:
            async for frame in frames:
                if (
                    isinstance(frame, PhaseSignal)
                    and frame.to_phase == Phase.FEEDBACK
                    and self._delay_task is None
                ):
                    self._delay_task = asyncio.create_task(self._fire_after_delay())
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

    async def _fire_after_delay(self) -> None:
        """Wait the configured lead-time, then ask the prompt."""
        delay = max(
            0,
            self._config.feedback_budget_seconds - self._config.prompt_lead_seconds,
        )
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        await self._ask(OUTCOME_PROMPT)

    async def _ask(self, prompt: str) -> None:
        """Speak a prompt, mark asked, schedule the response timeout."""
        if self._captured:
            return
        await self._speaker.say(SpeakRequest(text=prompt))
        self._asked_at = self._clock()
        await self._store.update_session(
            self._session_id,
            lambda s: _set_status(s, "asked"),
        )
        log.info(
            "outcome.prompt.spoken",
            session_id=self._session_id,
            lead_seconds=self._config.prompt_lead_seconds,
        )
        if self._timeout_task is not None:
            self._timeout_task.cancel()
        self._timeout_task = asyncio.create_task(self._fire_timeout())

    async def _fire_timeout(self) -> None:
        """Mark the probe skipped after the response timeout elapses."""
        try:
            await asyncio.sleep(self._config.response_timeout_seconds)
        except asyncio.CancelledError:
            return
        if not self._captured:
            await self._mark_skipped("timeout")

    async def _handle_response(self, text: str) -> None:
        """Classify a user transcript and either persist or re-prompt."""
        classification = classify_outcome(text)
        log.info(
            "outcome.classify",
            session_id=self._session_id,
            classification=classification,
            text_len=len(text),
        )
        if classification == "unclear":
            if self._reprompts < self._config.reprompt_limit:
                self._reprompts += 1
                log.info(
                    "outcome.reprompt",
                    session_id=self._session_id,
                    attempt=self._reprompts,
                )
                await self._ask(OUTCOME_REPROMPT)
                return
            await self._mark_skipped("unclear")
            return
        label = build_label(classification, text, captured_at=self._clock())
        if label is None:
            return
        await self._persist_label(label)

    async def _persist_label(self, label: OutcomeLabel) -> None:
        """Write the captured label to session.json and mark probe captured."""
        if self._captured:
            return
        self._captured = True
        await self._store.update_session(
            self._session_id,
            lambda s: _set_label(s, label),
        )
        latency_ms = (
            int((self._clock() - self._asked_at).total_seconds() * 1000)
            if self._asked_at is not None
            else 0
        )
        log.info(
            "outcome.captured",
            session_id=self._session_id,
            did_it_help=label.did_it_help,
            has_notes=label.notes is not None,
            latency_ms=latency_ms,
        )
        await self._store.append(
            self._session_id,
            "telemetry.jsonl",
            json.dumps(
                {
                    "event": "outcome.captured.latency_ms",
                    "value": latency_ms,
                    "ts": label.captured_at.isoformat(),
                }
            ),
        )

    async def _mark_skipped(self, reason: Literal["timeout", "unclear", "hangup"]) -> None:
        """Record that the probe ran but produced no usable label."""
        if self._captured:
            return
        await self._store.update_session(
            self._session_id,
            lambda s: _set_status(s, "skipped"),
        )
        log.info(
            "outcome.skipped",
            session_id=self._session_id,
            reason=reason,
        )

    async def _shutdown(self) -> None:
        """Cancel pending timers and finalize status if needed."""
        for task in (self._delay_task, self._timeout_task):
            if task is not None and not task.done():
                task.cancel()
        if self._asked_at is not None and not self._captured:
            await self._mark_skipped("hangup")

    @property
    def status(self) -> Literal["pending", "asked", "captured", "skipped"]:
        """Return the current probe status for inspection by the orchestrator."""
        if self._captured:
            return "captured"
        if self._asked_at is None:
            return "pending"
        return "asked"


def classify_outcome(text: str) -> OutcomeClassification:
    """Classify a free-form user response as positive, negative, or unclear.

    Mirrors the deterministic shape of consent classification: small lexicons
    of clear affirmatives/negatives, lowercased and stripped, with start-of-
    string matching. Misclassification toward 'unclear' is safe; a confident
    false capture is not.
    """
    norm = text.strip().lower().rstrip(".!?,")
    if not norm:
        return "unclear"
    for phrase in NEGATIVE_PHRASES:
        if norm == phrase or norm.startswith(phrase + " ") or norm.startswith(phrase + ","):
            return "negative"
    for phrase in POSITIVE_PHRASES:
        if norm == phrase or norm.startswith(phrase + " ") or norm.startswith(phrase + ","):
            return "positive"
    return "unclear"


_BARE_TOKENS: frozenset[str] = frozenset(
    {"y", "n", "yes", "no", "yeah", "yep", "yup", "nope", "nah"}
)


def build_label(
    classification: OutcomeClassification,
    body: str,
    *,
    captured_at: datetime,
) -> OutcomeLabel | None:
    """Build an OutcomeLabel from a classification + raw user text.

    Returns None when classification is 'unclear'. When the body is exactly
    a bare yes/no token (any punctuation stripped), `notes` is None;
    otherwise the verbatim user text is stored as `notes`.
    """
    if classification == "unclear":
        return None
    norm = body.strip().rstrip(".!?,").lower()
    notes: str | None = None if norm in _BARE_TOKENS else body.strip()
    return OutcomeLabel(
        captured_at=captured_at,
        did_it_help=(classification == "positive"),
        notes=notes,
    )


def _set_status(
    session: Session,
    status: Literal["pending", "asked", "captured", "skipped"],
) -> Session:
    """Set the outcome probe status on the session manifest."""
    session.outcome_probe_status = status
    return session


def _set_label(session: Session, label: OutcomeLabel) -> Session:
    """Persist a captured outcome label and mark the probe captured."""
    if session.outcome_label is not None:
        return session
    session.outcome_label = label
    session.outcome_probe_status = "captured"
    return session


__all__ = [
    "NEGATIVE_PHRASES",
    "OUTCOME_PROMPT",
    "OUTCOME_REPROMPT",
    "OutcomeClassification",
    "OutcomeProbe",
    "OutcomeProbeConfig",
    "POSITIVE_PHRASES",
    "build_label",
    "classify_outcome",
]
