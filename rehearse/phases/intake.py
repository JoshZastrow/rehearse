"""Capture intake structure and compile a practice persona during the live call.

This file observes transcript and phase events on the runtime bus, keeps a
simple deterministic intake record up to date during Phase 1, and compiles a
session-specific character persona when the call enters practice.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from typing import Any

import structlog

from rehearse.bus import FrameBus
from rehearse.frames import Frame, IntakeComplete, PhaseSignal, TranscriptDelta
from rehearse.personas import build_intake_record, compile_character
from rehearse.session.session import utcnow
from rehearse.storage import LocalFilesystemStore
from rehearse.types import Phase, Session, Speaker

log = structlog.get_logger(__name__)

# Words that signal a personal preference is being expressed
_PREFERENCE_SIGNALS = frozenset({
    "prefer", "rather", "easier", "better", "comfortable", "tend", "usually",
    "like", "feel", "feels", "honestly", "actually", "please", "probably",
    "think", "reckon", "guess", "maybe", "perhaps",
})

# Gender words only count when a preference signal is present, to avoid
# misclassifying "my manager told him" or "she runs the team" as a preference.
_FEMALE_WORDS = frozenset({"woman", "women", "female", "lady", "ladies", "she", "her"})
_MALE_WORDS = frozenset({"man", "men", "male", "guy", "guys", "gentleman", "he", "him"})

# Keywords that suggest the coach just asked about gender preference
_COACH_GENDER_QUESTION_SIGNALS = frozenset({
    "men or women", "man or woman", "male or female", "gender", "who you prefer",
    "who would you rather", "who'd you rather", "sparring with", "practice with",
    "comfortable with",
})


def classify_gender_preference(text: str) -> str | None:
    """Return 'male', 'female', or None from a user turn expressing a preference.

    Conservative: only triggers when a preference-signal word AND a gender word
    appear together. A plain gendered pronoun about a third party ('his decision')
    returns None.
    """
    if not text:
        return None
    import re as _re
    lowered = text.lower()
    words = set(_re.sub(r"[^\w\s]", " ", lowered).split())

    has_preference = bool(words & _PREFERENCE_SIGNALS)
    has_female = bool(words & _FEMALE_WORDS)
    has_male = bool(words & _MALE_WORDS)

    # Short answers like "a woman" or "male please" are unambiguous
    short = len(words) <= 4

    if has_female and not has_male and (has_preference or short):
        return "female"
    if has_male and not has_female and (has_preference or short):
        return "male"

    return None


def _coach_asked_gender_question(text: str) -> bool:
    """Return True if a coach utterance looks like the gender preference question."""
    lowered = text.lower()
    return any(signal in lowered for signal in _COACH_GENDER_QUESTION_SIGNALS)


class IntakeProcessor:
    """Persist structured intake and compiled persona from live transcript events."""

    def __init__(
        self,
        session_id: str,
        store: LocalFilesystemStore,
        *,
        phase_getter: Callable[[], Phase],
        clock: Callable[[], datetime] = utcnow,
        bus: FrameBus | None = None,
        llm_client: Any | None = None,
    ) -> None:
        """Store the session id, manifest store, and current-phase callback."""
        self._session_id = session_id
        self._store = store
        self._phase_getter = phase_getter
        self._clock = clock
        self._bus = bus
        self._llm_client = llm_client
        self._phase = Phase.INTAKE
        self._user_turns: list[str] = []
        self._intake_complete_emitted = False
        self._coach_asked_gender = False
        self._gender_preference: str | None = None
        self._topic_task: asyncio.Task[str] | None = None

    async def run(self, frames: AsyncIterator[Frame]) -> None:
        """Consume bus frames and update intake/persona artifacts as phases change."""
        async for frame in frames:
            if isinstance(frame, TranscriptDelta) and frame.is_final and self._phase == Phase.INTAKE:
                if frame.speaker == Speaker.GUIDE:
                    # Watch for the coach asking about gender preference
                    if _coach_asked_gender_question(frame.text):
                        self._coach_asked_gender = True

                elif frame.speaker == Speaker.USER:
                    if self._coach_asked_gender and self._gender_preference is None:
                        # Classify the first user turn after the gender question
                        detected = classify_gender_preference(frame.text.strip())
                        if detected is not None:
                            self._gender_preference = detected
                        self._coach_asked_gender = False  # consume the flag

                    self._user_turns.append(frame.text.strip())
                    if len(self._user_turns) == 1:
                        self._start_topic_classification()
                    await self._persist_intake()
                    if len(self._user_turns) >= 2:
                        await self._compile_persona()
                        await self._emit_intake_complete()

            elif isinstance(frame, PhaseSignal):
                self._phase = frame.to_phase
                if frame.to_phase == Phase.PRACTICE:
                    await self._compile_persona()
                    await self._emit_intake_complete()
        if self._user_turns and (
            self._phase != Phase.INTAKE or self._phase_getter() != Phase.INTAKE
        ):
            await self._compile_persona()
            await self._emit_intake_complete()

    def _start_topic_classification(self) -> None:
        """Kick off topic classification in the background after the first user turn.

        Runs concurrently while the coach speaks — result is picked up by the
        next _persist_intake call. Skipped when no LLM client is configured.
        """
        if self._llm_client is None or self._topic_task is not None:
            return
        situation = " ".join(self._user_turns)
        from rehearse.agents.topic import classify_topic
        self._topic_task = asyncio.create_task(
            classify_topic(situation, client=self._llm_client)
        )

    def _resolved_topic(self) -> str | None:
        """Return classified topic if the background task has completed."""
        if self._topic_task is not None and self._topic_task.done():
            try:
                return self._topic_task.result()
            except Exception as exc:
                log.warning("intake.topic_classification_failed", error=str(exc))
        return None

    async def _persist_intake(self) -> None:
        """Write the latest intake guess into the session manifest."""
        if not self._user_turns:
            return
        captured_at = self._clock()
        gender = self._gender_preference
        topic = self._resolved_topic()
        await self._store.update_session(
            self._session_id,
            lambda session: _apply_intake(
                session,
                build_intake_record(
                    session_id=self._session_id,
                    user_turns=self._user_turns,
                    captured_at=captured_at,
                    gender_preference=gender,
                    topic_category=topic,
                ),
            ),
        )

    async def _compile_persona(self) -> None:
        """Compile and persist the character persona from the stored intake."""
        compiled_at = self._clock()
        await self._store.update_session(
            self._session_id,
            lambda session: _apply_persona(session, compiled_at=compiled_at),
        )

    async def _emit_intake_complete(self) -> None:
        """Emit IntakeComplete on the bus once (idempotent) when a bus is configured."""
        if self._bus is None or self._intake_complete_emitted:
            return
        self._intake_complete_emitted = True
        intake_path = str(self._store.session_dir(self._session_id) / "intake.json")
        await self._bus.publish(
            IntakeComplete(session_id=self._session_id, intake_path=intake_path)
        )


def _apply_intake(session: Session, intake) -> Session:
    """Set the current intake record on the stored session manifest."""
    session.intake = intake
    return session


def _apply_persona(session: Session, *, compiled_at: datetime) -> Session:
    """Compile and store the session persona when intake data exists."""
    if session.intake is not None:
        session.persona = compile_character(session.intake, compiled_at=compiled_at)
    return session
