"""Capture verbal consent at the start of a live call.

The runtime owns one new component, `ConsentGate`, that speaks the consent
prompt as the first coach utterance, classifies the user's spoken response,
and either unblocks the phase machine on `granted` or terminates the call on
`declined`.

Spec: docs/specs/v2026-05-01-consent-and-outcome-capture.md §3.1, §7.1
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import structlog

from rehearse.bus import FrameBus
from rehearse.frames import ConsentResolved, EndOfCall, Frame, TranscriptDelta
from rehearse.participants import SpeakRequest, VoiceSpeaker
from rehearse.memory import CallerMemory, NullCallerMemory
from rehearse.personas import (
    CONSENT_DECLINE_ACK,
    CONSENT_PROMPT,
    CONSENT_REMINDERS,
    CONSENT_REPROMPT,
    classify_consent,
)
from rehearse.session import utcnow
from rehearse.storage import LocalFilesystemStore
from rehearse.types import ConsentState, Session, Speaker

log = structlog.get_logger(__name__)


@dataclass
class ConsentGateConfig:
    """Configuration knobs for the consent gate."""

    prompt_timeout_seconds: int = 45
    reprompt_limit: int = 1


class ConsentGate:
    """Run the consent prompt + classification before any practice content.

    The gate must be the first surface to act on the call. It speaks the
    consent prompt via the injected speaker, classifies the next
    final user transcript frame, and resolves the session into either
    `GRANTED` (intake proceeds) or `DECLINED` (call ends with the orchestrator
    finalize hook).
    """

    def __init__(
        self,
        session_id: str,
        store: LocalFilesystemStore,
        bus: FrameBus,
        *,
        speaker: VoiceSpeaker,
        on_decline: Callable[[], Awaitable[None]],
        config: ConsentGateConfig | None = None,
        clock: Callable[[], datetime] = utcnow,
        caller_hash: str | None = None,
        memory: CallerMemory | None = None,
    ) -> None:
        """Store identity, persistence, speaker, decline hook, and timing knobs."""
        self._session_id = session_id
        self._store = store
        self._bus = bus
        self._speaker = speaker
        self._on_decline = on_decline
        self._config = config or ConsentGateConfig()
        self._clock = clock
        self._caller_hash = caller_hash
        self._memory: CallerMemory = memory or NullCallerMemory()
        self._asked_at: datetime | None = None
        self._reprompts = 0
        self._resolved = False
        self._timeout_task: asyncio.Task[None] | None = None

    async def run(self, frames: AsyncIterator[Frame]) -> None:
        """Speak the prompt and consume bus frames until consent resolves."""
        if self._caller_hash and await self._memory.has_prior_consent(self._caller_hash):
            # Returning caller: pick a reminder phrase at random and proceed immediately.
            # No "yes" needed — they already consented on a prior call.
            reminder = random.choice(CONSENT_REMINDERS)
            await self._speaker.say(SpeakRequest(text=reminder))
            log.info("consent.reminder.spoken", session_id=self._session_id, text=reminder)
            await self._grant()
            return
        # First-time caller: full prompt, wait for spoken response.
        await self._ask(CONSENT_PROMPT)
        try:
            async for frame in frames:
                if self._resolved:
                    return
                if (
                    isinstance(frame, TranscriptDelta)
                    and frame.speaker == Speaker.USER
                    and frame.is_final
                ):
                    await self._handle_response(frame.text)
                    if self._resolved:
                        return
                elif isinstance(frame, EndOfCall):
                    return
        finally:
            if self._timeout_task is not None and not self._timeout_task.done():
                self._timeout_task.cancel()

    async def _ask(self, prompt: str) -> None:
        """Speak a prompt, mark asked, and reset the timeout watcher."""
        await self._speaker.say(SpeakRequest(text=prompt))
        self._asked_at = self._clock()
        log.info("consent.prompt.spoken", session_id=self._session_id)
        if self._timeout_task is not None:
            self._timeout_task.cancel()
        self._timeout_task = asyncio.create_task(self._fire_timeout())

    async def _fire_timeout(self) -> None:
        """Drive a no-response decline once the prompt timeout elapses."""
        try:
            await asyncio.sleep(self._config.prompt_timeout_seconds)
        except asyncio.CancelledError:
            return
        if not self._resolved:
            await self._decline(reason="timeout")

    async def _handle_response(self, text: str) -> None:
        """Classify the user's reply and resolve consent or re-prompt."""
        classification = classify_consent(text)
        log.info(
            "consent.classify",
            session_id=self._session_id,
            classification=classification,
            text_len=len(text),
        )
        if classification == "granted":
            await self._grant()
        elif classification == "declined":
            await self._decline(reason="explicit")
        else:
            if self._reprompts < self._config.reprompt_limit:
                self._reprompts += 1
                await self._ask(CONSENT_REPROMPT)
            else:
                await self._decline(reason="unclear")

    async def _grant(self) -> None:
        """Persist GRANTED, emit ConsentResolved, and let the phase machine proceed."""
        if self._resolved:
            return
        self._resolved = True
        if self._timeout_task is not None:
            self._timeout_task.cancel()
        now = self._clock()
        await self._store.update_session(
            self._session_id,
            lambda s: _set_consent(s, ConsentState.GRANTED),
        )
        latency_ms = (
            int((now - self._asked_at).total_seconds() * 1000)
            if self._asked_at is not None
            else 0
        )
        log.info(
            "consent.granted",
            session_id=self._session_id,
            latency_ms=latency_ms,
        )
        await self._store.append(
            self._session_id,
            "telemetry.jsonl",
            json.dumps(
                {
                    "event": "consent.granted.latency_ms",
                    "value": latency_ms,
                    "ts": now.isoformat(),
                }
            ),
        )
        await self._bus.publish(
            ConsentResolved(
                session_id=self._session_id,
                state=ConsentState.GRANTED,
                ts=now.timestamp(),
            )
        )
        if self._caller_hash:
            await self._memory.record_consent(self._caller_hash)

    async def _decline(self, *, reason: Literal["explicit", "timeout", "unclear"]) -> None:
        """Persist DECLINED, speak the acknowledgement, hand off to finalize."""
        if self._resolved:
            return
        self._resolved = True
        if self._timeout_task is not None:
            self._timeout_task.cancel()
        await self._store.update_session(
            self._session_id,
            lambda s: _set_consent(s, ConsentState.DECLINED),
        )
        log.info("consent.declined", session_id=self._session_id, reason=reason)
        try:
            await self._speaker.say(SpeakRequest(text=CONSENT_DECLINE_ACK))
        except Exception as exc:
            log.warning(
                "consent.decline.ack_failed",
                session_id=self._session_id,
                error=str(exc),
            )
        await self._bus.publish(
            ConsentResolved(
                session_id=self._session_id,
                state=ConsentState.DECLINED,
                ts=self._clock().timestamp(),
            )
        )
        await self._on_decline()


def _set_consent(session: Session, state: ConsentState) -> Session:
    """Update the consent state on the session manifest."""
    session.consent = state
    return session


__all__ = ["ConsentGate", "ConsentGateConfig"]
