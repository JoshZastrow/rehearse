"""Boot one rehearse session against a transport.

`RuntimeHost` is the single entry point used by both the eval harness and the
live telephony path. It wires `PhaseProcessor`, `IntakeProcessor`,
`TranscriptWriter`, and `TimingWriter` onto an internal `FrameBus`, drives the
turn-taking loop, and writes the artifact set to a `LocalFilesystemStore`.

The host knows nothing about who is on the other end of `transport`. The
abstraction is the same whether the remote side is a synthetic LLM customer
(`InMemoryTwoWayChannel`) or a live phone call (`TwilioPhoneBridge`).

`CoachVoiceAdapter` abstracts how the coach loop generates responses. In eval
mode `TextOnlyCoachAdapter` calls the Anthropic API directly (text-only). In
serving mode `HumeCoachAdapter` routes through `HumeEVIClient`.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import structlog

from rehearse.bus import FrameBus
from rehearse.frames import (
    EndOfCall,
    Frame,
    IntakeComplete,
    PhaseSignal,
    TranscriptDelta,
)
from rehearse.intake import IntakeProcessor
from rehearse.phases import PhaseBudgets, PhaseProcessor
from rehearse.session import utcnow
from rehearse.storage import LocalFilesystemStore
from rehearse.transport import InMemoryTwoWayChannel, RuntimeTransport, TransportEvent
from rehearse.types import ConsentState, Phase, Session, Speaker
from rehearse.writers import TimingWriter, TranscriptWriter

log = structlog.get_logger(__name__)


class RuntimePhaseTimeoutError(RuntimeError):
    """Raised when a phase exceeds its configured wall-clock budget."""


class CoachVoiceAdapter(Protocol):
    """Abstraction over how the coach generates a response to one user turn."""

    async def respond(self, user_text: str, session_id: str) -> str:
        """Return the coach's text response to the latest user utterance."""
        ...


class TextOnlyCoachAdapter:
    """Calls the Anthropic API directly. No audio. Used in eval mode."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 512,
        temperature: float = 0.7,
        client: Any = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = client
        # History per session_id: list of (role, text) tuples
        self._histories: dict[str, list[tuple[str, str]]] = {}

    def _get_client(self) -> Any:
        if self._client is None:
            import os

            from anthropic import AsyncAnthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            self._client = AsyncAnthropic(api_key=api_key)
        return self._client

    async def respond(self, user_text: str, session_id: str) -> str:
        from rehearse.personas import coach_system_prompt

        history = self._histories.setdefault(session_id, [])
        history.append(("user", user_text))

        messages = _history_to_messages(history)
        client = self._get_client()
        resp = await client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=coach_system_prompt(),
            messages=messages,
        )
        coach_text = "".join(
            block.text
            for block in resp.content
            if getattr(block, "type", None) == "text"
        ).strip()
        if not coach_text:
            coach_text = "I hear you. Tell me more."
        history.append(("coach", coach_text))
        return coach_text


class HumeCoachAdapter:
    """Routes coach responses through HumeEVIClient. Used in serving mode.

    In the serving path, Hume EVI generates coach responses autonomously from
    the audio stream. This adapter can be used to inject specific utterances
    (e.g., persona swap announcements) via `hume.say()`. The regular
    turn-taking loop in `RuntimeHost` is not used in serving mode — Hume
    drives the conversation independently.
    """

    def __init__(self, hume: Any) -> None:
        self._hume = hume

    async def respond(self, user_text: str, session_id: str) -> str:
        # In serving mode, Hume generates responses autonomously.
        # This method is a no-op for the regular turn loop; call hume.say()
        # directly for injected utterances.
        return ""


@dataclass(frozen=True)
class SessionArtifacts:
    """Paths to the artifact files produced by one RuntimeHost.run() call."""

    session_json: Path
    transcript_jsonl: Path
    intake_json: Path | None
    persona_json: Path | None
    phase_timing_json: Path


class RuntimeHost:
    """Boot one rehearse session against a transport.

    The same object is used in serving (with TwilioPhoneBridge + HumeCoachAdapter)
    and in eval (with InMemoryTwoWayChannel + TextOnlyCoachAdapter).
    """

    def __init__(
        self,
        store: LocalFilesystemStore,
        coach: CoachVoiceAdapter,
        *,
        budgets: PhaseBudgets | None = None,
        clock: Callable[[], datetime] = utcnow,
        phase_timeout_s: float = 60.0,
        bus: FrameBus | None = None,
    ) -> None:
        self._store = store
        self._coach = coach
        self._budgets = budgets or PhaseBudgets()
        self._clock = clock
        self._phase_timeout_s = phase_timeout_s
        self._external_bus = bus
        self._current_phase: Phase = Phase.INTAKE

    @property
    def current_phase(self) -> Phase:
        # Safe to read without a lock — asyncio is single-threaded.
        return self._current_phase

    async def run(
        self,
        *,
        session_id: str,
        transport: RuntimeTransport,
        consent: ConsentState = ConsentState.GRANTED,
    ) -> SessionArtifacts:
        """Drive one call to completion and return paths to produced artifacts."""
        bus = self._external_bus or FrameBus(session_id)
        run_dir = self._store.session_dir(session_id)

        # Create session manifest.
        session = Session(
            id=session_id,
            created_at=self._clock(),
            consent=consent,
        )
        await self._store.write(
            session_id, "session.json", session.model_dump_json(indent=2)
        )

        phase_processor = PhaseProcessor(
            session_id,
            self._store,
            bus,
            budgets=self._budgets,
            clock=self._clock,
            wait_for_intake_complete=True,
        )
        intake_processor = IntakeProcessor(
            session_id,
            self._store,
            phase_getter=lambda: self._current_phase,
            clock=self._clock,
            bus=bus,
        )
        transcript_writer = TranscriptWriter(
            session_id,
            self._store,
            phase_getter=lambda: self._current_phase,
        )

        await phase_processor.bootstrap()

        intake_json_path: Path | None = None
        intake_complete_event = asyncio.Event()

        async def _watch_signals(frames: AsyncIterator[Frame]) -> None:
            nonlocal intake_json_path
            async for frame in frames:
                if isinstance(frame, PhaseSignal):
                    self._current_phase = frame.to_phase
                    # Send the control event here, directly on the PhaseSignal,
                    # so we don't race against the asyncio.to_thread() file writes
                    # inside PhaseProcessor._transition().
                    await transport.send(
                        "control",
                        payload={
                            "event": "phase_transition",
                            "phase": frame.to_phase.upper(),
                        },
                    )
                elif isinstance(frame, IntakeComplete):
                    intake_json_path = Path(frame.intake_path)
                    intake_complete_event.set()
                elif isinstance(frame, EndOfCall):
                    break

        phase_task = asyncio.create_task(phase_processor.run(bus.subscribe()))
        intake_task = asyncio.create_task(intake_processor.run(bus.subscribe()))
        transcript_task = asyncio.create_task(transcript_writer.run(bus.subscribe()))
        signal_task = asyncio.create_task(_watch_signals(bus.subscribe()))

        # Give each subscriber task one event-loop tick to enter its async-for
        # loop and register its queue with the bus.  Without this, the first
        # publish inside _run_loop fires before any subscription exists, losing
        # the first frame.
        await asyncio.sleep(0)

        try:
            async with asyncio.timeout(self._phase_timeout_s * 3):
                await self._run_loop(session_id, transport, bus, intake_complete_event)
        except TimeoutError:
            raise RuntimePhaseTimeoutError(
                f"RuntimeHost session {session_id} exceeded total timeout "
                f"({self._phase_timeout_s * 3:.0f}s)"
            )
        finally:
            await bus.aclose()
            for task in (phase_task, intake_task, transcript_task, signal_task):
                with suppress(asyncio.CancelledError):
                    await task

        return await self._write_final_artifacts(
            session_id, run_dir, intake_json_path
        )

    async def _run_loop(
        self,
        session_id: str,
        transport: RuntimeTransport,
        bus: FrameBus,
        intake_complete_event: asyncio.Event,
    ) -> None:
        """Read user turns from transport, call coach, write coach turns back."""
        utterance_idx = 0
        while True:
            try:
                event: TransportEvent = await asyncio.wait_for(
                    transport.receive(), timeout=self._phase_timeout_s
                )
            except TimeoutError:
                raise RuntimePhaseTimeoutError(
                    f"{self._current_phase.upper()} phase exceeded "
                    f"{self._phase_timeout_s:.0f}s budget"
                )
            except Exception:
                break

            if event.kind == "control":
                ev = event.payload.get("event", "")
                if ev in ("customer_done", "end_of_call", "runtime_done"):
                    await bus.publish(
                        EndOfCall(
                            session_id=session_id,
                            reason="hangup",
                            ts=self._clock().timestamp(),
                        )
                    )
                    break
                continue

            if event.kind != "text":
                continue

            user_text = str(event.payload.get("text", "")).strip()
            if not user_text:
                continue

            utt_id = f"utt-{utterance_idx:04d}"
            utterance_idx += 1
            now = self._clock().timestamp()
            await bus.publish(
                TranscriptDelta(
                    session_id=session_id,
                    utterance_id=utt_id,
                    speaker=Speaker.USER,
                    text=user_text,
                    is_final=True,
                    ts_start=now,
                    ts_end=now,
                )
            )

            # Call the coach and publish / send the response.
            coach_text = await self._coach.respond(user_text, session_id)
            if not coach_text:
                continue

            coach_utt_id = f"utt-{utterance_idx:04d}"
            utterance_idx += 1
            coach_now = self._clock().timestamp()
            await bus.publish(
                TranscriptDelta(
                    session_id=session_id,
                    utterance_id=coach_utt_id,
                    speaker=Speaker.COACH,
                    text=coach_text,
                    is_final=True,
                    ts_start=coach_now,
                    ts_end=coach_now,
                )
            )
            await transport.send(
                "text",
                payload={"text": coach_text, "role": "coach"},
            )

    async def _write_final_artifacts(
        self,
        session_id: str,
        run_dir: Path,
        intake_json_path: Path | None,
    ) -> SessionArtifacts:
        """Write intake.json, persona.json, phase_timing.json from session manifest."""
        session_path = run_dir / "session.json"
        session = Session.model_validate_json(session_path.read_text())

        if session.intake is not None:
            # intake_json_path may have been set by _watch_signals when it received
            # IntakeComplete; that path is declared by IntakeProcessor but the file
            # isn't actually written to disk until here.
            intake_json_path = intake_json_path or run_dir / "intake.json"
            intake_json_path.write_text(session.intake.model_dump_json(indent=2))
        else:
            intake_json_path = None

        persona_json_path: Path | None = None
        if session.persona is not None:
            persona_json_path = run_dir / "persona.json"
            persona_json_path.write_text(session.persona.model_dump_json(indent=2))

        phase_timing_path = run_dir / "phase_timing.json"
        phase_timing_path.write_text(
            json.dumps(
                [pt.model_dump(mode="json") for pt in session.phase_timings],
                indent=2,
            )
        )

        return SessionArtifacts(
            session_json=session_path,
            transcript_jsonl=run_dir / "transcript.jsonl",
            intake_json=intake_json_path,
            persona_json=persona_json_path,
            phase_timing_json=phase_timing_path,
        )


def _history_to_messages(
    history: list[tuple[str, str]],
) -> list[dict[str, str]]:
    """Render alternating (role, text) history as Anthropic-style messages."""
    messages: list[dict[str, str]] = []
    for role, text in history:
        api_role = "user" if role == "user" else "assistant"
        messages.append({"role": api_role, "content": text})
    if not messages or messages[0]["role"] != "user":
        messages.insert(0, {"role": "user", "content": "Begin the coaching session."})
    return messages
