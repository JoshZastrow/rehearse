"""Transport-agnostic conversation session runner.

`run_session()` owns the full conversation pipeline for one call: phase
machine, intake, consent, outcome probe, survey, writers, and memory
recording. The caller (Twilio, WebRTC, or scripted eval) and the backend
(Hume EVI, Pipeline, or stub) are injected as dependencies.

Callers produce audio; the backend produces provider responses. Nothing in
this module knows or cares which caller or backend is active.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

import anyio
import structlog

from rehearse.agents.intake_recorder import IntakeMemoryRecorder
from rehearse.agents.persona_selection_recorder import PersonaSelectionRecorder
from rehearse.agents.persona_swap import PersonaSwapCoordinator
from rehearse.backends.base import ConversationBackend
from rehearse.bus import FrameBus
from rehearse.phases.consent import ConsentGate, ConsentGateConfig
from rehearse.frames import AudioChunk, EndOfCall
from rehearse.phases.intake import IntakeProcessor
from rehearse.memory.memory import CallerMemory, NullCallerMemory
from rehearse.phases.outcome import OutcomeProbe, OutcomeProbeConfig
from rehearse.audio.participants import VoiceParticipant
from rehearse.phases.phases import PhaseBudgets, PhaseProcessor
from rehearse.session.session import utcnow
from rehearse.storage import LocalFilesystemStore
from rehearse.phases.survey import SurveyAgent, SurveyAgentConfig
from rehearse.types import ConsentState, Session, Speaker
from rehearse.writers import (
    AudioRecorder,
    ProsodyWriter,
    TelemetryLogger,
    TimingWriter,
    TranscriptWriter,
)

log = structlog.get_logger(__name__)


async def run_session(
    session_id: str,
    caller: VoiceParticipant,
    backend: ConversationBackend,
    *,
    store: LocalFilesystemStore,
    memory: CallerMemory | None = None,
    caller_hash: str | None = None,
    budgets: PhaseBudgets | None = None,
    skip_consent: bool = False,
    enable_consent: bool = True,
    consent_timeout_seconds: int = 20,
    consent_reprompt_limit: int = 1,
    outcome_timeout_seconds: int = 15,
    outcome_reprompt_limit: int = 1,
    outcome_lead_seconds: int = 15,
    model_id: str = "default",
    llm_client: Any = None,
    on_consent_declined: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Run the full conversation pipeline for one session.

    Extracts the orchestration body of telephony.py:media_stream() so any
    caller (Twilio, WebRTC, scripted eval) can drive the same pipeline.

    Args:
        session_id: Unique identifier for this session.
        caller: Audio source / sink (produces chunks, receives provider audio).
        backend: Provider backend (Hume EVI, Pipeline, stub).
        store: Session artifact store.
        memory: Cross-session caller memory (defaults to NullCallerMemory).
        caller_hash: Stable hash of caller identity for memory ops.
        budgets: Phase time budgets (defaults to PhaseBudgets()).
        skip_consent: Pre-grant consent and skip ConsentGate (use in eval).
        enable_consent: Whether to run ConsentGate at all.
        consent_timeout_seconds: Timeout for the consent prompt response.
        consent_reprompt_limit: Max re-prompts before declining.
        outcome_timeout_seconds: Timeout for the outcome probe response.
        outcome_reprompt_limit: Max re-prompts before skipping.
        outcome_lead_seconds: Seconds before feedback end to ask outcome.
        model_id: Backend model identifier written to telemetry.
        llm_client: Anthropic client for intake LLM refinement (optional).
        on_consent_declined: Callback when consent is declined (e.g. finalize).
    """
    _memory = memory or NullCallerMemory()
    _budgets = budgets or PhaseBudgets()

    bus = FrameBus(session_id)
    consent_state: dict[str, bool] = {"declined": False}

    def _consent_getter() -> ConsentState:
        path = store.session_dir(session_id) / "session.json"
        try:
            return Session.model_validate_json(path.read_text()).consent
        except Exception:
            return ConsentState.PENDING

    phase_processor = PhaseProcessor(
        session_id,
        store,
        bus,
        budgets=_budgets,
        consent_getter=_consent_getter,
    )
    intake_processor = IntakeProcessor(
        session_id,
        store,
        phase_getter=lambda: phase_processor.current_phase,
        llm_client=llm_client,
    )

    async def _on_decline() -> None:
        consent_state["declined"] = True
        await bus.publish(
            EndOfCall(
                session_id=session_id,
                reason="consent_decline",
                ts=utcnow().timestamp(),
            )
        )
        if on_consent_declined is not None:
            await on_consent_declined()

    if skip_consent or not enable_consent:
        await store.update_session(session_id, _grant_consent)

    if enable_consent and not skip_consent:
        consent_gate: ConsentGate | None = ConsentGate(
            session_id,
            store,
            bus,
            speaker=backend,
            on_decline=_on_decline,
            config=ConsentGateConfig(
                prompt_timeout_seconds=consent_timeout_seconds,
                reprompt_limit=consent_reprompt_limit,
            ),
            caller_hash=caller_hash,
            memory=_memory,
        )
    else:
        consent_gate = None

    outcome_probe = OutcomeProbe(
        session_id,
        store,
        speaker=backend,
        config=OutcomeProbeConfig(
            response_timeout_seconds=outcome_timeout_seconds,
            reprompt_limit=outcome_reprompt_limit,
            prompt_lead_seconds=outcome_lead_seconds,
            feedback_budget_seconds=_budgets.feedback_seconds,
        ),
    )
    survey_agent = SurveyAgent(
        session_id,
        store,
        speaker=backend,
        config=SurveyAgentConfig(),
    )
    persona_swap = PersonaSwapCoordinator(
        session_id,
        store,
        speaker=backend,
        backend=backend,
    )

    await phase_processor.bootstrap()

    async def _consent_noop() -> None:
        pass

    consent_task = asyncio.create_task(
        consent_gate.run(bus.subscribe()) if consent_gate else _consent_noop()
    )
    outcome_task = asyncio.create_task(outcome_probe.run(bus.subscribe()))
    survey_task = asyncio.create_task(survey_agent.run(bus.subscribe()))
    phase_task = asyncio.create_task(phase_processor.run(bus.subscribe()))
    intake_task = asyncio.create_task(intake_processor.run(bus.subscribe()))
    intake_recorder_task = asyncio.create_task(
        IntakeMemoryRecorder(session_id, caller_hash, _memory).run(bus.subscribe())
    )
    persona_selection_task = asyncio.create_task(
        PersonaSelectionRecorder(session_id, caller_hash, _memory, store).run(bus.subscribe())
    )
    persona_swap_task = asyncio.create_task(persona_swap.run(bus.subscribe()))
    transcript_task = asyncio.create_task(
        TranscriptWriter(
            session_id,
            store,
            phase_getter=lambda: phase_processor.current_phase,
        ).run(bus.subscribe())
    )
    prosody_task = asyncio.create_task(
        ProsodyWriter(session_id, store).run(bus.subscribe())
    )
    audio_task = asyncio.create_task(
        AudioRecorder(session_id, store).run(bus.subscribe())
    )
    timing_task = asyncio.create_task(
        TimingWriter(session_id, store).run(bus.subscribe())
    )
    telemetry_task = asyncio.create_task(
        TelemetryLogger(
            session_id,
            store,
            model=model_id,
            phase_getter=lambda: phase_processor.current_phase,
        ).run(bus.subscribe())
    )
    provider_audio_task = asyncio.create_task(_pump_provider_audio(caller, bus))
    backend_task = asyncio.create_task(backend.start(session_id, bus))

    try:
        async for chunk in caller.audio_stream(bus):
            if consent_state["declined"]:
                break
            await backend.send_caller_audio(chunk)
    finally:
        with anyio.CancelScope(shield=True):
            await bus.aclose()
            provider_audio_task.cancel()
            backend_task.cancel()
            with suppress(asyncio.CancelledError):
                await provider_audio_task
            with suppress(asyncio.CancelledError):
                await backend_task
            await consent_task
            await outcome_task
            await survey_task
            await phase_task
            await intake_task
            await intake_recorder_task
            await persona_selection_task
            await persona_swap_task
            await transcript_task
            await prosody_task
            await audio_task
            await timing_task
            await telemetry_task
            if caller_hash:
                transcript = await _read_transcript(session_id, store)
                if transcript:
                    try:
                        await _memory.store_session(
                            caller_hash,
                            transcript,
                            rehearse_session_id=session_id,
                        )
                    except Exception as exc:
                        log.warning(
                            "conversation.store_session.failed",
                            session_id=session_id,
                            error=str(exc),
                        )


async def _pump_provider_audio(caller: VoiceParticipant, bus: FrameBus) -> None:
    """Forward provider audio frames from the bus back to the caller."""
    async for frame in bus.subscribe():
        if isinstance(frame, AudioChunk) and frame.speaker != Speaker.USER:
            await caller.receive_audio(frame.pcm16_16k)


def _grant_consent(session: Session) -> Session:
    session.consent = ConsentState.GRANTED
    return session


async def _read_transcript(session_id: str, store: LocalFilesystemStore) -> list[dict]:
    try:
        content = await store.read(session_id, "transcript.jsonl")
    except FileNotFoundError:
        return []
    import json
    messages: list[dict] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            frame = json.loads(line)
            text = (frame.get("text") or "").strip()
            if not text or frame.get("is_interim", False):
                continue
            role = "user" if frame.get("speaker") == "user" else "assistant"
            messages.append({"role": role, "content": text})
        except Exception:
            continue
    return messages
