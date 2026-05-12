"""Handle Twilio webhooks, REST calls, and media streams for the runtime.

This file is the live runtime entrypoint for telephony. It owns inbound SMS and
voice webhooks, outbound call placement, call-status callbacks, and the live
Twilio media websocket that feeds audio into Hume.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Protocol

import anyio
import structlog
from fastapi import (
    BackgroundTasks,
    FastAPI,
    Form,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import PlainTextResponse, Response
from twilio.request_validator import RequestValidator
from twilio.rest import Client as TwilioClient

from rehearse.agents.persona_swap import PersonaSwapCoordinator
from rehearse.audio.twilio_stream import TwilioCallerParticipant, TwilioStream
from rehearse.bus import FrameBus
from rehearse.config import RuntimeConfig
from rehearse.consent import ConsentGate, ConsentGateConfig
from rehearse.frames import AudioChunk, EndOfCall
from rehearse.intake import IntakeProcessor
from rehearse.memory import CallerMemory, HonchoCallerMemory, MCPCallerMemory, NullCallerMemory
from rehearse.outcome import OutcomeProbe, OutcomeProbeConfig
from rehearse.phases import PhaseBudgets, PhaseProcessor
from rehearse.services.hume_evi import HumeEVIParticipant
from rehearse.session import SessionOrchestrator, TriggerEvent, utcnow
from rehearse.types import Session, Speaker
from rehearse.writers import (
    AudioRecorder,
    ProsodyWriter,
    TelemetryLogger,
    TimingWriter,
    TranscriptWriter,
)

log = structlog.get_logger(__name__)


class TelephonyClient(Protocol):
    """Small interface for the Twilio operations the runtime needs."""

    async def place_call(self, to: str, callback_url: str, status_callback: str) -> str: ...
    async def send_sms(self, to: str, body: str) -> str: ...


class TwilioRestClient:
    """Real Twilio REST client wrapper used by the live runtime."""

    def __init__(self, config: RuntimeConfig) -> None:
        """Create a wrapper around the Twilio Python SDK."""
        self._config = config
        self._client = TwilioClient(config.twilio_account_sid, config.twilio_auth_token)

    async def place_call(self, to: str, callback_url: str, status_callback: str) -> str:
        """Place one outbound call and return the created Twilio call SID."""
        def _create() -> str:
            call = self._client.calls.create(
                to=to,
                from_=self._config.twilio_from_number,
                url=callback_url,
                status_callback=status_callback,
                status_callback_event=["initiated", "answered", "completed"],
                status_callback_method="POST",
            )
            return call.sid

        return await asyncio.to_thread(_create)

    async def send_sms(self, to: str, body: str) -> str:
        """Send one SMS message and return the created Twilio message SID."""
        def _send() -> str:
            msg = self._client.messages.create(
                to=to,
                from_=self._config.twilio_from_number,
                body=body,
            )
            return msg.sid

        return await asyncio.to_thread(_send)


def mount_twilio_routes(
    app: FastAPI,
    orchestrator: SessionOrchestrator,
    client: TelephonyClient,
    config: RuntimeConfig,
) -> None:
    """Register all Twilio HTTP and websocket routes on the FastAPI app."""
    validator = RequestValidator(config.twilio_auth_token)

    async def _validate(request: Request) -> None:
        """Reject a webhook request when the Twilio signature is invalid."""
        if not config.validate_twilio_signature:
            return
        signature = request.headers.get("X-Twilio-Signature", "")
        url = str(request.url)
        form = await request.form()
        params = {k: str(v) for k, v in form.items()}
        if not validator.validate(url, params, signature):
            log.warning("twilio.signature.invalid", url=url)
            raise HTTPException(status_code=403, detail="invalid twilio signature")

    @app.post("/twilio/sms")
    async def twilio_sms(
        request: Request,
        background: BackgroundTasks,
        From: str = Form(...),
        Body: str = Form(""),
    ) -> Response:
        """Start a new session from an inbound SMS and queue the outbound call."""
        await _validate(request)
        trigger = TriggerEvent(from_number=From, body=Body, received_at=utcnow())
        handle = await orchestrator.start(trigger)

        voice_url = f"{config.public_base_url}/twilio/voice?session_id={handle.session_id}"
        status_url = f"{config.public_base_url}/twilio/status?session_id={handle.session_id}"

        async def _place() -> None:
            """Place the outbound call and attach its SID to the session."""
            try:
                call_sid = await client.place_call(From, voice_url, status_url)
                await orchestrator.attach_call(handle.session_id, call_sid)
            except Exception as exc:
                log.exception(
                    "twilio.place_call.failed",
                    session_id=handle.session_id,
                    error=str(exc),
                )
                await orchestrator.finalize(handle.session_id, "failed")

        background.add_task(_place)
        return PlainTextResponse(
            '<?xml version="1.0" encoding="UTF-8"?><Response/>',
            media_type="application/xml",
        )

    @app.post("/twilio/voice")
    async def twilio_voice(request: Request, session_id: str) -> Response:
        """Return TwiML that connects an outbound call to the media stream."""
        await _validate(request)
        log.info("twilio.voice", session_id=session_id)
        return PlainTextResponse(_stream_twiml(config, session_id), media_type="application/xml")

    @app.post("/twilio/voice/inbound")
    async def twilio_voice_inbound(
        request: Request,
        From: str = Form(...),
        CallSid: str = Form(""),
    ) -> Response:
        """Start a new session from a direct inbound phone call."""
        await _validate(request)
        trigger = TriggerEvent(from_number=From, body="<inbound-call>", received_at=utcnow())
        handle = await orchestrator.start(trigger)
        if CallSid:
            await orchestrator.attach_call(handle.session_id, CallSid)
        log.info("twilio.voice.inbound", session_id=handle.session_id, call_sid=CallSid)
        return PlainTextResponse(
            _stream_twiml(config, handle.session_id),
            media_type="application/xml",
        )

    @app.post("/twilio/status")
    async def twilio_status(
        request: Request,
        session_id: str | None = None,
        CallStatus: str = Form(...),
        CallSid: str = Form(""),
    ) -> Response:
        """Handle Twilio call-status callbacks and finalize finished sessions."""
        await _validate(request)
        resolved = session_id or (orchestrator.find_by_call_sid(CallSid) if CallSid else None)
        log.info(
            "twilio.status",
            session_id=resolved,
            call_sid=CallSid,
            status=CallStatus,
        )
        if resolved and CallStatus in {"completed", "failed", "no-answer", "busy", "canceled"}:
            outcome = "complete" if CallStatus == "completed" else "failed"
            await orchestrator.finalize(resolved, outcome)
        return PlainTextResponse("", status_code=204)

    @app.websocket("/media/{session_id}")
    async def media_stream(ws: WebSocket, session_id: str) -> None:
        """Bridge the live Twilio media websocket to Hume and the frame bus."""
        await ws.accept()
        log.info("media.connect", session_id=session_id)
        bus = FrameBus(session_id)
        budgets = PhaseBudgets()
        consent_state = {"declined": False}

        def _consent_getter():
            return _read_consent_from_disk(orchestrator, session_id)

        phase_processor = PhaseProcessor(
            session_id,
            orchestrator.store,
            bus,
            budgets=budgets,
            consent_getter=_consent_getter,
        )
        intake_processor = IntakeProcessor(
            session_id,
            orchestrator.store,
            phase_getter=lambda: phase_processor.current_phase,
        )
        try:
            async with TwilioStream(ws) as twilio, HumeEVIParticipant(
                api_key=config.hume_api_key,
                config_id=config.hume_config_id,
                session_id=session_id,
            ) as coach:
                caller = TwilioCallerParticipant(twilio, session_id)
                await orchestrator.store.update_session(
                    session_id,
                    lambda s: _set_participants(s, [caller.config, coach.config]),
                )

                async def _on_decline() -> None:
                    """Mark the session for shutdown on consent decline."""
                    consent_state["declined"] = True
                    await bus.publish(
                        EndOfCall(
                            session_id=session_id,
                            reason="consent_decline",
                            ts=utcnow().timestamp(),
                        )
                    )

                _session_obj = Session.model_validate_json(
                    await orchestrator.store.read(session_id, "session.json")
                )
                caller_hash = _session_obj.phone_number_hash
                # Priority: MEMORY_MCP_URL → HONCHO_BASE_URL (self-hosted, local)
                #           → HONCHO_API_KEY (cloud) → NullCallerMemory
                # Self-hosted always wins over cloud when both are configured.
                if config.memory_mcp_url:
                    _memory: CallerMemory = MCPCallerMemory(config.memory_mcp_url)
                elif config.honcho_base_url:
                    _memory = HonchoCallerMemory(
                        workspace_id=config.honcho_workspace_id,
                        base_url=config.honcho_base_url,
                    )
                elif config.honcho_api_key:
                    _memory = HonchoCallerMemory(
                        api_key=config.honcho_api_key,
                        workspace_id=config.honcho_workspace_id,
                    )
                else:
                    _memory = NullCallerMemory()
                consent_gate = ConsentGate(
                    session_id,
                    orchestrator.store,
                    bus,
                    speaker=coach,
                    on_decline=_on_decline,
                    config=ConsentGateConfig(
                        prompt_timeout_seconds=config.consent_prompt_timeout_seconds,
                        reprompt_limit=config.consent_reprompt_limit,
                    ),
                    caller_hash=caller_hash,
                    memory=_memory,
                )
                outcome_probe = OutcomeProbe(
                    session_id,
                    orchestrator.store,
                    speaker=coach,
                    config=OutcomeProbeConfig(
                        response_timeout_seconds=config.outcome_response_timeout_seconds,
                        reprompt_limit=config.outcome_reprompt_limit,
                        prompt_lead_seconds=config.outcome_prompt_lead_seconds,
                        feedback_budget_seconds=budgets.feedback_seconds,
                    ),
                )
                await phase_processor.bootstrap()
                persona_swap = PersonaSwapCoordinator(
                    session_id,
                    orchestrator.store,
                    speaker=coach,
                )
                consent_task = asyncio.create_task(consent_gate.run(bus.subscribe()))
                outcome_task = asyncio.create_task(outcome_probe.run(bus.subscribe()))
                phase_task = asyncio.create_task(phase_processor.run(bus.subscribe()))
                intake_task = asyncio.create_task(intake_processor.run(bus.subscribe()))
                persona_swap_task = asyncio.create_task(
                    persona_swap.run(bus.subscribe())
                )
                transcript_task = asyncio.create_task(
                    TranscriptWriter(
                        session_id,
                        orchestrator.store,
                        phase_getter=lambda: phase_processor.current_phase,
                    ).run(bus.subscribe())
                )
                prosody_task = asyncio.create_task(
                    ProsodyWriter(session_id, orchestrator.store).run(bus.subscribe())
                )
                audio_task = asyncio.create_task(
                    AudioRecorder(session_id, orchestrator.store).run(bus.subscribe())
                )
                timing_task = asyncio.create_task(
                    TimingWriter(session_id, orchestrator.store).run(bus.subscribe())
                )
                telemetry_task = asyncio.create_task(
                    TelemetryLogger(
                        session_id,
                        orchestrator.store,
                        model=config.hume_config_id,
                        phase_getter=lambda: phase_processor.current_phase,
                    ).run(bus.subscribe())
                )
                assistant_task = asyncio.create_task(_pump_assistant_audio(caller, bus))
                coach_task = asyncio.create_task(coach.run(bus))
                try:
                    async for chunk in caller.audio_stream(bus):
                        if consent_state["declined"] or coach_task.done():
                            break
                        await coach.receive_audio(chunk)
                finally:
                    # Shield cleanup from external cancellation (e.g. starlette
                    # TestClient closes the anyio CancelScope immediately after
                    # the WebSocket disconnects, which would interrupt I/O in
                    # progress and leave session artifacts unwritten).
                    with anyio.CancelScope(shield=True):
                        await bus.aclose()
                        assistant_task.cancel()
                        coach_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await assistant_task
                        with suppress(asyncio.CancelledError):
                            await coach_task
                        await consent_task
                        await outcome_task
                        await phase_task
                        await intake_task
                        await persona_swap_task
                        await transcript_task
                        await prosody_task
                        await audio_task
                        await timing_task
                        await telemetry_task
                        if consent_state["declined"]:
                            await orchestrator.finalize(session_id, "partial")
        except WebSocketDisconnect:
            log.info("media.disconnect", session_id=session_id)


def _stream_twiml(config: RuntimeConfig, session_id: str) -> str:
    """Return the TwiML document that connects Twilio to our websocket."""
    ws_base = config.public_base_url.replace("https://", "wss://").replace("http://", "ws://")
    stream_url = f"{ws_base}/media/{session_id}"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<Stream url="{stream_url}"/>'
        "</Connect>"
        "</Response>"
    )


def _read_consent_from_disk(orchestrator: SessionOrchestrator, session_id: str):
    """Best-effort fallback to read consent from the manifest when no handle exists."""
    from rehearse.types import ConsentState, Session

    path = orchestrator.store.session_dir(session_id) / "session.json"
    try:
        return Session.model_validate_json(path.read_text()).consent
    except Exception:
        return ConsentState.PENDING


def _set_participants(session, participants):
    """Persist the live-call participant identities on the session manifest."""
    session.participants = participants
    return session


async def _pump_assistant_audio(caller: TwilioCallerParticipant, bus: FrameBus) -> None:
    """Forward assistant audio frames from the bus back to Twilio."""
    async for frame in bus.subscribe():
        if isinstance(frame, AudioChunk) and frame.speaker != Speaker.USER:
            await caller.receive_audio(frame.pcm16_16k)
