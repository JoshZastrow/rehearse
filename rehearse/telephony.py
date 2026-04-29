"""Handle Twilio webhooks, REST calls, and media streams for the runtime.

This file is the live runtime entrypoint for telephony. It owns inbound SMS and
voice webhooks, outbound call placement, call-status callbacks, and the live
Twilio media websocket that feeds audio into Hume.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Protocol

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

from rehearse.audio.twilio_stream import TwilioStream
from rehearse.bus import FrameBus
from rehearse.config import RuntimeConfig
from rehearse.frames import AudioChunk
from rehearse.intake import IntakeProcessor
from rehearse.phases import PhaseProcessor
from rehearse.services.hume_evi import HumeEVIClient
from rehearse.session import SessionOrchestrator, TriggerEvent, utcnow
from rehearse.types import Speaker
from rehearse.writers import AudioRecorder, ProsodyWriter, TelemetryLogger, TranscriptWriter

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
        phase_processor = PhaseProcessor(session_id, orchestrator.store, bus)
        intake_processor = IntakeProcessor(
            session_id,
            orchestrator.store,
            phase_getter=lambda: phase_processor.current_phase,
        )
        try:
            async with TwilioStream(ws) as twilio, HumeEVIClient(
                api_key=config.hume_api_key,
                config_id=config.hume_config_id,
                bus=bus,
                session_id=session_id,
            ) as hume:
                await phase_processor.bootstrap()
                phase_task = asyncio.create_task(phase_processor.run(bus.subscribe()))
                intake_task = asyncio.create_task(intake_processor.run(bus.subscribe()))
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
                telemetry_task = asyncio.create_task(
                    TelemetryLogger(
                        session_id,
                        orchestrator.store,
                        model=config.hume_config_id,
                        phase_getter=lambda: phase_processor.current_phase,
                    ).run(bus.subscribe())
                )
                assistant_task = asyncio.create_task(_pump_assistant_audio(twilio, bus))
                hume_task = asyncio.create_task(hume.run_event_loop())
                try:
                    async for chunk in twilio.inbound():
                        await hume.send_audio(chunk)
                        await bus.publish(
                            AudioChunk(
                                session_id=session_id,
                                speaker=Speaker.USER,
                                pcm16_16k=chunk,
                                ts=0.0,
                            )
                        )
                finally:
                    await bus.aclose()
                    assistant_task.cancel()
                    hume_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await assistant_task
                    with suppress(asyncio.CancelledError):
                        await hume_task
                    await phase_task
                    await intake_task
                    await transcript_task
                    await prosody_task
                    await audio_task
                    await telemetry_task
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


async def _pump_assistant_audio(twilio: TwilioStream, bus: FrameBus) -> None:
    """Forward assistant audio frames from the bus back to Twilio."""
    async for frame in bus.subscribe():
        if isinstance(frame, AudioChunk) and frame.speaker != Speaker.USER:
            await twilio.send(frame.pcm16_16k)
