"""Telephony Gateway — all Twilio interaction.

Phase R1 surface:
  - POST /twilio/sms              inbound SMS trigger (places outbound call)
  - POST /twilio/voice            TwiML for the outbound call placed by /sms
  - POST /twilio/voice/inbound    direct dial-in trigger (mints session inline)
  - POST /twilio/status           call status callbacks (finalize)
  - WS   /media/{id}              Twilio Media Streams duplex (no-op accept/log in R1)

Outbound calls and SMS are placed via the Twilio REST client.
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
from rehearse.services.hume_evi import HumeEVIClient
from rehearse.session import SessionOrchestrator, TriggerEvent, utcnow
from rehearse.types import Speaker

log = structlog.get_logger(__name__)


class TelephonyClient(Protocol):
    async def place_call(self, to: str, callback_url: str, status_callback: str) -> str: ...
    async def send_sms(self, to: str, body: str) -> str: ...


class TwilioRestClient:
    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config
        self._client = TwilioClient(config.twilio_account_sid, config.twilio_auth_token)

    async def place_call(self, to: str, callback_url: str, status_callback: str) -> str:
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
    validator = RequestValidator(config.twilio_auth_token)

    async def _validate(request: Request) -> None:
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
        await _validate(request)
        trigger = TriggerEvent(from_number=From, body=Body, received_at=utcnow())
        handle = await orchestrator.start(trigger)

        voice_url = f"{config.public_base_url}/twilio/voice?session_id={handle.session_id}"
        status_url = f"{config.public_base_url}/twilio/status?session_id={handle.session_id}"

        async def _place() -> None:
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
        await _validate(request)
        log.info("twilio.voice", session_id=session_id)
        return PlainTextResponse(_stream_twiml(config, session_id), media_type="application/xml")

    @app.post("/twilio/voice/inbound")
    async def twilio_voice_inbound(
        request: Request,
        From: str = Form(...),
        CallSid: str = Form(""),
    ) -> Response:
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
        await ws.accept()
        log.info("media.connect", session_id=session_id)
        bus = FrameBus(session_id)
        try:
            async with TwilioStream(ws) as twilio, HumeEVIClient(
                api_key=config.hume_api_key,
                config_id=config.hume_config_id,
                bus=bus,
                session_id=session_id,
            ) as hume:
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
        except WebSocketDisconnect:
            log.info("media.disconnect", session_id=session_id)


def _stream_twiml(config: RuntimeConfig, session_id: str) -> str:
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
    async for frame in bus.subscribe():
        if isinstance(frame, AudioChunk) and frame.speaker != Speaker.USER:
            await twilio.send(frame.pcm16_16k)
