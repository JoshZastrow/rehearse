"""Handle Twilio webhooks, REST calls, and media streams for the runtime.

This file is the live runtime entrypoint for telephony. It owns inbound SMS and
voice webhooks, outbound call placement, call-status callbacks, and the live
Twilio media websocket that feeds audio into Hume.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

import httpx

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

from rehearse.audio.twilio_stream import TwilioCallerParticipant, TwilioStream
from rehearse.backends.factory import create_backend
from rehearse.config import RuntimeConfig
from rehearse.session.conversation import run_session
from rehearse.memory.memory import HonchoCallerMemory, MCPCallerMemory, NullCallerMemory
from rehearse.phases.phases import PhaseBudgets
from rehearse.session.session import SessionOrchestrator, TriggerEvent, utcnow
from rehearse.types import ParticipantConfig, Session

log = structlog.get_logger(__name__)


import time as _time


def _health_url(endpoint: str) -> str:
    return endpoint.replace("wss://", "https://").replace("ws://", "http://").replace("/ws", "/health")


async def _inference_ready(endpoint: str) -> bool:
    """Return True if the inference server responds 200 on /health."""
    if not endpoint:
        return True
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(_health_url(endpoint))
        return resp.status_code == 200
    except Exception:
        return False


async def _prewarm_modal(endpoint: str) -> None:
    """Trigger the inference server cold-start clock by hitting /health."""
    if not endpoint:
        return
    url = _health_url(endpoint)
    log.info("Prewarming inference server", url=url)
    t0 = _time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.get(url)
        log.info("Inference server responding", elapsed=f"{_time.monotonic() - t0:.1f}s")
    except Exception as exc:
        log.debug("Prewarm request failed (server may still be booting)", error=str(exc))


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
        log.info("twilio.sms", session_id=handle.session_id, from_number=From)

        voice_url = f"{config.public_base_url}/twilio/voice?session_id={handle.session_id}"
        status_url = f"{config.public_base_url}/twilio/status?session_id={handle.session_id}"

        async def _place() -> None:
            """Place the outbound call and attach its SID to the session."""
            await _prewarm_modal(config.interactive_modal_endpoint)
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
        background: BackgroundTasks,
        From: str = Form(...),
        CallSid: str = Form(""),
    ) -> Response:
        """Start a new session from a direct inbound phone call."""
        await _validate(request)
        background.add_task(_prewarm_modal, config.interactive_modal_endpoint)
        trigger = TriggerEvent(from_number=From, body="<inbound-call>", received_at=utcnow())
        handle = await orchestrator.start(trigger)
        if CallSid:
            await orchestrator.attach_call(handle.session_id, CallSid)
        log.info("twilio.voice.inbound", session_id=handle.session_id, call_sid=CallSid)
        if config.interactive_modal_endpoint:
            twiml = _hold_twiml(config, handle.session_id, first=True, attempt=0)
        else:
            twiml = _stream_twiml(config, handle.session_id)
        return PlainTextResponse(
            twiml,
            media_type="application/xml",
        )

    @app.post("/twilio/voice/wait")
    async def twilio_voice_wait(request: Request, session_id: str, attempt: int = 0) -> Response:
        """Poll Modal until the inference container is ready, then connect the stream.

        Twilio calls this repeatedly via <Redirect> while the Modal GPU cold-starts.
        Each iteration either connects the stream (Modal ready), pauses 3s and loops,
        or hangs up after 90s (30 attempts × 3s).
        """
        await _validate(request)
        if attempt >= 30:
            log.warning("Inference server not ready after 90s — hanging up", session_id=session_id)
            return PlainTextResponse(
                '<?xml version="1.0" encoding="UTF-8"?>'
                "<Response>"
                '<Say voice="alice">Sorry, I was unable to connect. Please try again in a moment.</Say>'
                "<Hangup/>"
                "</Response>",
                media_type="application/xml",
            )
        if await _inference_ready(config.interactive_modal_endpoint):
            log.info("Inference server ready — connecting stream", session_id=session_id, attempt=attempt)
            return PlainTextResponse(_stream_twiml(config, session_id), media_type="application/xml")
        log.debug("Inference server not yet ready, retrying...", session_id=session_id, attempt=attempt)
        return PlainTextResponse(_hold_twiml(config, session_id, first=False, attempt=attempt + 1), media_type="application/xml")

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
        """Bridge the live Twilio media websocket to the conversation pipeline."""
        await ws.accept()
        log.info("media.connect", session_id=session_id)
        try:
            _session_obj = Session.model_validate_json(
                await orchestrator.store.read(session_id, "session.json")
            )
            caller_hash = _session_obj.phone_number_hash

            if config.memory_mcp_url:
                from rehearse.memory.memory import MCPCallerMemory
                _memory = MCPCallerMemory(config.memory_mcp_url)
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

            _llm_client = None
            if config.litellm_base_url:
                from rehearse.transports.openai_compat import OpenAICompatLLMClient
                _llm_client = OpenAICompatLLMClient(
                    base_url=config.litellm_base_url,
                    api_key=config.litellm_api_key or "local",
                )
            elif config.anthropic_api_key:
                from anthropic import AsyncAnthropic
                _llm_client = AsyncAnthropic(api_key=config.anthropic_api_key)

            async with TwilioStream(ws) as twilio, create_backend(config, store=orchestrator.store) as backend:
                caller = TwilioCallerParticipant(twilio, session_id)
                await orchestrator.store.update_session(
                    session_id,
                    lambda s: _set_participants(
                        s, [caller.config, _backend_participant_config(session_id, config)]
                    ),
                )
                await run_session(
                    session_id,
                    caller,
                    backend,
                    store=orchestrator.store,
                    memory=_memory,
                    caller_hash=caller_hash,
                    budgets=PhaseBudgets(),
                    skip_consent=False,
                    enable_consent=config.enable_consent,
                    consent_timeout_seconds=config.consent_prompt_timeout_seconds,
                    consent_reprompt_limit=config.consent_reprompt_limit,
                    outcome_timeout_seconds=config.outcome_response_timeout_seconds,
                    outcome_reprompt_limit=config.outcome_reprompt_limit,
                    outcome_lead_seconds=config.outcome_prompt_lead_seconds,
                    model_id=config.hume_config_id,
                    llm_client=_llm_client,
                    on_consent_declined=lambda: orchestrator.finalize(session_id, "partial"),
                )
        except WebSocketDisconnect:
            log.info("media.disconnect", session_id=session_id)


def _hold_twiml(config: RuntimeConfig, session_id: str, *, first: bool, attempt: int) -> str:
    """TwiML that says a hold message (first call) or pauses silently, then redirects back."""
    wait_url = f"{config.public_base_url}/twilio/voice/wait?session_id={session_id}&amp;attempt={attempt}"
    hold = (
        '<Say voice="alice">Just a moment while I connect you.</Say>'
        if first
        else '<Pause length="3"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<Response>{hold}"
        f'<Redirect method="POST">{wait_url}</Redirect>'
        "</Response>"
    )


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


def _set_participants(session: Session, participants: list) -> Session:
    """Persist the live-call participant identities on the session manifest."""
    session.participants = participants
    return session


def _backend_participant_config(session_id: str, config: RuntimeConfig) -> ParticipantConfig:
    """Return a ParticipantConfig for the active conversation backend."""
    return ParticipantConfig(
        participant_id=session_id,
        role="coach",
        backend=config.backend_type,
    )
