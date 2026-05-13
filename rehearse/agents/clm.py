"""Serve Hume's custom language model webhook with OpenAI-style SSE chunks.

This file accepts CLM requests from Hume, chooses a responder, and streams the
assistant text back in the SSE shape Hume expects from an OpenAI-compatible
`/chat/completions` endpoint.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from typing import Any, Protocol

import anthropic
import structlog
from anthropic import AsyncAnthropic
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

log = structlog.get_logger(__name__)

# Spoken when the LLM fails mid-call. Plain language, neutral, short. The
# point is to keep the SSE stream well-formed so Hume's session stays alive
# rather than dropping with a clean 1000 OK close.
CLM_FALLBACK_LINE = "Sorry, I had a brief glitch — what were you saying?"

from rehearse.config import RuntimeConfig
from rehearse.session import utcnow
from rehearse.storage import LocalFilesystemStore
from rehearse.types import Phase, Session


class CLMMessage(BaseModel):
    """One message from Hume's CLM payload."""

    model_config = ConfigDict(extra="allow")

    role: str | None = None
    content: str | None = None
    type: str | None = None
    models: dict[str, Any] = Field(default_factory=dict)
    time: dict[str, int] | None = None


class CLMChatRequest(BaseModel):
    """Body for an OpenAI-style streaming chat completions request."""

    model_config = ConfigDict(extra="allow")

    messages: list[CLMMessage] = Field(default_factory=list)
    model: str | None = None
    stream: bool = True


class CLMResponder(Protocol):
    """Small interface for anything that can stream assistant text."""

    async def stream_reply(
        self,
        *,
        session_id: str | None,
        role: str,
        request: CLMChatRequest,
    ) -> AsyncIterator[str]:
        """Yield assistant text fragments for one CLM turn."""


class ScriptedCLMResponder:
    """Fallback responder used when no upstream LLM key is configured."""

    def __init__(self, store: LocalFilesystemStore) -> None:
        """Store the session store used to resolve persona context."""
        self._store = store

    async def stream_reply(
        self,
        *,
        session_id: str | None,
        role: str,
        request: CLMChatRequest,
    ) -> AsyncIterator[str]:
        """Yield a short deterministic coaching reply for local testing."""
        session = await _load_session(session_id, self._store)
        last_user_text = _last_user_text(request.messages) or "Tell me what happened."
        reply = _scripted_reply(role=role, last_user_text=last_user_text, session=session)
        for chunk in _chunk_text(reply):
            yield chunk


async def validate_anthropic_credentials(client: AsyncAnthropic, model: str) -> None:
    """Send a cheap token-count request to verify the key works at startup.

    Anthropic's `messages.count_tokens` does not consume model credits and
    raises `AuthenticationError` for an invalid key. Boot fails loudly here
    rather than silently killing every CLM turn during a live call. Other
    transient errors (network, 5xx) are logged and tolerated — the runtime
    can still boot, the in-call fallback line absorbs them.
    """
    try:
        await client.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
        )
    except anthropic.AuthenticationError as exc:
        raise RuntimeError(
            f"Anthropic authentication failed for model {model!r}: {exc}. "
            "Check ANTHROPIC_API_KEY in the environment."
        ) from exc
    except Exception as exc:
        log.warning(
            "clm.anthropic_validation_skipped",
            error=str(exc),
            error_type=type(exc).__name__,
        )


def build_clm_responder(config: RuntimeConfig) -> CLMResponder:
    """Return the live CLM responder chosen from the runtime config.

    Builds the new NewCLMResponder (transport + router + memory) when an
    Anthropic API key is configured. Falls back to ScriptedCLMResponder for
    local development without credentials.
    """
    store = LocalFilesystemStore(root=config.session_root, public_base_url=config.public_base_url)
    if config.anthropic_api_key:
        from rehearse.agents.registry import AgentRegistry
        from rehearse.agents.roles.character import CharacterAgent, FemaleCharacterAgent, MaleCharacterAgent
        from rehearse.agents.roles.feedback import FeedbackCoachAgent
        from rehearse.agents.roles.intake import IntakeCoachAgent
        from anthropic import AsyncAnthropic

        from rehearse.agents.router import IntakeAwareRouter
        from rehearse.memory import (
            HonchoCallerMemory,
            MCPCallerMemory,
            NullCallerMemory,
        )
        from rehearse.memory_manager import MemoryManager
        from rehearse.new_clm_responder import NewCLMResponder
        from rehearse.transports.anthropic import AnthropicTransport

        if config.memory_mcp_url:
            _provider = MCPCallerMemory(config.memory_mcp_url)
        elif config.honcho_base_url:
            _provider = HonchoCallerMemory(
                workspace_id=config.honcho_workspace_id,
                base_url=config.honcho_base_url,
            )
        elif config.honcho_api_key:
            _provider = HonchoCallerMemory(
                api_key=config.honcho_api_key,
                workspace_id=config.honcho_workspace_id,
            )
        else:
            _provider = NullCallerMemory()

        llm_client = AsyncAnthropic(api_key=config.anthropic_api_key)
        memory = MemoryManager(_provider)
        transport = AnthropicTransport(api_key=config.anthropic_api_key)
        registry = AgentRegistry()
        registry.register(IntakeCoachAgent(memory))
        registry.register(CharacterAgent(memory))
        registry.register(MaleCharacterAgent(memory))
        registry.register(FemaleCharacterAgent(memory))
        registry.register(FeedbackCoachAgent(memory))
        router = IntakeAwareRouter(registry, _provider, llm_client=llm_client)
        return NewCLMResponder(
            transport=transport,
            router=router,
            memory=memory,
            store=store,
            model=config.anthropic_model,
        )
    return ScriptedCLMResponder(store=store)


class _SessionRegistry:
    """Track one cancellation event per live session to stop orphaned streams.

    When Hume sends a new CLM request because the caller interrupted, the
    previous Claude generation is still running.  This registry lets the new
    request signal the old stream to stop before it starts its own.
    """

    def __init__(self) -> None:
        self._events: dict[str, asyncio.Event] = {}

    def begin(self, session_id: str) -> asyncio.Event:
        """Cancel any running stream for this session; return a fresh event."""
        old = self._events.get(session_id)
        if old is not None:
            old.set()
        event = asyncio.Event()
        self._events[session_id] = event
        return event

    def done(self, session_id: str, event: asyncio.Event) -> None:
        """Remove the entry when a stream finishes normally."""
        if self._events.get(session_id) is event:
            self._events.pop(session_id, None)


def mount_clm_routes(app: FastAPI, responder: CLMResponder, config: RuntimeConfig) -> None:
    """Register the OpenAI-compatible CLM webhook endpoints on the app."""
    store = LocalFilesystemStore(root=config.session_root, public_base_url=config.public_base_url)
    registry = _SessionRegistry()

    @app.post("/chat/completions")
    async def chat_completions(
        payload: CLMChatRequest,
        custom_session_id: str | None = Query(default=None),
        role: str | None = Query(default=None),
        authorization: str | None = Header(default=None),
    ) -> Response:
        """Handle Hume's recommended SSE CLM endpoint."""
        await _verify_clm_auth(config, authorization)
        resolved_role = await _resolve_role(role=role, session_id=custom_session_id, store=store)
        cancel_event = registry.begin(custom_session_id) if custom_session_id else None
        return await _handle_clm_request(
            payload=payload,
            responder=responder,
            session_id=custom_session_id,
            role=resolved_role,
            registry=registry,
            cancel_event=cancel_event,
        )

    @app.post("/hume/clm/{session_id}")
    async def hume_clm(
        session_id: str,
        payload: CLMChatRequest,
        role: str | None = Query(default=None),
        authorization: str | None = Header(default=None),
    ) -> Response:
        """Handle the older path-based CLM endpoint from the runtime spec."""
        await _verify_clm_auth(config, authorization)
        resolved_role = await _resolve_role(role=role, session_id=session_id, store=store)
        cancel_event = registry.begin(session_id)
        return await _handle_clm_request(
            payload=payload,
            responder=responder,
            session_id=session_id,
            role=resolved_role,
            registry=registry,
            cancel_event=cancel_event,
        )


async def _handle_clm_request(
    *,
    payload: CLMChatRequest,
    responder: CLMResponder,
    session_id: str | None,
    role: str,
    registry: _SessionRegistry | None = None,
    cancel_event: asyncio.Event | None = None,
) -> Response:
    """Return either an SSE stream or one fully buffered OpenAI-style response."""
    model = payload.model or role
    if payload.stream:
        return StreamingResponse(
            _stream_openai_chunks(
                responder.stream_reply(
                    session_id=session_id,
                    role=role,
                    request=payload,
                ),
                model=model,
                session_id=session_id,
                cancel_event=cancel_event,
                registry=registry,
            ),
            media_type="text/event-stream",
        )

    text = "".join(
        [
            chunk
            async for chunk in responder.stream_reply(
                session_id=session_id,
                role=role,
                request=payload,
            )
        ]
    )
    response_id = _response_id(session_id)
    return JSONResponse(
        {
            "id": response_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "system_fingerprint": session_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }
    )


async def _verify_clm_auth(config: RuntimeConfig, authorization: str | None) -> None:
    """Reject requests when a CLM bearer secret is configured and missing."""
    if not config.hume_clm_secret:
        return
    expected = f"Bearer {config.hume_clm_secret}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="invalid clm token")


async def _stream_openai_chunks(
    chunks: AsyncIterator[str],
    *,
    model: str,
    session_id: str | None,
    cancel_event: asyncio.Event | None = None,
    registry: _SessionRegistry | None = None,
) -> AsyncIterator[str]:
    """Wrap plain text chunks into OpenAI-compatible SSE events.

    Stops early and closes the upstream generator if the session's cancel event
    is set — which happens when a new CLM request arrives for the same session
    (i.e. the caller interrupted the coach mid-response).
    """
    response_id = _response_id(session_id)
    created = int(time.time())
    cancelled = False
    yield _sse_data(
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": session_id,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
    )
    try:
        async for chunk in chunks:
            if cancel_event is not None and cancel_event.is_set():
                log.info("clm.stream_cancelled", session_id=session_id)
                cancelled = True
                break
            yield _sse_data(
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "system_fingerprint": session_id,
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                }
            )
    finally:
        if cancelled and hasattr(chunks, "aclose"):
            await chunks.aclose()
        if registry is not None and session_id is not None and cancel_event is not None:
            registry.done(session_id, cancel_event)
    yield _sse_data(
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": session_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    yield "data: [DONE]\n\n"


def _response_id(session_id: str | None) -> str:
    """Return a stable-looking response id for one streamed assistant turn."""
    suffix = session_id or str(int(time.time() * 1000))
    return f"chatcmpl-{suffix}"


def _sse_data(payload: dict[str, Any]) -> str:
    """Serialize one SSE `data:` event."""
    return f"data: {json.dumps(payload)}\n\n"


def _last_user_text(messages: Iterable[CLMMessage]) -> str | None:
    """Return the most recent user message text from the CLM history."""
    for message in reversed(list(messages)):
        role = message.role if message.role in {"user", "assistant"} else (
            "assistant" if message.type == "assistant_message" else "user"
        )
        if role == "user" and message.content:
            return message.content.strip()
    return None


def _scripted_reply(*, role: str, last_user_text: str, session: Session | None) -> str:
    """Return a short fallback reply when no upstream model is configured."""
    if role == "character":
        relationship = (
            session.persona.relationship if session and session.persona else "the other person"
        )
        return (
            f"As {relationship}, I need you to say that more directly, because right now "
            "it still sounds like you're circling the point."
        )
    return (
        "Let's make this concrete. In one sentence, what do you most want to achieve, "
        f"starting from this: {last_user_text}"
    )


def _chunk_text(text: str, *, words_per_chunk: int = 8) -> list[str]:
    """Split one spoken reply into a few larger chunks for SSE streaming."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    for index in range(0, len(words), words_per_chunk):
        piece = " ".join(words[index : index + words_per_chunk])
        if index + words_per_chunk < len(words):
            piece = f"{piece} "
        chunks.append(piece)
    return chunks


async def _load_session(session_id: str | None, store: LocalFilesystemStore) -> Session | None:
    """Load the stored session manifest when a session id is available."""
    if not session_id:
        return None
    try:
        payload = await store.read(session_id, "session.json")
    except FileNotFoundError:
        return None
    return Session.model_validate_json(payload)


async def _resolve_role(
    *,
    role: str | None,
    session_id: str | None,
    store: LocalFilesystemStore,
) -> str:
    """Return an explicit role override or infer one from the live session phase."""
    if role in {"coach", "character", "feedback_coach"}:
        return role
    session = await _load_session(session_id, store)
    if session is None:
        return "coach"
    phase = _current_phase(session)
    if phase == Phase.PRACTICE:
        return "character"
    if phase == Phase.FEEDBACK:
        return "feedback_coach"
    return "coach"


def _current_phase(session: Session) -> Phase:
    """Return the currently active phase for a stored session manifest."""
    for timing in reversed(session.phase_timings):
        if timing.ended_at is None:
            return timing.phase
    return Phase.INTAKE
