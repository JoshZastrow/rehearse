"""Serve Hume's custom language model webhook with OpenAI-style SSE chunks.

This file accepts CLM requests from Hume, chooses a responder, and streams the
assistant text back in the SSE shape Hume expects from an OpenAI-compatible
`/chat/completions` endpoint.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Iterable
from typing import Any, Protocol

from anthropic import AsyncAnthropic
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from rehearse.config import RuntimeConfig
from rehearse.personas import character_system_prompt, coach_system_prompt
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


class AnthropicCLMResponder:
    """Wrap Claude so Hume can use it as the live conversation brain."""

    def __init__(self, api_key: str, model: str, store: LocalFilesystemStore) -> None:
        """Store Anthropic credentials and create the async client lazily."""
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model
        self._store = store

    async def stream_reply(
        self,
        *,
        session_id: str | None,
        role: str,
        request: CLMChatRequest,
    ) -> AsyncIterator[str]:
        """Yield text chunks from Anthropic's streaming messages API."""
        session = await _load_session(session_id, self._store)
        system = _system_prompt_for_role(role, session)
        messages = _anthropic_messages(request.messages)
        if not messages:
            messages = [{"role": "user", "content": "Greet the caller and start the coaching."}]
        if session_id:
            system = f"{system}\n\nSession ID: {session_id}"
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=512,
            temperature=0.4,
            system=system,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                if text:
                    yield text


def build_clm_responder(config: RuntimeConfig) -> CLMResponder:
    """Return the live CLM responder chosen from the runtime config."""
    store = LocalFilesystemStore(root=config.session_root, public_base_url=config.public_base_url)
    if config.anthropic_api_key:
        return AnthropicCLMResponder(
            api_key=config.anthropic_api_key,
            model=config.anthropic_model,
            store=store,
        )
    return ScriptedCLMResponder(store=store)


def mount_clm_routes(app: FastAPI, responder: CLMResponder, config: RuntimeConfig) -> None:
    """Register the OpenAI-compatible CLM webhook endpoints on the app."""
    store = LocalFilesystemStore(root=config.session_root, public_base_url=config.public_base_url)

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
        return await _handle_clm_request(
            payload=payload,
            responder=responder,
            session_id=custom_session_id,
            role=resolved_role,
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
        return await _handle_clm_request(
            payload=payload,
            responder=responder,
            session_id=session_id,
            role=resolved_role,
        )


async def _handle_clm_request(
    *,
    payload: CLMChatRequest,
    responder: CLMResponder,
    session_id: str | None,
    role: str,
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
) -> AsyncIterator[str]:
    """Wrap plain text chunks into OpenAI-compatible SSE events."""
    response_id = _response_id(session_id)
    created = int(time.time())
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
    async for chunk in chunks:
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


def _anthropic_messages(messages: Iterable[CLMMessage]) -> list[dict[str, str]]:
    """Convert Hume's message history into Anthropic's message format."""
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = _message_role(message)
        content = _message_content(message)
        if not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def _message_role(message: CLMMessage) -> str:
    """Return the normalized chat role for one CLM message."""
    if message.role in {"user", "assistant"}:
        return message.role
    if message.type == "assistant_message":
        return "assistant"
    return "user"


def _message_content(message: CLMMessage) -> str:
    """Return the text content for one CLM message, with light prosody context."""
    text = (message.content or "").strip()
    if not text:
        return ""
    prosody_scores = {}
    if isinstance(message.models, dict):
        prosody_scores = message.models.get("prosody", {}).get("scores", {})
    if not prosody_scores:
        return text
    top_emotions = sorted(prosody_scores.items(), key=lambda item: item[1], reverse=True)[:3]
    emotion_summary = ", ".join(f"{name}={score:.2f}" for name, score in top_emotions)
    return f"{text}\n\nProsody cues: {emotion_summary}"


def _last_user_text(messages: Iterable[CLMMessage]) -> str | None:
    """Return the most recent user message text from the CLM history."""
    for message in reversed(list(messages)):
        if _message_role(message) == "user" and message.content:
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


def _system_prompt_for_role(role: str, session: Session | None) -> str:
    """Return the correct system prompt for the requested CLM role."""
    if role == "character":
        if session and session.persona is not None:
            return character_system_prompt(session.persona)
        return character_system_prompt("Be the other person in the conversation.")
    return coach_system_prompt()


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
    if role in {"coach", "character"}:
        return role
    session = await _load_session(session_id, store)
    if session is None:
        return "coach"
    phase = _current_phase(session)
    if phase == Phase.PRACTICE:
        return "character"
    return "coach"


def _current_phase(session: Session) -> Phase:
    """Return the currently active phase for a stored session manifest."""
    for timing in reversed(session.phase_timings):
        if timing.ended_at is None:
            return timing.phase
    return Phase.INTAKE
