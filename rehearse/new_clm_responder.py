"""NewCLMResponder — pure orchestration. Replaces AnthropicCLMResponder.

This class wires transport + router + memory into the per-turn CLM lifecycle:
  1. Load session from disk
  2. Route to the correct agent (via PhaseRouter)
  3. Recall cross-session context (via agent.recall → memory.prefetch)
  4. Build system prompt (via agent.system_prompt)
  5. Stream response (via transport.stream)
  6. Call agent.after_turn

It knows nothing about Anthropic's API, Honcho, or any other provider.
Named 'NewCLMResponder' temporarily to avoid import conflict with the existing
CLMResponder Protocol in clm.py. Will be renamed to CLMResponder in Cycle 6
when AnthropicCLMResponder is deleted.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from datetime import datetime

import structlog

from rehearse.agents.clm import CLMChatRequest, CLMMessage, CLM_FALLBACK_LINE
from rehearse.agents.registry import AgentRegistry
from rehearse.agents.roles.base import wrap_memory_context
from rehearse.agents.router import AgentRouter
from rehearse.agents.timecard import build_time_card, render_time_card
from rehearse.memory_manager import MemoryManager
from rehearse.session.session import utcnow
from rehearse.storage import LocalFilesystemStore
from rehearse.transports.base import LLMTransport
from rehearse.types import Session

log = structlog.get_logger(__name__)


class NewCLMResponder:
    """Orchestrate one CLM turn: route → recall → prompt → stream → after_turn."""

    def __init__(
        self,
        transport: LLMTransport,
        router: AgentRouter,
        memory: MemoryManager,
        store: LocalFilesystemStore,
        *,
        model: str,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._transport = transport
        self._router = router
        self._memory = memory
        self._store = store
        self._model = model
        self._clock = clock

    async def stream_reply(
        self,
        *,
        session_id: str | None,
        request: CLMChatRequest,
        role: str | None = None,  # accepted for backward compat with mount_clm_routes
    ) -> AsyncIterator[str]:
        """Drive the full per-turn lifecycle and yield text chunks."""
        session = await _load_session(session_id, self._store)

        # 1. Route — which agent handles this turn?
        agent = await self._router.route(session or _empty_session(), role_hint=role)

        # 2. Recall — pull cross-session context before the LLM call
        memory_context = ""
        if session:
            try:
                raw = await agent.recall(session)
                memory_context = wrap_memory_context(raw)
            except Exception as exc:
                log.warning("clm_responder.recall_failed", agent=agent.name, error=str(exc))

        # 3. System prompt
        system_prompt = agent.system_prompt(session or _empty_session(), memory_context)
        system_blocks: list[dict] = [
            {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
        ]
        if session and session.phase_timings:
            try:
                card = build_time_card(session, now=self._clock())
                if card:
                    system_blocks.append({"type": "text", "text": render_time_card(card)})
            except Exception:
                pass

        # 4. Convert messages
        messages = self._transport.convert_messages(request.messages)
        if not messages:
            messages = [{"role": "user", "content": "Greet the caller and start the session."}]

        user_text = _last_user_text(request.messages) or ""
        full_response: list[str] = []

        # 5. Stream
        try:
            async for text in self._transport.stream(
                system_blocks=system_blocks,
                messages=messages,
                model=self._model,
            ):
                full_response.append(text)
                yield text
        except Exception as exc:
            log.warning(
                "clm_responder.stream_failed",
                agent=agent.name,
                session_id=session_id,
                error=str(exc),
            )
            yield CLM_FALLBACK_LINE
            return

        # 6. After-turn hook
        if session and full_response:
            try:
                await agent.after_turn(session, user_text, "".join(full_response))
            except Exception as exc:
                log.warning("clm_responder.after_turn_failed", agent=agent.name, error=str(exc))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _load_session(session_id: str | None, store: LocalFilesystemStore) -> Session | None:
    if not session_id:
        return None
    try:
        payload = await store.read(session_id, "session.json")
        return Session.model_validate_json(payload)
    except FileNotFoundError:
        return None


def _empty_session() -> Session:
    return Session(created_at=utcnow())


def _last_user_text(messages: list[CLMMessage]) -> str | None:
    for message in reversed(messages):
        role = message.role
        if message.type == "assistant_message":
            continue
        if role in {None, "user"} and message.content:
            return message.content.strip()
    return None
