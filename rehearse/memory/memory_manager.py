"""MemoryManager — lifecycle orchestration over a CallerMemoryProvider.

Agents call prefetch() before building their system prompt.
CLMResponder / telephony.py call store_session() at call end.
ConsentGate uses has_prior_consent() and record_consent() directly.

All methods fail open: errors are logged as warnings, never raised.
The empty-caller-hash guard on prefetch() avoids a network round-trip
for sessions with no phone number (test or eval calls).
"""

from __future__ import annotations

from typing import Any

import structlog

from rehearse.memory.memory import CallerMemory

log = structlog.get_logger(__name__)


class MemoryManager:
    """Thin delegation wrapper with failure isolation."""

    def __init__(self, provider: CallerMemory) -> None:
        self._provider = provider

    # ------------------------------------------------------------------
    # Semantic recall (read path — agents call this before the LLM)
    # ------------------------------------------------------------------

    async def prefetch(self, caller_hash: str, query: str) -> str:
        """Return synthesized memory context for this caller and query.

        Returns "" for an empty caller_hash or on any provider error.
        """
        if not caller_hash:
            return ""
        try:
            return await self._provider.prefetch(caller_hash, query)
        except Exception as exc:
            log.warning("memory_manager.prefetch.failed", query=query[:40], error=str(exc))
            return ""

    # ------------------------------------------------------------------
    # Session storage (write path — called at call end by telephony.py)
    # ------------------------------------------------------------------

    async def store_session(
        self,
        caller_hash: str,
        messages: list[dict],
        **kwargs: Any,
    ) -> None:
        """Persist the call transcript to the memory backend."""
        try:
            await self._provider.store_session(caller_hash, messages, **kwargs)
        except Exception as exc:
            log.warning(
                "memory_manager.store_session.failed",
                caller_hash=caller_hash[:8] if caller_hash else "",
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Consent (used by ConsentGate)
    # ------------------------------------------------------------------

    async def has_prior_consent(self, caller_hash: str) -> bool:
        return await self._provider.has_prior_consent(caller_hash)

    async def record_consent(self, caller_hash: str) -> None:
        await self._provider.record_consent(caller_hash)

    # ------------------------------------------------------------------
    # Per-topic agent preference (used by IntakeCoachAgent + IntakeAwareRouter)
    # ------------------------------------------------------------------

    async def get_agent_preference(self, caller_hash: str, topic_category: str) -> str | None:
        """Return stored gender preference for a topic, or None."""
        if not caller_hash:
            return None
        try:
            return await self._provider.get_agent_preference(caller_hash, topic_category)
        except Exception as exc:
            log.warning(
                "memory_manager.get_agent_preference.failed",
                topic=topic_category,
                error=str(exc),
            )
            return None
