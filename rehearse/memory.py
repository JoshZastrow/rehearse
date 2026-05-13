"""Caller memory interface and implementations for stateful Rehearse agents.

Each agent in the call (consent, intake, coach, feedback) can use the
CallerMemory protocol to recall facts about a caller across sessions.

v1 stores one fact: whether the caller has previously granted consent.

Implementations
---------------
NullCallerMemory       No-op. Always returns no prior consent. Used when
                       HONCHO_API_KEY is not configured.

InMemoryCallerMemory   In-process set. Shared across calls in the same process.
                       Used in tests and local development.

HonchoCallerMemory     Honcho-backed. Persists caller facts to the Honcho
                       cloud API keyed by phone_number_hash. Used in
                       production when HONCHO_API_KEY is set.

MCPCallerMemory        MCP-protocol client. Connects to any MCP server that
                       exposes has_prior_consent and record_consent tools.
                       Used when MEMORY_MCP_URL is set; lets Honcho (or any
                       other backend) be swapped without changing call code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CallerMemory(Protocol):
    """Standard memory interface for Rehearse agents.

    Any class that provides these methods satisfies this protocol.
    No inheritance required.
    """

    async def has_prior_consent(self, caller_hash: str) -> bool:
        """Return True if this caller has previously granted consent."""
        ...

    async def record_consent(self, caller_hash: str) -> None:
        """Persist that this caller has granted consent."""
        ...

    async def get_recent_intakes(self, caller_hash: str, n: int = 3) -> list[str]:
        """Return up to n situation strings from this caller's most recent intakes.

        Newest first. Returns [] for a first-time caller.
        """
        ...

    async def record_intake(self, caller_hash: str, situation: str) -> None:
        """Persist one intake situation for future retrieval."""
        ...


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


class NullCallerMemory:
    """No-op memory. Every caller is treated as new. Default when no backend is configured."""

    async def has_prior_consent(self, caller_hash: str) -> bool:
        return False

    async def record_consent(self, caller_hash: str) -> None:
        pass

    async def get_recent_intakes(self, caller_hash: str, n: int = 3) -> list[str]:
        return []

    async def record_intake(self, caller_hash: str, situation: str) -> None:
        pass


class InMemoryCallerMemory:
    """In-process memory. Used in tests."""

    def __init__(self) -> None:
        self._consented: set[str] = set()
        self._intakes: dict[str, list[str]] = {}  # caller_hash → [newest, ..., oldest]

    async def has_prior_consent(self, caller_hash: str) -> bool:
        return caller_hash in self._consented

    async def record_consent(self, caller_hash: str) -> None:
        self._consented.add(caller_hash)

    async def get_recent_intakes(self, caller_hash: str, n: int = 3) -> list[str]:
        return self._intakes.get(caller_hash, [])[:n]

    async def record_intake(self, caller_hash: str, situation: str) -> None:
        self._intakes.setdefault(caller_hash, []).insert(0, situation)


class HonchoCallerMemory:
    """Honcho-backed caller memory. Persists consent state across restarts.

    Uses Honcho peer metadata:
    - One peer per caller, keyed by phone_number_hash.
    - `metadata["consented"] = True` is set when consent is recorded.
    - `get_metadata()` is called on each new call to check prior consent.

    Fails open: any Honcho API error causes has_prior_consent to return False
    (caller hears the full prompt) and record_consent to log a warning.
    """

    COACH_PEER_ID = "rehearse_coach"

    def __init__(
        self,
        api_key: str = "",
        workspace_id: str = "rehearse",
        base_url: str | None = None,
    ) -> None:
        from honcho import Honcho

        self._honcho = Honcho(
            api_key=api_key or None,
            workspace_id=workspace_id,
            base_url=base_url,
        )

    async def has_prior_consent(self, caller_hash: str) -> bool:
        try:
            peer = await self._honcho.aio.peer(caller_hash)
            metadata = await peer.aio.get_metadata()
            return bool(metadata.get("consented"))
        except Exception as exc:
            log.warning(
                "honcho.has_prior_consent.failed",
                caller_hash=caller_hash[:8],
                error=str(exc),
            )
            return False

    async def record_consent(self, caller_hash: str) -> None:
        try:
            peer = await self._honcho.aio.peer(caller_hash)
            await peer.aio.set_metadata({"consented": True})
            log.info("honcho.consent_recorded", caller_hash=caller_hash[:8])
        except Exception as exc:
            log.warning(
                "honcho.record_consent.failed",
                caller_hash=caller_hash[:8],
                error=str(exc),
            )

    async def get_recent_intakes(self, caller_hash: str, n: int = 3) -> list[str]:
        try:
            peer = await self._honcho.aio.peer(caller_hash)
            metadata = await peer.aio.get_metadata()
            intakes = metadata.get("intakes", [])
            return list(intakes)[:n] if isinstance(intakes, list) else []
        except Exception as exc:
            log.warning(
                "honcho.get_recent_intakes.failed",
                caller_hash=caller_hash[:8],
                error=str(exc),
            )
            return []

    async def record_intake(self, caller_hash: str, situation: str) -> None:
        try:
            peer = await self._honcho.aio.peer(caller_hash)
            metadata = await peer.aio.get_metadata()
            intakes: list[str] = list(metadata.get("intakes", []))
            intakes.insert(0, situation)
            intakes = intakes[:20]  # cap at 20 stored intakes
            await peer.aio.set_metadata({**metadata, "intakes": intakes})
            log.info("honcho.intake_recorded", caller_hash=caller_hash[:8])
        except Exception as exc:
            log.warning(
                "honcho.record_intake.failed",
                caller_hash=caller_hash[:8],
                error=str(exc),
            )

    async def store_session(
        self, caller_hash: str, messages: list[dict]
    ) -> None:
        """Store call transcript as Honcho messages for Deriver processing.

        Uses sessions + messages so the Deriver can extract observations and
        the Dialectic can answer queries about the caller in future calls.

        Peer configuration:
          caller peer — observe_me=True (default): Honcho builds a representation
          coach peer  — observe_me=False: deterministic agent, no representation needed
        """
        import time
        from honcho.api_types import PeerConfig
        try:
            caller_peer = await self._honcho.aio.peer(caller_hash)
            coach_peer = await self._honcho.aio.peer(
                self.COACH_PEER_ID,
                configuration=PeerConfig(observe_me=False),
            )
            session = await self._honcho.aio.session(
                f"rehearse-{caller_hash}-{int(time.time())}"
            )
            honcho_messages = [
                caller_peer.message(m["content"]) if m["role"] == "user"
                else coach_peer.message(m["content"])
                for m in messages
                if m.get("content", "").strip()
            ]
            if honcho_messages:
                await session.aio.add_messages(honcho_messages)
            log.info(
                "honcho.session_stored",
                caller_hash=caller_hash[:8],
                messages=len(honcho_messages),
            )
        except Exception as exc:
            log.warning(
                "honcho.store_session.failed",
                caller_hash=caller_hash[:8],
                error=str(exc),
            )

    async def prefetch(self, caller_hash: str, query: str) -> str:
        """Query the Honcho Dialectic for synthesized insights about this caller.

        Calls peer.aio.chat(query) which uses Honcho's Dialectic — an LLM agent
        that searches stored observations and returns a synthesized answer.
        Returns "" for a first-time caller or if the Deriver hasn't run yet.
        """
        try:
            peer = await self._honcho.aio.peer(caller_hash)
            result = await peer.aio.chat(query)
            return result or ""
        except Exception as exc:
            log.warning(
                "honcho.prefetch.failed",
                caller_hash=caller_hash[:8],
                error=str(exc),
            )
            return ""


class MCPCallerMemory:
    """Connects to any MCP server implementing has_prior_consent + record_consent tools.

    The server endpoint is passed as ``url`` (e.g. ``http://localhost:3333/mcp``).
    If the trailing ``/mcp`` path is omitted it is appended automatically to
    match the FastMCP default ``streamable_http_path``.

    Fails open: any connection or tool error causes has_prior_consent to return
    False (caller hears the full prompt) and record_consent to log a warning.
    """

    def __init__(self, url: str) -> None:
        if not url.endswith("/mcp"):
            url = url.rstrip("/") + "/mcp"
        self._url = url

    async def has_prior_consent(self, caller_hash: str) -> bool:
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client

            async with streamablehttp_client(self._url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        "has_prior_consent", {"caller_hash": caller_hash}
                    )
            # The tool returns a bool serialised as a text content block.
            raw = result.content[0].text if result.content else "false"
            value = raw.strip().lower() in ("true", "1", "yes")
            log.debug("mcp.has_prior_consent", caller_hash=caller_hash[:8], result=value)
            return value
        except Exception as exc:
            log.warning(
                "mcp.has_prior_consent.failed",
                caller_hash=caller_hash[:8],
                error=str(exc),
            )
            return False

    async def record_consent(self, caller_hash: str) -> None:
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client

            async with streamablehttp_client(self._url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    await session.call_tool("record_consent", {"caller_hash": caller_hash})
            log.info("mcp.consent_recorded", caller_hash=caller_hash[:8])
        except Exception as exc:
            log.warning(
                "mcp.record_consent.failed",
                caller_hash=caller_hash[:8],
                error=str(exc),
            )
