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

    Any class that provides these two async methods satisfies this protocol.
    No inheritance required.
    """

    async def has_prior_consent(self, caller_hash: str) -> bool:
        """Return True if this caller has previously granted consent."""
        ...

    async def record_consent(self, caller_hash: str) -> None:
        """Persist that this caller has granted consent."""
        ...


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


class NullCallerMemory:
    """No-op memory. Every caller is treated as new. Default when Honcho is not configured."""

    async def has_prior_consent(self, caller_hash: str) -> bool:
        return False

    async def record_consent(self, caller_hash: str) -> None:
        pass


class InMemoryCallerMemory:
    """In-process memory backed by a Python set. Used in tests."""

    def __init__(self) -> None:
        self._consented: set[str] = set()

    async def has_prior_consent(self, caller_hash: str) -> bool:
        return caller_hash in self._consented

    async def record_consent(self, caller_hash: str) -> None:
        self._consented.add(caller_hash)


class HonchoCallerMemory:
    """Honcho-backed caller memory. Persists consent state across restarts.

    Uses Honcho peer metadata:
    - One peer per caller, keyed by phone_number_hash.
    - `metadata["consented"] = True` is set when consent is recorded.
    - `get_metadata()` is called on each new call to check prior consent.

    Fails open: any Honcho API error causes has_prior_consent to return False
    (caller hears the full prompt) and record_consent to log a warning.
    """

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
            # honcho.peer() is sync and returns a Peer; .aio.get_metadata() is async.
            peer = self._honcho.peer(caller_hash)
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
            peer = self._honcho.peer(caller_hash)
            await peer.aio.set_metadata({"consented": True})
            log.info("honcho.consent_recorded", caller_hash=caller_hash[:8])
        except Exception as exc:
            log.warning(
                "honcho.record_consent.failed",
                caller_hash=caller_hash[:8],
                error=str(exc),
            )


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
