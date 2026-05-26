"""MCP memory server — exposes caller memory operations as MCP tools.

Exposed tools:
  has_prior_consent(caller_hash) → bool
  record_consent(caller_hash)
  store_session(caller_hash, messages)
  prefetch(caller_hash, query) → str

Default backend: Honcho (when HONCHO_API_KEY or HONCHO_BASE_URL is set).
Fallback backend: InMemoryCallerMemory (useful for local dev / tests).

Usage
-----
    uv run python3 -m rehearse.services.memory_mcp_server
    uv run python3 -m rehearse.services.memory_mcp_server --port 3333
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import structlog
from mcp.server.fastmcp import FastMCP

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Backend: Honcho or in-memory dict
# ---------------------------------------------------------------------------

_honcho_api_key = os.environ.get("HONCHO_API_KEY") or None
_honcho_base_url = os.environ.get("HONCHO_BASE_URL") or None
_honcho_workspace_id = os.environ.get("HONCHO_WORKSPACE_ID", "rehearse")

_use_honcho = bool(_honcho_api_key or _honcho_base_url)

_memory_provider = None


def _get_memory():
    global _memory_provider
    if _memory_provider is None:
        if _use_honcho:
            from rehearse.memory.memory import HonchoCallerMemory
            _memory_provider = HonchoCallerMemory(
                api_key=_honcho_api_key or "",
                workspace_id=_honcho_workspace_id,
                base_url=_honcho_base_url,
            )
            log.info("memory_mcp_server.backend", backend="honcho")
        else:
            from rehearse.memory.memory import InMemoryCallerMemory
            _memory_provider = InMemoryCallerMemory()
            log.info("memory_mcp_server.backend", backend="in-memory")
    return _memory_provider


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP("rehearse-memory")


@mcp.tool()
async def has_prior_consent(caller_hash: str) -> bool:
    """Return True if this caller has previously granted consent.

    Args:
        caller_hash: Opaque identifier for the caller (hashed phone number).
    """
    result = await _get_memory().has_prior_consent(caller_hash)
    log.debug("memory.has_prior_consent", caller_hash=caller_hash[:8], result=result)
    return result


@mcp.tool()
async def record_consent(caller_hash: str) -> None:
    """Persist that this caller has granted consent.

    Args:
        caller_hash: Opaque identifier for the caller (hashed phone number).
    """
    await _get_memory().record_consent(caller_hash)
    log.info("memory.record_consent", caller_hash=caller_hash[:8])


@mcp.tool()
async def store_session(caller_hash: str, messages: list[dict[str, Any]]) -> None:
    """Persist a completed call transcript for future semantic recall.

    The backend stores messages so the Honcho Deriver can extract observations
    or Hindsight can index them for semantic search.

    Args:
        caller_hash: Opaque identifier for the caller (hashed phone number).
        messages: List of {"role": "user"|"assistant", "content": str} dicts.
    """
    await _get_memory().store_session(caller_hash, messages)
    log.info("memory.store_session", caller_hash=caller_hash[:8], messages=len(messages))


@mcp.tool()
async def prefetch(caller_hash: str, query: str) -> str:
    """Query the memory backend for synthesized context about this caller.

    Uses Honcho's Dialectic (or Hindsight recall) to answer a natural-language
    question about the caller's history. Returns "" for a first-time caller.

    Args:
        caller_hash: Opaque identifier for the caller (hashed phone number).
        query: Natural-language question about the caller's history.
    """
    result = await _get_memory().prefetch(caller_hash, query)
    log.debug("memory.prefetch", caller_hash=caller_hash[:8], has_result=bool(result))
    return result or ""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rehearse-memory MCP server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MEMORY_MCP_PORT", "3333")),
        help="Port to listen on (default: 3333)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("MEMORY_MCP_HOST", "127.0.0.1"),
        help="Host to bind (default: 127.0.0.1)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    backend = "honcho" if _use_honcho else "in-memory"
    print(
        f"rehearse-memory MCP server starting on http://{args.host}:{args.port}/mcp  "
        f"[backend={backend}]",
        file=sys.stderr,
        flush=True,
    )
    log.info("memory_mcp_server.start", host=args.host, port=args.port, backend=backend)
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.run(transport="streamable-http")
