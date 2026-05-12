"""MCP memory server — exposes has_prior_consent / record_consent as MCP tools.

Default backend: Honcho (when HONCHO_API_KEY or HONCHO_BASE_URL is set).
Fallback backend: in-memory dict (useful for local dev / tests).

Usage
-----
    uv run python3 -m rehearse.services.memory_mcp_server
    uv run python3 -m rehearse.services.memory_mcp_server --port 3333
"""

from __future__ import annotations

import argparse
import os
import sys

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

# Lazy-initialised; avoids import errors when honcho is not installed.
_honcho_client = None
_in_memory_store: set[str] = set()


def _get_honcho():
    global _honcho_client
    if _honcho_client is None:
        from honcho import Honcho

        _honcho_client = Honcho(
            api_key=_honcho_api_key,
            workspace_id=_honcho_workspace_id,
            base_url=_honcho_base_url,
        )
    return _honcho_client


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
    if _use_honcho:
        try:
            honcho = _get_honcho()
            peer = honcho.peer(caller_hash)
            metadata = await peer.aio.get_metadata()
            result = bool(metadata.get("consented"))
            log.debug("memory.has_prior_consent", caller_hash=caller_hash[:8], result=result, backend="honcho")
            return result
        except Exception as exc:
            log.warning(
                "memory.has_prior_consent.honcho_error",
                caller_hash=caller_hash[:8],
                error=str(exc),
            )
            return False
    else:
        result = caller_hash in _in_memory_store
        log.debug("memory.has_prior_consent", caller_hash=caller_hash[:8], result=result, backend="in-memory")
        return result


@mcp.tool()
async def record_consent(caller_hash: str) -> None:
    """Persist that this caller has granted consent.

    Args:
        caller_hash: Opaque identifier for the caller (hashed phone number).
    """
    if _use_honcho:
        try:
            honcho = _get_honcho()
            peer = honcho.peer(caller_hash)
            await peer.aio.set_metadata({"consented": True})
            log.info("memory.record_consent", caller_hash=caller_hash[:8], backend="honcho")
        except Exception as exc:
            log.warning(
                "memory.record_consent.honcho_error",
                caller_hash=caller_hash[:8],
                error=str(exc),
            )
    else:
        _in_memory_store.add(caller_hash)
        log.info("memory.record_consent", caller_hash=caller_hash[:8], backend="in-memory")


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
    log.info(
        "memory_mcp_server.start",
        host=args.host,
        port=args.port,
        backend=backend,
    )
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.run(transport="streamable-http")
