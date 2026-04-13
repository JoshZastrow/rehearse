"""
realtalk.api_server — Optional FastAPI server for LLM proxy.

Exposes the LiteLLMClient as HTTP endpoints for distributed game servers.
Not required for v1.0 — game loop uses LiteLLMClient directly (in-process).
Can be deployed in v2.0+ for multi-instance game servers.

Design:
- /health: Simple readiness check
- /api/complete: Streaming LLM completion endpoint (SSE, Server-Sent Events)
  Returns: Content-Type: text/event-stream
  Each event: {"type": "text_delta", "text": "..."} or similar

This is a skeleton in v1.3. Full implementation (auth, rate limiting, caching)
deferred to v2.0.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "FastAPI is required for api_server. Install with: pip install fastapi uvicorn"
    )

from realtalk.api import ApiRequest

# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Health check response."""

    status: str


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifecycle (startup/shutdown)."""
    # Startup
    yield
    # Shutdown (cleanup if needed)


app = FastAPI(
    title="RealTalk LLM Proxy",
    description="Multi-provider LLM streaming endpoint for game servers",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint.

    Returns 200 OK if the service is ready.
    """
    return HealthResponse(status="ok")


@app.post("/api/complete")
async def complete(request: ApiRequest):
    """Stream a completion from the LLM.

    Request body: ApiRequest (system_prompt, messages, tools, model, max_tokens)
    Response: Server-Sent Events (SSE) stream
    Content-Type: text/event-stream

    Each event is JSON on a single line:
    - {"type": "text_delta", "text": "..."}
    - {"type": "usage_event", "input_tokens": N, "output_tokens": N, ...}
    - {"type": "tool_use", "id": "...", "name": "...", "input": "..."}
    - {"type": "message_stop", "stop_reason": "..."}

    Example curl:
        curl -X POST http://localhost:8000/api/complete \\
          -H "Content-Type: application/json" \\
          -d '{
            "system_prompt": ["You are helpful."],
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [],
            "model": "claude-3-5-sonnet-20241022"
          }'

    [SKELETON in v1.3 — full implementation (auth, rate limiting) in v2.0+]
    """
    raise NotImplementedError(
        "POST /api/complete is a skeleton. Implement in v1.3.1 or defer to v2.0. "
        "For now, use LiteLLMClient directly in the game loop (in-process)."
    )


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with structured response."""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "status_code": 500,
    }


# ---------------------------------------------------------------------------
# For testing: dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
