"""Modal deploy for the consumer web backend: token server + per-call agent.

Two cheap **CPU** containers, both **scale-to-zero** (no GPU, ~$0 idle):

  token_server   — @modal.asgi_app wrapping web/livekit/token_server.py. Mints
                   gated per-session LiveKit tokens and runs /api/livekit/warm.
                   Cold-starts on request (seconds).
  serve_room_job — one call in one room (web/livekit/agent/agent.py:serve_room).
                   Invoked via .spawn() per dispatch, runs for the call's
                   duration in its own container, then scales to zero.

Dispatch is Modal .spawn() (not HTTP): a web endpoint that returned 202 would be
reaped mid-call, so the agent must run as a spawned function. The token server's
`dispatcher` seam (injectable) makes this a drop-in — the token server code is
unchanged from the local HTTP-dispatch path.

The expensive GPU model lives in the separate `rehearse-interactive` app and
stays scale-to-zero there. Config comes from the `rehearse-web` Modal secret
(see envs/prod.env.example + `make sync-secrets`).

Deploy:  make deploy-web   (or: modal deploy infra/web.py)

Session data: serve_room_job mounts the shared `rehearse-sessions` Volume and
writes each call's transcript.jsonl / manifest to /mnt/sessions/<session_id>/ —
the SAME dir the GPU interactive server commits its audio to (session_id matches,
since the agent passes it in the model handshake). So a real call lands as one
complete, trainable record (transcript + audio) for the continual-learning loop.
The commit happens on call end; a mid-call crash loses that call's data.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import modal

MINUTES = 60
_REPO = Path(__file__).parents[1]

# CPU image: runtime deps for the token server + agent, plus the local source.
# No GPU, no torch/moshi — this is just the LiveKit/HTTP glue. (hindsight and the
# heavy core deps are not on this import path, so they're intentionally omitted.)
web_image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install(
        "fastapi>=0.115",
        "uvicorn>=0.30",
        "python-dotenv>=1.0",
        "structlog>=24.4",
        "pydantic>=2.9",
        "websockets>=13",
        "aiohttp>=3.9",
        "numpy>=1.26",
        "livekit>=1.0",
        "livekit-api>=1.0",
        "httpx>=0.27",
        "stripe>=10.0",
        "pyjwt[crypto]>=2.9",
        "psycopg[binary]>=3.1",
    )
    .env({"PYTHONPATH": "/app"})
    # Mount only what runs — NOT web/livekit/app (node_modules is huge).
    .add_local_dir(str(_REPO / "rehearse"), remote_path="/app/rehearse")
    .add_local_dir(str(_REPO / "web" / "livekit" / "agent"), remote_path="/app/web/livekit/agent")
    .add_local_file(
        str(_REPO / "web" / "livekit" / "token_server.py"),
        remote_path="/app/web/livekit/token_server.py",
    )
)

app = modal.App("rehearse-web")

# All config (Clerk, LiveKit, Stripe, DATABASE_URL, INTERACTIVE_PROVIDER_ENDPOINT,
# ALLOWED_ORIGINS, AGENT_DISPATCH_URL=spawn) lives here. `make sync-secrets`
# writes envs/prod.env into it.
_secret = modal.Secret.from_name("rehearse-web")

# Shared session store — the same Volume the GPU interactive servers commit audio
# to (infra/interactive.py). The agent writes transcripts here under the matching
# <session_id>, so each real call becomes one complete record for training data.
sessions_vol = modal.Volume.from_name("rehearse-sessions", create_if_missing=True)
_SESSIONS_MOUNT = "/mnt/sessions"


def _load(path: str, name: str):
    """Import a source file by path (avoids needing __init__.py packages)."""
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@app.function(
    image=web_image,
    secrets=[_secret],
    volumes={_SESSIONS_MOUNT: sessions_vol},
    timeout=30 * MINUTES,
    scaledown_window=1 * MINUTES,
)
def serve_room_job(room: str) -> None:
    """Serve one call in `room` (spawned per dispatch), then scale to zero."""
    import asyncio  # noqa: PLC0415

    # Write this call's transcript/manifest onto the shared Volume (Modal does
    # not auto-persist Volume writes — the commit below is required).
    os.environ["SESSION_ROOT"] = _SESSIONS_MOUNT
    agent = _load("/app/web/livekit/agent/agent.py", "_rehearse_agent")
    asyncio.run(agent.serve_room(room))
    sessions_vol.commit()


@app.function(image=web_image, secrets=[_secret], timeout=150, scaledown_window=2 * MINUTES)
@modal.asgi_app()
def token_server():
    """Gated LiveKit token server + /warm. Dispatches the agent via .spawn()."""
    from rehearse.auth.clerk import build_clerk_verifier  # noqa: PLC0415
    from rehearse.billing.store import build_billing_store  # noqa: PLC0415

    ts = _load("/app/web/livekit/token_server.py", "_rehearse_token_server")

    async def _spawn_dispatch(_marker: str, room: str) -> None:
        # Dispatch = spawn the per-call agent function. `_marker` is the truthy
        # AGENT_DISPATCH_URL ("spawn") that enables dispatch in the token server;
        # there is no HTTP URL on the Modal path.
        serve_room_job.spawn(room)

    config = ts.TokenServerConfig.from_env()
    verifier = build_clerk_verifier(
        os.environ.get("CLERK_JWKS_URL"), issuer=os.environ.get("CLERK_ISSUER")
    )
    store = build_billing_store(os.environ.get("DATABASE_URL"))
    origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
    return ts.create_app(
        verifier, store, config, allowed_origins=origins, dispatcher=_spawn_dispatch
    )
