"""LiveKit JWT token server — the auth + billing choke point.

GET /api/livekit/token
    Headers: Authorization: Bearer <clerk-session-jwt>
    →  200 {"token": "<livekit-jwt>", "room": "rehearse-<uid>-<uuid>", "url": ...}
    →  401 if the Clerk JWT is missing/invalid
    →  402 if the user has no payment method on file or is over the monthly cap
    →  429 if the user is rate-limited
    →  503 if the server is not configured with a Clerk verifier

GET /healthz
    →  200 {"status": "ok"} — unauthenticated readiness probe (CI / Playwright).

GET /api/livekit/warm  (same auth/billing/rate-limit gate as /token)
    →  200 {"warm": true|false}  — GETs the provider's /health to trigger the
       scale-to-zero model's ~60s cold start early. Frontend fires this on the
       call screen so the wait overlaps with UI time. Gated so only paying users
       can spin up a GPU.

Every request is authenticated (Clerk JWT), gated on billing readiness, and
issued a *per-session* room (`rehearse-<clerk_user_id>-<uuid>`) with a short-TTL
token — no shared anonymous room. The agent joins the same room and recovers the
clerk_user_id by parsing the room name.

Configuration (env):
  LIVEKIT_API_KEY / LIVEKIT_API_SECRET   sign the LiveKit JWT (required in prod)
  LIVEKIT_URL                            returned to the client (LiveKit Cloud wss)
  CLERK_JWKS_URL / CLERK_ISSUER          Clerk JWT verification
  DATABASE_URL                           billing store (unset → in-memory dev store)
  ALLOWED_ORIGINS                        comma-separated CORS allowlist
  MONTHLY_CREDIT_CEILING                 per-user post-pay cap (default 5000 credits)
  TOKEN_TTL_SECONDS                      LiveKit token TTL (default 900)
  RATE_LIMIT_PER_MINUTE                  token requests per user per minute (default 6)
  AGENT_DISPATCH_URL                     agent service base URL; each minted room
                                         is POSTed to <url>/dispatch so the agent
                                         joins it (unset → no dispatch)
  INTERACTIVE_PROVIDER_ENDPOINT          model ws endpoint; /warm derives its
                                         /health URL to trigger the cold start
  WARM_TIMEOUT_SECONDS                   /warm health-probe timeout (default 90)
  BILLING_DEV_ALLOW_ALL                  DEV ONLY: 1 → bypass the billing gate for
                                         any authenticated user (never in prod)
  TOKEN_SERVER_PORT                      dev server port (default 8765)

The app is built by `create_app(verifier, store, config)` so tests inject a fake
verifier + `InMemoryBillingStore` and never touch Clerk, a DB, or the network.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# Walk up from web/livekit/ to find the repo root — for both the .env and, when
# run standalone (`python web/livekit/token_server.py`), the `rehearse` package.
_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
load_dotenv(_REPO_ROOT / ".env")

from rehearse.auth.clerk import (  # noqa: E402
    ClerkVerifier,
    InvalidClerkToken,
    build_clerk_verifier,
)
from rehearse.auth.rooms import clerk_id_from_room, make_room_name  # noqa: E402,F401
from rehearse.billing.store import BillingStore, build_billing_store  # noqa: E402

log = logging.getLogger("rehearse.token_server")


@dataclass
class TokenServerConfig:
    """Signing + policy configuration for the token server."""

    livekit_api_key: str
    livekit_api_secret: str
    livekit_url: str = ""
    monthly_credit_ceiling: float = 5000.0
    token_ttl_seconds: int = 900
    rate_limit_per_minute: int = 6
    stripe_webhook_secret: str = ""
    # Base URL of the agent dispatch service. When set, the token server POSTs
    # each minted room to `<url>/dispatch` so the agent joins that exact room.
    # Unset → no dispatch (local dev where the agent is driven another way).
    agent_dispatch_url: str = ""
    # The interactive model ws endpoint. /api/livekit/warm derives its /health
    # URL and GETs it to trigger the scale-to-zero cold start early. Unset → warm
    # is a no-op.
    interactive_provider_endpoint: str = ""
    warm_timeout_seconds: float = 90.0
    # DEV ONLY: treat any authenticated user as billing_ready, bypassing the
    # payment gate. For local smoke tests without Stripe/Postgres. Off by default;
    # must never be set in production.
    dev_allow_all_billing: bool = False

    @classmethod
    def from_env(cls) -> TokenServerConfig:
        return cls(
            livekit_api_key=os.environ.get("LIVEKIT_API_KEY", "devkey"),
            livekit_api_secret=os.environ.get("LIVEKIT_API_SECRET", "secret"),
            livekit_url=os.environ.get("LIVEKIT_URL", ""),
            monthly_credit_ceiling=float(os.environ.get("MONTHLY_CREDIT_CEILING", "5000")),
            token_ttl_seconds=int(os.environ.get("TOKEN_TTL_SECONDS", "900")),
            rate_limit_per_minute=int(os.environ.get("RATE_LIMIT_PER_MINUTE", "6")),
            stripe_webhook_secret=os.environ.get("STRIPE_WEBHOOK_SECRET", ""),
            agent_dispatch_url=os.environ.get("AGENT_DISPATCH_URL", ""),
            interactive_provider_endpoint=os.environ.get("INTERACTIVE_PROVIDER_ENDPOINT", ""),
            warm_timeout_seconds=float(os.environ.get("WARM_TIMEOUT_SECONDS", "90")),
            dev_allow_all_billing=os.environ.get("BILLING_DEV_ALLOW_ALL", "").lower()
            in ("1", "true"),
        )


@dataclass
class _SlidingWindowRateLimiter:
    """Per-key sliding-window limiter. In-memory; fine for a single instance."""

    max_per_minute: int
    _hits: dict[str, deque[float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def allow(self, key: str, *, now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        window_start = now - 60.0
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] < window_start:
                hits.popleft()
            if len(hits) >= self.max_per_minute:
                return False
            hits.append(now)
            return True


def _bearer_token(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    return authorization.split(" ", 1)[1].strip()


async def _http_dispatch(base_url: str, room: str) -> None:
    """POST the room to the agent's /dispatch endpoint so it joins the call.

    Fast by design: the agent returns 202 after scheduling the session, so this
    does not wait on the model's cold start. Raises on any non-2xx / network
    error so the caller can surface a loud failure instead of a silent room.
    """
    import httpx  # noqa: PLC0415

    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.post(f"{base_url.rstrip('/')}/dispatch", json={"room": room})
        resp.raise_for_status()


def _provider_health_url(ws_endpoint: str) -> str | None:
    """Derive the provider's /health URL from its ws endpoint (…/ws → …/health).

    wss://host/ws → https://host/health ; ws://host/ws → http://host/health.
    Returns None when no endpoint is configured.
    """
    if not ws_endpoint:
        return None
    from urllib.parse import urlsplit, urlunsplit  # noqa: PLC0415

    parts = urlsplit(ws_endpoint)
    scheme = {"wss": "https", "ws": "http"}.get(parts.scheme, parts.scheme)
    path = parts.path[:-3] + "/health" if parts.path.endswith("/ws") else "/health"
    return urlunsplit((scheme, parts.netloc, path, "", ""))


async def _http_warm(health_url: str, timeout: float) -> bool:
    """GET the provider's /health to trigger + observe the cold start.

    The model server only starts serving /health after it has finished loading,
    so a 200 means talk-ready; hitting it while cold is what starts the ~60s
    spin-up. Returns True on 200, False otherwise.
    """
    import httpx  # noqa: PLC0415

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(health_url)
        return resp.status_code == 200


def create_app(
    verifier: ClerkVerifier | None,
    store: BillingStore,
    config: TokenServerConfig,
    *,
    allowed_origins: list[str] | None = None,
    dispatcher=_http_dispatch,
    prober=_http_warm,
) -> FastAPI:
    """Build the token-server FastAPI app with injected auth + billing deps."""
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or ["http://localhost:5173"],
        allow_methods=["GET"],
        allow_headers=["Authorization", "Content-Type"],
    )
    limiter = _SlidingWindowRateLimiter(max_per_minute=config.rate_limit_per_minute)

    @app.get("/healthz")
    async def healthz() -> dict:  # noqa: D401
        # Unauthenticated readiness probe. The token endpoint requires a Clerk
        # JWT and returns 503/401 without one, so it can't double as a "server
        # up?" check — CI and the local Playwright config poll this instead.
        return {"status": "ok"}

    if config.dev_allow_all_billing:
        log.warning(
            "⚠️  BILLING_DEV_ALLOW_ALL is ON — the payment gate is bypassed for "
            "every authenticated user. DEV ONLY; never enable this in production."
        )

    async def _gate(authorization: str | None) -> str:
        """Auth + rate-limit + billing gate shared by /token and /warm.

        Returns the clerk_user_id or raises the matching 503/401/429/402. Both
        endpoints spend GPU-adjacent resources, so both must pass the same gate —
        in particular /warm, which would otherwise let anyone trigger a cold
        start (GPU cost) without a payment method on file.
        """
        import asyncio  # noqa: PLC0415

        if verifier is None:
            raise HTTPException(status_code=503, detail="auth not configured")
        token = _bearer_token(authorization)
        try:
            clerk_user_id = await asyncio.to_thread(verifier.verify, token)
        except InvalidClerkToken as exc:
            raise HTTPException(status_code=401, detail="invalid token") from exc
        if not limiter.allow(clerk_user_id):
            raise HTTPException(status_code=429, detail="rate limited")
        if config.dev_allow_all_billing:
            # DEV bypass: authenticated, but skip the payment gate. Loud on every
            # request so it can't quietly ship to prod.
            log.warning(
                "BILLING BYPASS active (BILLING_DEV_ALLOW_ALL) — %s treated as "
                "billing_ready. DEV ONLY; must not be set in production.",
                clerk_user_id,
            )
            return clerk_user_id
        user = await asyncio.to_thread(store.get_user, clerk_user_id)
        if user is None or not user.billing_ready:
            raise HTTPException(status_code=402, detail="billing not set up")
        if user.monthly_credits_used >= config.monthly_credit_ceiling:
            raise HTTPException(status_code=402, detail="monthly credit limit reached")
        return clerk_user_id

    @app.get("/api/livekit/warm")
    async def warm(  # noqa: D401
        authorization: str | None = Header(default=None),
    ) -> dict:
        """Trigger the model's cold start early so it overlaps with UI time.

        Fire-and-forget from the frontend on the call screen. Gated like /token.
        """
        await _gate(authorization)
        health_url = _provider_health_url(config.interactive_provider_endpoint)
        if not health_url:
            return {"warm": False, "reason": "no provider endpoint configured"}
        try:
            is_warm = await prober(health_url, config.warm_timeout_seconds)
        except Exception:  # noqa: BLE001 — best-effort warm; never fail the caller
            return {"warm": False, "reason": "probe failed"}
        return {"warm": is_warm}

    @app.get("/api/livekit/token")
    async def get_token(  # noqa: D401
        authorization: str | None = Header(default=None),
    ) -> dict:
        clerk_user_id = await _gate(authorization)

        # Issue a per-session room + short-TTL LiveKit token.
        from livekit.api import AccessToken, VideoGrants  # noqa: PLC0415

        room = make_room_name(clerk_user_id)
        lk_token = (
            AccessToken(config.livekit_api_key, config.livekit_api_secret)
            .with_identity(clerk_user_id)
            .with_ttl(timedelta(seconds=config.token_ttl_seconds))
            .with_grants(VideoGrants(room_join=True, room=room))
            .to_jwt()
        )

        # Dispatch the agent into this exact room. Without this the caller
        # connects to an empty room and the AI never joins. Fail loud (503) if
        # the agent is unreachable rather than hand back a silent room.
        if config.agent_dispatch_url:
            try:
                await dispatcher(config.agent_dispatch_url, room)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=503, detail="agent unavailable") from exc

        return {"token": lk_token, "room": room, "url": config.livekit_url}

    @app.post("/api/stripe/webhook")
    async def stripe_webhook(request: Request) -> dict:  # noqa: D401
        """Apply Stripe billing events (payment method on file, invoice paid)."""
        import asyncio  # noqa: PLC0415

        from rehearse.billing.webhook import (  # noqa: PLC0415
            InvalidStripeSignature,
            apply_stripe_event,
            verify_stripe_event,
        )

        if not config.stripe_webhook_secret:
            raise HTTPException(status_code=503, detail="stripe webhook not configured")
        payload = await request.body()
        sig = request.headers.get("stripe-signature", "")
        try:
            event = verify_stripe_event(payload, sig, config.stripe_webhook_secret)
        except InvalidStripeSignature as exc:
            raise HTTPException(status_code=400, detail="invalid signature") from exc
        handled = await asyncio.to_thread(apply_stripe_event, event, store)
        return {"handled": handled}

    return app


def _build_default_app() -> FastAPI:
    """Construct the production app from environment configuration."""
    config = TokenServerConfig.from_env()
    verifier = build_clerk_verifier(
        os.environ.get("CLERK_JWKS_URL"),
        issuer=os.environ.get("CLERK_ISSUER"),
    )
    store = build_billing_store(os.environ.get("DATABASE_URL"))
    origins_env = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173")
    allowed_origins = [o.strip() for o in origins_env.split(",") if o.strip()]
    return create_app(verifier, store, config, allowed_origins=allowed_origins)


app = _build_default_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("TOKEN_SERVER_PORT", "8765"))
    uvicorn.run(app, host="0.0.0.0", port=port)
