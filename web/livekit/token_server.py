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
  TOKEN_SERVER_PORT                      dev server port (default 8765)

The app is built by `create_app(verifier, store, config)` so tests inject a fake
verifier + `InMemoryBillingStore` and never touch Clerk, a DB, or the network.
"""

from __future__ import annotations

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


def create_app(
    verifier: ClerkVerifier | None,
    store: BillingStore,
    config: TokenServerConfig,
    *,
    allowed_origins: list[str] | None = None,
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

    @app.get("/api/livekit/token")
    async def get_token(  # noqa: D401
        authorization: str | None = Header(default=None),
    ) -> dict:
        import asyncio  # noqa: PLC0415

        # 1. Authenticate the Clerk JWT → clerk_user_id.
        if verifier is None:
            raise HTTPException(status_code=503, detail="auth not configured")
        token = _bearer_token(authorization)
        try:
            clerk_user_id = await asyncio.to_thread(verifier.verify, token)
        except InvalidClerkToken as exc:
            raise HTTPException(status_code=401, detail="invalid token") from exc

        # 2. Rate-limit per user.
        if not limiter.allow(clerk_user_id):
            raise HTTPException(status_code=429, detail="rate limited")

        # 3. Billing gate: payment method on file + under the monthly ceiling.
        user = await asyncio.to_thread(store.get_user, clerk_user_id)
        if user is None or not user.billing_ready:
            raise HTTPException(status_code=402, detail="billing not set up")
        if user.monthly_credits_used >= config.monthly_credit_ceiling:
            raise HTTPException(status_code=402, detail="monthly credit limit reached")

        # 4. Issue a per-session room + short-TTL LiveKit token.
        from livekit.api import AccessToken, VideoGrants  # noqa: PLC0415

        room = make_room_name(clerk_user_id)
        lk_token = (
            AccessToken(config.livekit_api_key, config.livekit_api_secret)
            .with_identity(clerk_user_id)
            .with_ttl(timedelta(seconds=config.token_ttl_seconds))
            .with_grants(VideoGrants(room_join=True, room=room))
            .to_jwt()
        )
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
