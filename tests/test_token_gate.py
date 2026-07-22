"""Hermetic tests for the LiveKit token server's auth + billing gate.

No Clerk, no database, no network: a fake `ClerkVerifier` returns a canned user
id and an `InMemoryBillingStore` supplies billing state. Covers the rejection
matrix (503/401/402/429), CORS locking, and — when livekit-api is installed —
the 200 success path issuing a per-session room.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rehearse.auth.clerk import InvalidClerkToken
from rehearse.auth.rooms import clerk_id_from_room
from rehearse.billing.store import InMemoryBillingStore

_ENDPOINT = "/api/livekit/token"
_ALLOWED = "https://rehearse.example.com"


def _load_token_server():
    """Import web/livekit/token_server.py by path (not an installed package)."""
    path = Path(__file__).parent.parent / "web" / "livekit" / "token_server.py"
    spec = importlib.util.spec_from_file_location("_rehearse_token_server", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: module-level @dataclass resolves cls.__module__ via
    # sys.modules, which is None otherwise.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_TS = _load_token_server()


class _FakeVerifier:
    def __init__(self, uid: str = "user_abc", raises: bool = False) -> None:
        self._uid = uid
        self._raises = raises

    def verify(self, token: str) -> str:
        if self._raises:
            raise InvalidClerkToken("bad token")
        return self._uid


def _config(**overrides):
    defaults = dict(
        livekit_api_key="devkey",
        livekit_api_secret="secret",
        livekit_url="wss://example.livekit.cloud",
    )
    defaults.update(overrides)
    return _TS.TokenServerConfig(**defaults)


def _client(verifier, store, **cfg):
    app = _TS.create_app(verifier, store, _config(**cfg), allowed_origins=[_ALLOWED])
    return TestClient(app)


def _ready_store(uid: str = "user_abc", **kw) -> InMemoryBillingStore:
    store = InMemoryBillingStore()
    store.upsert_user(uid, billing_ready=True, stripe_customer_id="cus_1", **kw)
    return store


def test_503_when_no_verifier_configured() -> None:
    client = _client(None, _ready_store())
    resp = client.get(_ENDPOINT, headers={"Authorization": "Bearer x"})
    assert resp.status_code == 503


def test_401_without_bearer_token() -> None:
    client = _client(_FakeVerifier(), _ready_store())
    assert client.get(_ENDPOINT).status_code == 401


def test_401_on_invalid_token() -> None:
    client = _client(_FakeVerifier(raises=True), _ready_store())
    resp = client.get(_ENDPOINT, headers={"Authorization": "Bearer bad"})
    assert resp.status_code == 401


def test_402_when_no_user_record() -> None:
    client = _client(_FakeVerifier(), InMemoryBillingStore())
    resp = client.get(_ENDPOINT, headers={"Authorization": "Bearer good"})
    assert resp.status_code == 402


def test_402_when_billing_not_ready() -> None:
    store = InMemoryBillingStore()
    store.upsert_user("user_abc", billing_ready=False)
    client = _client(_FakeVerifier(), store)
    resp = client.get(_ENDPOINT, headers={"Authorization": "Bearer good"})
    assert resp.status_code == 402


def test_402_when_over_monthly_ceiling() -> None:
    store = _ready_store(monthly_credits_used=6000.0)
    client = _client(_FakeVerifier(), store, monthly_credit_ceiling=5000.0)
    resp = client.get(_ENDPOINT, headers={"Authorization": "Bearer good"})
    assert resp.status_code == 402


def test_429_when_rate_limited() -> None:
    client = _client(_FakeVerifier(), _ready_store(), rate_limit_per_minute=2)
    headers = {"Authorization": "Bearer good"}
    codes = [client.get(_ENDPOINT, headers=headers).status_code for _ in range(4)]
    # livekit may be absent → success path 500s, but rate limiting happens first.
    assert codes.count(429) >= 2
    assert codes[-1] == 429


def test_cors_rejects_disallowed_origin() -> None:
    client = _client(_FakeVerifier(), _ready_store())
    resp = client.get(
        _ENDPOINT,
        headers={"Authorization": "Bearer good", "Origin": "https://evil.example"},
    )
    assert "access-control-allow-origin" not in {k.lower() for k in resp.headers}


def test_cors_allows_configured_origin() -> None:
    client = _client(_FakeVerifier(), _ready_store())
    resp = client.get(
        _ENDPOINT,
        headers={"Authorization": "Bearer good", "Origin": _ALLOWED},
    )
    assert resp.headers.get("access-control-allow-origin") == _ALLOWED


def test_200_issues_per_session_room() -> None:
    pytest.importorskip("livekit.api", reason="livekit-api not installed")
    client = _client(_FakeVerifier(uid="user_xyz"), _ready_store("user_xyz"))
    resp = client.get(_ENDPOINT, headers={"Authorization": "Bearer good"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["token"]
    assert body["url"] == "wss://example.livekit.cloud"
    assert clerk_id_from_room(body["room"]) == "user_xyz"


def test_each_call_gets_a_fresh_room() -> None:
    pytest.importorskip("livekit.api", reason="livekit-api not installed")
    client = _client(_FakeVerifier(), _ready_store(), rate_limit_per_minute=10)
    headers = {"Authorization": "Bearer good"}
    r1 = client.get(_ENDPOINT, headers=headers).json()["room"]
    r2 = client.get(_ENDPOINT, headers=headers).json()["room"]
    assert r1 != r2


# ---------------------------------------------------------------------------
# Agent dispatch: the caller's per-session room must reach the agent, or the
# agent never joins the room and the call is silent. The token server pushes
# the minted room to the agent's /dispatch endpoint before returning.
# ---------------------------------------------------------------------------


class _RecordingDispatcher:
    """Captures dispatched rooms; optionally raises to simulate an agent outage."""

    def __init__(self, raises: bool = False) -> None:
        self.rooms: list[str] = []
        self._raises = raises

    async def __call__(self, url: str, room: str) -> None:
        self.rooms.append(room)
        if self._raises:
            raise RuntimeError("agent unreachable")


def _dispatch_client(dispatcher, **cfg):
    cfg.setdefault("agent_dispatch_url", "http://agent.local")
    app = _TS.create_app(
        _FakeVerifier(uid="user_xyz"),
        _ready_store("user_xyz"),
        _config(**cfg),
        allowed_origins=[_ALLOWED],
        dispatcher=dispatcher,
    )
    return TestClient(app)


def test_dispatches_minted_room_to_agent() -> None:
    pytest.importorskip("livekit.api", reason="livekit-api not installed")
    dispatcher = _RecordingDispatcher()
    client = _dispatch_client(dispatcher)
    body = client.get(_ENDPOINT, headers={"Authorization": "Bearer good"}).json()
    # The exact room the caller is handed is the room the agent is told to join.
    assert dispatcher.rooms == [body["room"]]


def test_503_when_agent_dispatch_fails() -> None:
    pytest.importorskip("livekit.api", reason="livekit-api not installed")
    client = _dispatch_client(_RecordingDispatcher(raises=True))
    resp = client.get(_ENDPOINT, headers={"Authorization": "Bearer good"})
    assert resp.status_code == 503


def test_no_dispatch_when_url_unset() -> None:
    pytest.importorskip("livekit.api", reason="livekit-api not installed")
    dispatcher = _RecordingDispatcher()
    client = _dispatch_client(dispatcher, agent_dispatch_url="")
    resp = client.get(_ENDPOINT, headers={"Authorization": "Bearer good"})
    assert resp.status_code == 200
    assert dispatcher.rooms == []


# ---------------------------------------------------------------------------
# Pre-warm: the frontend hits /api/livekit/warm on the call screen so the Modal
# model's ~60s cold start overlaps with the user reading the UI instead of
# starting at "tap to start". Auth + billing gated so only paying users can spin
# up a GPU. A GET to the provider's /health both triggers and observes warmup.
# ---------------------------------------------------------------------------

_WARM = "/api/livekit/warm"


def test_provider_health_url_derivation() -> None:
    f = _TS._provider_health_url
    assert f("wss://host--x.modal.run/ws") == "https://host--x.modal.run/health"
    assert f("ws://localhost:8000/ws") == "http://localhost:8000/health"
    assert f("") is None


class _RecordingProber:
    def __init__(self, warm: bool = True, raises: bool = False) -> None:
        self.urls: list[str] = []
        self._warm = warm
        self._raises = raises

    async def __call__(self, health_url: str, timeout: float) -> bool:
        self.urls.append(health_url)
        if self._raises:
            raise RuntimeError("probe failed")
        return self._warm


def _warm_client(prober, uid: str = "user_abc", store=None, **cfg):
    cfg.setdefault("interactive_provider_endpoint", "wss://host--x.modal.run/ws")
    app = _TS.create_app(
        _FakeVerifier(uid=uid),
        store if store is not None else _ready_store(uid),
        _config(**cfg),
        allowed_origins=[_ALLOWED],
        prober=prober,
    )
    return TestClient(app)


def test_warm_probes_provider_health() -> None:
    prober = _RecordingProber(warm=True)
    client = _warm_client(prober)
    resp = client.get(_WARM, headers={"Authorization": "Bearer good"})
    assert resp.status_code == 200
    assert resp.json()["warm"] is True
    assert prober.urls == ["https://host--x.modal.run/health"]


def test_warm_gated_on_billing() -> None:
    store = InMemoryBillingStore()
    store.upsert_user("user_abc", billing_ready=False)
    prober = _RecordingProber()
    client = _warm_client(prober, store=store)
    resp = client.get(_WARM, headers={"Authorization": "Bearer good"})
    assert resp.status_code == 402
    assert prober.urls == []  # no GPU spin-up for non-paying users


def test_warm_no_endpoint_configured() -> None:
    prober = _RecordingProber()
    client = _warm_client(prober, interactive_provider_endpoint="")
    resp = client.get(_WARM, headers={"Authorization": "Bearer good"})
    assert resp.status_code == 200
    assert resp.json()["warm"] is False
    assert prober.urls == []
