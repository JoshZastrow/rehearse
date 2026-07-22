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
