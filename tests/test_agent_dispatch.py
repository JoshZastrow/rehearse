"""Hermetic tests for the LiveKit agent's /dispatch endpoint.

The token server mints a per-session room (`rehearse-<uid>-<uuid>`) and POSTs it
to the agent's /dispatch endpoint; the agent must join *that* room, not a fixed
one. These tests cover the routing seam — that a dispatched room name is handed
to serve_room — without a LiveKit server, Modal, or a real rtc.Room. serve_room
itself (which needs livekit-rtc) is injected as a recording launcher.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from fastapi.testclient import TestClient


def _load_agent():
    """Import web/livekit/agent/agent.py by path (not an installed package)."""
    path = Path(__file__).parent.parent / "web" / "livekit" / "agent" / "agent.py"
    spec = importlib.util.spec_from_file_location("_rehearse_agent", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_token_server():
    """Import web/livekit/token_server.py by path (not an installed package)."""
    path = Path(__file__).parent.parent / "web" / "livekit" / "token_server.py"
    spec = importlib.util.spec_from_file_location("_rehearse_token_server_dsp", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_AGENT = _load_agent()
_TS = _load_token_server()


def test_dispatch_routes_room_to_launcher() -> None:
    launched: list[str] = []
    app = _AGENT.build_dispatch_app(launch=launched.append)
    client = TestClient(app)
    resp = client.post("/dispatch", json={"room": "rehearse-user_xyz-ab12cd34"})
    assert resp.status_code == 202
    assert launched == ["rehearse-user_xyz-ab12cd34"]


def test_dispatch_rejects_missing_room() -> None:
    launched: list[str] = []
    app = _AGENT.build_dispatch_app(launch=launched.append)
    client = TestClient(app)
    resp = client.post("/dispatch", json={})
    assert resp.status_code == 400
    assert launched == []


def test_health_ok() -> None:
    app = _AGENT.build_dispatch_app(launch=lambda room: None)
    client = TestClient(app)
    assert client.get("/health").status_code == 200


async def test_token_server_dispatch_contract_matches_agent_endpoint(monkeypatch) -> None:
    """The token server's real _http_dispatch must agree with the agent's real
    /dispatch endpoint on path, JSON shape, and status handling — the exact
    contract whose mismatch would silently leave the AI out of the room."""
    import httpx

    launched: list[str] = []
    agent_app = _AGENT.build_dispatch_app(launch=launched.append)
    transport = httpx.ASGITransport(app=agent_app)

    real_client = httpx.AsyncClient

    def client_over_agent(*args, **kwargs):
        kwargs.setdefault("transport", transport)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_over_agent)

    # Raises if path/shape/status disagree; returns cleanly otherwise.
    await _TS._http_dispatch("http://agent.local", "rehearse-user_x-ab12cd34")
    assert launched == ["rehearse-user_x-ab12cd34"]
