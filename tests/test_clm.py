"""Exercise the runtime's custom language model webhook endpoints.

These tests verify the CLM route contract, bearer-token checks, and the
OpenAI-style SSE chunk format Hume expects from the runtime.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from rehearse.agents.clm import CLMChatRequest, CLMResponder, mount_clm_routes
from rehearse.app import create_app
from rehearse.config import RuntimeConfig
from rehearse.types import ConsentState, Phase, PhaseTiming, Session

_NOW = datetime(2026, 4, 28, 12, 0, tzinfo=UTC)


class FakeResponder:
    """Record incoming CLM requests and stream a fixed reply."""

    def __init__(self, parts: list[str]) -> None:
        """Store the response chunks and track observed requests."""
        self.parts = parts
        self.calls: list[dict[str, object]] = []

    async def stream_reply(
        self,
        *,
        session_id: str | None,
        role: str,
        request: CLMChatRequest,
    ) -> AsyncIterator[str]:
        """Yield the configured chunks and remember the request details."""
        self.calls.append(
            {
                "session_id": session_id,
                "role": role,
                "messages": [message.model_dump() for message in request.messages],
            }
        )
        for part in self.parts:
            yield part


def _config(tmp_path: Path, *, clm_secret: str | None = None) -> RuntimeConfig:
    """Build a runtime config fixture for CLM route tests."""
    return RuntimeConfig(
        twilio_account_sid="AC_test",
        twilio_auth_token="test_token",
        twilio_from_number="+15555550100",
        public_base_url="https://example.test",
        hume_api_key="hume_test_key",
        hume_config_id="cfg_test",
        session_root=tmp_path,
        anthropic_api_key=None,
        anthropic_model="claude-sonnet-4-6",
        hume_clm_secret=clm_secret,
        log_level="warning",
        validate_twilio_signature=False,
    )


def _client(
    tmp_path: Path,
    responder: CLMResponder,
    *,
    clm_secret: str | None = None,
) -> TestClient:
    """Mount the CLM routes on a small FastAPI app and return a test client."""
    app = FastAPI()
    mount_clm_routes(app, responder, _config(tmp_path, clm_secret=clm_secret))
    return TestClient(app)


def test_chat_completions_streams_openai_sse(tmp_path: Path) -> None:
    """The recommended `/chat/completions` route should return SSE chunks."""
    responder = FakeResponder(["hello ", "world"])
    client = _client(tmp_path, responder)

    resp = client.post(
        "/chat/completions",
        params={"custom_session_id": "session-123", "role": "coach"},
        json={
            "model": "coach",
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": "I need help negotiating my job offer.",
                    "models": {"prosody": {"scores": {"Joy": 0.2, "Anxiety": 0.8}}},
                }
            ],
        },
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert 'data: {"id": "chatcmpl-session-123"' in body
    assert '"role": "assistant"' in body
    assert '"content": "hello "' in body
    assert '"content": "world"' in body
    assert "data: [DONE]" in body
    assert responder.calls == [
        {
            "session_id": "session-123",
            "role": "coach",
            "messages": [
                {
                    "role": "user",
                    "content": "I need help negotiating my job offer.",
                    "type": None,
                    "models": {"prosody": {"scores": {"Joy": 0.2, "Anxiety": 0.8}}},
                    "time": None,
                }
            ],
        }
    ]


def test_path_based_hume_clm_route_supports_non_stream_json(tmp_path: Path) -> None:
    """The older path-based route should still work for buffered responses."""
    responder = FakeResponder(["One short reply."])
    client = _client(tmp_path, responder)

    resp = client.post(
        "/hume/clm/session-abc",
        params={"role": "character"},
        json={
            "model": "character",
            "stream": False,
            "messages": [{"type": "assistant_message", "content": "Previous line."}],
        },
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["id"] == "chatcmpl-session-abc"
    assert payload["choices"][0]["message"]["content"] == "One short reply."
    assert responder.calls[0]["role"] == "character"
    assert responder.calls[0]["session_id"] == "session-abc"


def test_chat_completions_infers_character_role_from_practice_phase(tmp_path: Path) -> None:
    """The CLM route should infer `character` when the live session is in practice."""
    responder = FakeResponder(["Practice reply."])
    client = _client(tmp_path, responder)
    session = Session(
        id="session-practice",
        created_at=_NOW,
        consent=ConsentState.PENDING,
        phase_timings=[
            PhaseTiming(
                phase=Phase.INTAKE,
                started_at=_NOW,
                ended_at=_NOW,
                budget_seconds=60,
            ),
            PhaseTiming(
                phase=Phase.PRACTICE,
                started_at=_NOW,
                budget_seconds=180,
            ),
        ],
    )
    session_dir = tmp_path / session.id
    session_dir.mkdir()
    (session_dir / "session.json").write_text(session.model_dump_json(indent=2))

    resp = client.post(
        "/chat/completions",
        params={"custom_session_id": session.id},
        json={"messages": [], "stream": False},
    )

    assert resp.status_code == 200
    assert responder.calls[0]["role"] == "character"


def test_chat_completions_defaults_to_coach_when_phase_not_practice(tmp_path: Path) -> None:
    """The CLM route should default to coach outside the practice phase."""
    responder = FakeResponder(["Coach reply."])
    client = _client(tmp_path, responder)
    session = Session(
        id="session-feedback",
        created_at=_NOW,
        consent=ConsentState.PENDING,
        phase_timings=[
            PhaseTiming(
                phase=Phase.FEEDBACK,
                started_at=_NOW,
                budget_seconds=60,
            )
        ],
    )
    session_dir = tmp_path / session.id
    session_dir.mkdir()
    (session_dir / "session.json").write_text(session.model_dump_json(indent=2))

    resp = client.post(
        "/hume/clm/session-feedback",
        json={"messages": [], "stream": False},
    )

    assert resp.status_code == 200
    assert responder.calls[0]["role"] == "coach"


def test_chat_completions_rejects_bad_bearer_token(tmp_path: Path) -> None:
    """A configured CLM secret should reject missing or incorrect tokens."""
    client = _client(tmp_path, FakeResponder(["ignored"]), clm_secret="secret-123")

    resp = client.post("/chat/completions", json={"messages": []})
    assert resp.status_code == 401

    bad = client.post(
        "/chat/completions",
        headers={"Authorization": "Bearer wrong"},
        json={"messages": []},
    )
    assert bad.status_code == 401

    ok = client.post(
        "/chat/completions",
        headers={"Authorization": "Bearer secret-123"},
        json={"messages": [], "stream": False},
    )
    assert ok.status_code == 200


def test_chat_completions_returns_json_when_stream_disabled(tmp_path: Path) -> None:
    """The CLM route should also support one buffered OpenAI-style response."""
    responder = FakeResponder(["First part. ", "Second part."])
    client = _client(tmp_path, responder)

    resp = client.post(
        "/chat/completions",
        params={"custom_session_id": "session-xyz"},
        json={"messages": [], "stream": False, "model": "coach"},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["object"] == "chat.completion"
    assert payload["system_fingerprint"] == "session-xyz"
    assert payload["choices"][0]["message"] == {
        "role": "assistant",
        "content": "First part. Second part.",
    }


def test_chat_completions_sse_chunks_are_valid_json_lines(tmp_path: Path) -> None:
    """Each streamed SSE line should hold valid JSON until the final DONE marker."""
    client = _client(tmp_path, FakeResponder(["alpha", "beta"]))
    resp = client.post("/chat/completions", json={"messages": []})

    assert resp.status_code == 200
    data_lines = [line for line in resp.text.splitlines() if line.startswith("data: ")]
    json_lines = [line.removeprefix("data: ") for line in data_lines[:-1]]
    assert data_lines[-1] == "data: [DONE]"
    for line in json_lines:
        parsed = json.loads(line)
        assert parsed["object"] == "chat.completion.chunk"


def test_create_app_mounts_the_scripted_clm_route(tmp_path: Path) -> None:
    """The real runtime app should expose the CLM route with the scripted fallback."""
    client = TestClient(create_app(_config(tmp_path)))

    resp = client.post(
        "/chat/completions",
        params={"custom_session_id": "session-live"},
        json={
            "messages": [
                {"role": "user", "content": "I need help asking for more equity."},
            ]
        },
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in resp.text
