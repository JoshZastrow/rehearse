"""Tests for the new CLMResponder — pure orchestration class.

Follows the same TestClient + mount_clm_routes pattern as test_clm.py.
Uses FakeLLMTransport and FakeMemoryManager from tests/fakes.py.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rehearse.agents.clm import CLMChatRequest, mount_clm_routes
from rehearse.agents.registry import AgentRegistry
from rehearse.agents.roles.character import CharacterAgent
from rehearse.agents.roles.feedback import FeedbackCoachAgent
from rehearse.agents.roles.intake import IntakeCoachAgent
from rehearse.agents.router import PhaseRouter
from rehearse.config import RuntimeConfig
from rehearse.memory.memory_manager import MemoryManager
from rehearse.memory.memory import NullCallerMemory
from rehearse.agents.new_clm_responder import NewCLMResponder
from rehearse.types import ConsentState, CounterpartyPersona, Phase, PhaseTiming, Session
from tests.fakes import FakeMemoryManager, FakeLLMTransport

_CLM_FALLBACK = "Sorry, I had a brief glitch — what were you saying?"


def _config(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        twilio_account_sid="AC_test",
        twilio_auth_token="test_token",
        twilio_from_number="+15555550100",
        public_base_url="https://example.test",
        hume_api_key="hume_key",
        hume_config_id="cfg",
        session_root=tmp_path,
        anthropic_api_key=None,
        anthropic_model="claude-sonnet-4-6",
        log_level="warning",
        validate_twilio_signature=False,
    )


def _write_session(tmp_path: Path, session_id: str, phase: Phase = Phase.INTAKE) -> None:
    """Write a session.json so the responder can load it."""
    session = Session(
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        phone_number_hash="caller-abc",
        consent=ConsentState.GRANTED,
        phase_timings=[
            PhaseTiming(phase=phase, started_at=datetime(2026, 1, 1, tzinfo=UTC), budget_seconds=60)
        ],
    )
    session_dir = tmp_path / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "session.json").write_text(session.model_dump_json())


def _make_responder(
    tmp_path: Path,
    *,
    transport: FakeLLMTransport | None = None,
    memory: FakeMemoryManager | None = None,
    model: str = "claude-sonnet-4-6",
) -> NewCLMResponder:
    from rehearse.storage import LocalFilesystemStore

    store = LocalFilesystemStore(root=tmp_path, public_base_url="https://example.test")
    _memory = memory or FakeMemoryManager()
    _transport = transport or FakeLLMTransport()

    registry = AgentRegistry()
    registry.register(IntakeCoachAgent(_memory))
    registry.register(CharacterAgent(_memory))
    registry.register(FeedbackCoachAgent(_memory))
    router = PhaseRouter(registry)

    return NewCLMResponder(
        transport=_transport,
        router=router,
        memory=_memory,
        store=store,
        model=model,
    )


def _client(tmp_path: Path, responder: NewCLMResponder) -> TestClient:
    app = FastAPI()
    mount_clm_routes(app, responder, _config(tmp_path))
    return TestClient(app)


# ---------------------------------------------------------------------------
# Core streaming
# ---------------------------------------------------------------------------


def test_clm_responder_streams_transport_chunks(tmp_path: Path) -> None:
    _write_session(tmp_path, "sess-1")
    transport = FakeLLMTransport(chunks=("Hello ", "caller."))
    responder = _make_responder(tmp_path, transport=transport)
    client = _client(tmp_path, responder)

    resp = client.post(
        "/chat/completions",
        params={"custom_session_id": "sess-1"},
        json={"messages": [], "stream": True},
    )
    assert resp.status_code == 200
    assert "Hello " in resp.text
    assert "caller." in resp.text


def test_clm_responder_injects_memory_context_when_recall_non_empty(tmp_path: Path) -> None:
    _write_session(tmp_path, "sess-1")
    memory = FakeMemoryManager(prefetch_response="Prior: salary negotiation")
    transport = FakeLLMTransport()
    responder = _make_responder(tmp_path, transport=transport, memory=memory)
    client = _client(tmp_path, responder)

    client.post(
        "/chat/completions",
        params={"custom_session_id": "sess-1"},
        json={"messages": [], "stream": True},
    )

    assert transport.last_call is not None
    system_texts = " ".join(b["text"] for b in transport.last_call["system_blocks"])
    assert "<memory-context>" in system_texts
    assert "salary negotiation" in system_texts


def test_clm_responder_skips_memory_block_when_recall_empty(tmp_path: Path) -> None:
    _write_session(tmp_path, "sess-1")
    memory = FakeMemoryManager(prefetch_response="")
    transport = FakeLLMTransport()
    responder = _make_responder(tmp_path, transport=transport, memory=memory)
    client = _client(tmp_path, responder)

    client.post(
        "/chat/completions",
        params={"custom_session_id": "sess-1"},
        json={"messages": [], "stream": True},
    )

    assert transport.last_call is not None
    system_texts = " ".join(b["text"] for b in transport.last_call["system_blocks"])
    assert "<memory-context>" not in system_texts


def test_clm_responder_fallback_on_stream_error(tmp_path: Path) -> None:
    _write_session(tmp_path, "sess-1")
    transport = FakeLLMTransport(raise_on_stream=True)
    responder = _make_responder(tmp_path, transport=transport)
    client = _client(tmp_path, responder)

    resp = client.post(
        "/chat/completions",
        params={"custom_session_id": "sess-1"},
        json={"messages": [], "stream": True},
    )
    assert resp.status_code == 200
    # Fallback line is SSE-encoded; check it appears in one of the delta chunks
    assert _CLM_FALLBACK in resp.text or "glitch" in resp.text


# ---------------------------------------------------------------------------
# Timecard — two system blocks when session has phase_timings
# ---------------------------------------------------------------------------


def test_clm_responder_injects_timecard_as_second_system_block(tmp_path: Path) -> None:
    """A session with phase_timings produces a timecard as the second system block."""
    started = datetime(2026, 1, 1, tzinfo=UTC)
    session = Session(
        created_at=started,
        consent=ConsentState.GRANTED,
        phase_timings=[
            PhaseTiming(phase=Phase.INTAKE, started_at=started, budget_seconds=60)
        ],
    )
    session_dir = tmp_path / "sess-tc"
    session_dir.mkdir()
    (session_dir / "session.json").write_text(session.model_dump_json())

    transport = FakeLLMTransport()
    responder = _make_responder(tmp_path, transport=transport)
    client = _client(tmp_path, responder)

    client.post(
        "/chat/completions",
        params={"custom_session_id": "sess-tc"},
        json={"messages": [], "stream": True},
    )

    assert transport.last_call is not None
    blocks = transport.last_call["system_blocks"]
    assert len(blocks) == 2
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert "intake" in blocks[1]["text"].lower()


# ---------------------------------------------------------------------------
# Phase routing via session
# ---------------------------------------------------------------------------


def test_clm_responder_routes_practice_phase_to_character(tmp_path: Path) -> None:
    _write_session(tmp_path, "sess-1", phase=Phase.PRACTICE)
    transport = FakeLLMTransport(chunks=("in character",))
    responder = _make_responder(tmp_path, transport=transport)
    client = _client(tmp_path, responder)

    resp = client.post(
        "/chat/completions",
        params={"custom_session_id": "sess-1"},
        json={"messages": [], "stream": True},
    )
    assert "in character" in resp.text


# ---------------------------------------------------------------------------
# Transport swappability
# ---------------------------------------------------------------------------


def test_clm_responder_transport_swappable(tmp_path: Path) -> None:
    """Same router/memory — only transport differs."""
    _write_session(tmp_path, "sess-a")
    _write_session(tmp_path, "sess-b")

    transport_a = FakeLLMTransport(chunks=("from A",))
    transport_b = FakeLLMTransport(chunks=("from B",))

    from rehearse.storage import LocalFilesystemStore
    store = LocalFilesystemStore(root=tmp_path, public_base_url="https://example.test")
    memory = FakeMemoryManager()
    registry = AgentRegistry()
    registry.register(IntakeCoachAgent(memory))
    registry.register(CharacterAgent(memory))
    registry.register(FeedbackCoachAgent(memory))
    router = PhaseRouter(registry)

    resp_a = NewCLMResponder(transport=transport_a, router=router, memory=memory, store=store, model="m")
    resp_b = NewCLMResponder(transport=transport_b, router=router, memory=memory, store=store, model="m")

    app_a = FastAPI()
    mount_clm_routes(app_a, resp_a, _config(tmp_path))
    app_b = FastAPI()
    mount_clm_routes(app_b, resp_b, _config(tmp_path))

    r_a = TestClient(app_a).post("/chat/completions", params={"custom_session_id": "sess-a"}, json={"messages": [], "stream": True})
    r_b = TestClient(app_b).post("/chat/completions", params={"custom_session_id": "sess-b"}, json={"messages": [], "stream": True})

    assert "from A" in r_a.text
    assert "from B" in r_b.text


# ---------------------------------------------------------------------------
# No session — graceful fallback
# ---------------------------------------------------------------------------


def test_clm_responder_works_without_session_id(tmp_path: Path) -> None:
    transport = FakeLLMTransport(chunks=("no session ok",))
    responder = _make_responder(tmp_path, transport=transport)
    client = _client(tmp_path, responder)

    resp = client.post(
        "/chat/completions",
        json={"messages": [], "stream": True},
    )
    assert resp.status_code == 200
    assert "no session ok" in resp.text
