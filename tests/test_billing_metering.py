"""Hermetic tests for the session finalize → metering hook.

Covers `finalize_and_bill` (rehearse/session/livekit_session.py) and its wiring
into the agent's `serve_session` finally block:
  - stamps finalized_at + completion_status=complete on the manifest,
  - records exactly one usage_events row with the right credits,
  - reports the Stripe meter event once,
  - is idempotent on session_id (a retried finalize can't double-bill).
No GPU, no LiveKit, no Stripe.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta
from pathlib import Path

from rehearse.audio.livekit_stream import FakeRoomStream
from rehearse.billing.cost import session_credits
from rehearse.billing.store import InMemoryBillingStore
from rehearse.session.livekit_session import finalize_and_bill, write_session_manifest
from rehearse.storage import LocalFilesystemStore
from rehearse.types import Session
from tests._fakes import FakeRoom, _ScriptedCoachBackend

_SILENCE_FRAME = b"\x00" * 640


def _make_store(tmp_path: Path) -> LocalFilesystemStore:
    return LocalFilesystemStore(tmp_path, "http://localhost")


def _created_at(store: LocalFilesystemStore, session_id: str) -> datetime:
    raw = json.loads((store.session_dir(session_id) / "session.json").read_text())
    return Session.model_validate(raw).created_at


class _RecordingReporter:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, customer_id, credits, session_id) -> bool:
        self.calls.append((customer_id, credits, session_id))
        return True


async def test_finalize_stamps_manifest(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    session_id = "sess-final-1"
    write_session_manifest(store, session_id)
    created = _created_at(store, session_id)

    cost = await finalize_and_bill(store, session_id, now=created + timedelta(seconds=120))

    manifest = json.loads((tmp_path / session_id / "session.json").read_text())
    assert manifest["finalized_at"] is not None
    assert manifest["completion_status"] == "complete"
    assert cost is not None
    assert cost.gpu_seconds == 120.0


async def test_finalize_records_usage_and_meters_once(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    session_id = "sess-bill-1"
    write_session_manifest(store, session_id)
    created = _created_at(store, session_id)

    billing = InMemoryBillingStore()
    reporter = _RecordingReporter()

    cost = await finalize_and_bill(
        store,
        session_id,
        billing_store=billing,
        clerk_user_id="user_1",
        stripe_customer_id="cus_1",
        meter_reporter=reporter,
        now=created + timedelta(seconds=300),
    )

    assert cost is not None
    # Ledger row matches the computed credits.
    assert session_id in billing.usage
    assert billing.usage[session_id]["credits"] == cost.credits
    assert billing.get_user("user_1").monthly_credits_used == cost.credits
    # Reported to Stripe exactly once, with the right args.
    assert reporter.calls == [("cus_1", cost.credits, session_id)]


async def test_finalize_is_idempotent(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    session_id = "sess-bill-2"
    write_session_manifest(store, session_id)
    created = _created_at(store, session_id)

    billing = InMemoryBillingStore()
    reporter = _RecordingReporter()

    common = dict(
        billing_store=billing,
        clerk_user_id="user_2",
        stripe_customer_id="cus_2",
        meter_reporter=reporter,
        now=created + timedelta(seconds=180),
    )
    await finalize_and_bill(store, session_id, **common)
    await finalize_and_bill(store, session_id, **common)

    # Second finalize does not double-count or re-report.
    assert len(billing.usage) == 1
    expected = session_credits(
        Session.model_validate(
            json.loads((tmp_path / session_id / "session.json").read_text())
        )
    )
    assert billing.get_user("user_2").monthly_credits_used == expected.credits
    assert len(reporter.calls) == 1


def _load_agent_module():
    agent_path = Path(__file__).parent.parent / "web" / "livekit" / "agent" / "agent.py"
    spec = importlib.util.spec_from_file_location("_rehearse_livekit_agent", agent_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _caller_frames(n: int = 15):
    import asyncio

    q: asyncio.Queue = asyncio.Queue()
    for _ in range(n):
        q.put_nowait(_SILENCE_FRAME)
    q.put_nowait(None)
    return q


async def test_serve_session_finalizes_and_bills(tmp_path: Path) -> None:
    """The agent's serve_session finally block finalizes + meters a real run."""
    agent_mod = _load_agent_module()
    store = _make_store(tmp_path)
    session_id = "sess-serve-1"
    write_session_manifest(store, session_id)

    billing = InMemoryBillingStore()
    room = FakeRoom(participants=1)
    stream = FakeRoomStream(_caller_frames())
    backend = _ScriptedCoachBackend()

    await agent_mod.serve_session(
        room,
        stream,
        backend,
        store,
        session_id,
        participant_wait_s=1.0,
        billing_store=billing,
        clerk_user_id="user_serve",
        stripe_customer_id="cus_serve",
    )

    assert room.disconnected
    manifest = json.loads((tmp_path / session_id / "session.json").read_text())
    assert manifest["finalized_at"] is not None
    # A usage row was recorded for the session.
    assert session_id in billing.usage
    assert billing.usage[session_id]["clerk_id"] == "user_serve"


async def test_serve_session_no_backend_does_not_bill(tmp_path: Path) -> None:
    """No backend → session never ran → no finalize, no usage row."""
    agent_mod = _load_agent_module()
    store = _make_store(tmp_path)
    session_id = "sess-serve-2"
    write_session_manifest(store, session_id)

    billing = InMemoryBillingStore()
    room = FakeRoom(participants=1)
    stream = FakeRoomStream(_caller_frames())

    await agent_mod.serve_session(
        room,
        stream,
        None,  # backend is None → session never runs
        store,
        session_id,
        participant_wait_s=1.0,
        billing_store=billing,
        clerk_user_id="user_serve",
    )

    assert room.disconnected
    assert billing.usage == {}
    manifest = json.loads((tmp_path / session_id / "session.json").read_text())
    assert manifest["finalized_at"] is None
