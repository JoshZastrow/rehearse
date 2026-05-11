"""Integration tests for the consent gate and declined-session finalize path."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rehearse.bus import FrameBus
from rehearse.consent import ConsentGate, ConsentGateConfig
from rehearse.frames import ConsentResolved, TranscriptDelta
from rehearse.participants import SpeakRequest
from rehearse.personas import (
    CONSENT_DECLINE_ACK,
    CONSENT_PROMPT,
    CONSENT_REPROMPT,
    classify_consent,
)
from rehearse.session import SessionHandle, SessionOrchestrator
from rehearse.storage import LocalFilesystemStore
from rehearse.synthesis import SessionSynthesizer
from rehearse.types import ConsentState, Session, Speaker

# ---------------------------------------------------------------------------
# classify_consent fixture (spec §6.1)
# ---------------------------------------------------------------------------

GRANTED_FIXTURE = [
    "yes",
    "Yeah.",
    "yep",
    "sure, that's fine",
    "okay",
    "go ahead",
    "absolutely",
    "alright",
    "yes, that works",
    "sounds good",
]

DECLINED_FIXTURE = [
    "no",
    "Nope.",
    "I'd rather not",
    "no thanks",
    "decline",
    "stop",
    "do not",
    "no, please don't",
]

UNCLEAR_FIXTURE = [
    "maybe",
    "what does that mean",
    "I'm not sure",
    "",
    "   ",
]


@pytest.mark.parametrize("text", GRANTED_FIXTURE)
def test_classify_consent_granted(text: str) -> None:
    assert classify_consent(text) == "granted"


@pytest.mark.parametrize("text", DECLINED_FIXTURE)
def test_classify_consent_declined(text: str) -> None:
    assert classify_consent(text) == "declined"


@pytest.mark.parametrize("text", UNCLEAR_FIXTURE)
def test_classify_consent_unclear(text: str) -> None:
    assert classify_consent(text) == "unclear"


# ---------------------------------------------------------------------------
# ConsentGate lifecycle
# ---------------------------------------------------------------------------


@pytest.fixture
def gate_store(tmp_path: Path) -> tuple[LocalFilesystemStore, str]:
    store = LocalFilesystemStore(tmp_path, "https://example.test")
    session = Session(created_at=datetime.now(UTC), consent=ConsentState.PENDING)
    (store.session_dir(session.id) / "session.json").write_text(
        session.model_dump_json(indent=2)
    )
    return store, session.id


def _read_manifest(store: LocalFilesystemStore, session_id: str) -> dict:
    return json.loads((store.session_dir(session_id) / "session.json").read_text())


def _user_frame(session_id: str, text: str, *, ts: float = 0.5) -> TranscriptDelta:
    return TranscriptDelta(
        session_id=session_id,
        utterance_id=f"u-{ts}",
        speaker=Speaker.USER,
        text=text,
        is_final=True,
        ts_start=ts,
        ts_end=ts + 0.1,
    )


class _Speaker:
    def __init__(self, spoken: list[str]) -> None:
        self._spoken = spoken

    async def say(self, request: SpeakRequest) -> None:
        self._spoken.append(request.text)


@pytest.mark.asyncio
async def test_consent_gate_grants_on_yes(
    gate_store: tuple[LocalFilesystemStore, str],
) -> None:
    store, session_id = gate_store
    bus = FrameBus(session_id)
    spoken: list[str] = []
    declined_called: list[bool] = []

    async def on_decline() -> None:
        declined_called.append(True)

    received: list[ConsentResolved] = []

    async def collect():
        async for frame in bus.subscribe():
            if isinstance(frame, ConsentResolved):
                received.append(frame)

    collector = asyncio.create_task(collect())
    gate = ConsentGate(
        session_id, store, bus,
        speaker=_Speaker(spoken),
        on_decline=on_decline,
        config=ConsentGateConfig(prompt_timeout_seconds=10, reprompt_limit=1),
    )
    task = asyncio.create_task(gate.run(bus.subscribe()))
    await asyncio.sleep(0.01)
    await bus.publish(_user_frame(session_id, "yes"))
    await asyncio.sleep(0.05)
    await bus.aclose()
    await task
    await collector

    assert spoken == [CONSENT_PROMPT]
    assert declined_called == []
    manifest = _read_manifest(store, session_id)
    assert manifest["consent"] == "granted"
    assert any(r.state == ConsentState.GRANTED for r in received)


@pytest.mark.asyncio
async def test_consent_gate_declines_on_no(
    gate_store: tuple[LocalFilesystemStore, str],
) -> None:
    store, session_id = gate_store
    bus = FrameBus(session_id)
    spoken: list[str] = []
    declined_called: list[bool] = []

    async def on_decline() -> None:
        declined_called.append(True)

    gate = ConsentGate(
        session_id, store, bus,
        speaker=_Speaker(spoken),
        on_decline=on_decline,
        config=ConsentGateConfig(prompt_timeout_seconds=10, reprompt_limit=1),
    )
    task = asyncio.create_task(gate.run(bus.subscribe()))
    await asyncio.sleep(0)
    await bus.publish(_user_frame(session_id, "no"))
    await asyncio.sleep(0)
    await bus.aclose()
    await task

    assert spoken == [CONSENT_PROMPT, CONSENT_DECLINE_ACK]
    assert declined_called == [True]
    manifest = _read_manifest(store, session_id)
    assert manifest["consent"] == "declined"


@pytest.mark.asyncio
async def test_consent_gate_reprompts_then_declines_on_unclear(
    gate_store: tuple[LocalFilesystemStore, str],
) -> None:
    store, session_id = gate_store
    bus = FrameBus(session_id)
    spoken: list[str] = []

    async def on_decline() -> None:
        return None

    gate = ConsentGate(
        session_id, store, bus,
        speaker=_Speaker(spoken),
        on_decline=on_decline,
        config=ConsentGateConfig(prompt_timeout_seconds=10, reprompt_limit=1),
    )
    task = asyncio.create_task(gate.run(bus.subscribe()))
    await asyncio.sleep(0)
    await bus.publish(_user_frame(session_id, "what do you mean", ts=0.1))
    await asyncio.sleep(0)
    await bus.publish(_user_frame(session_id, "I'm not sure", ts=0.2))
    await asyncio.sleep(0)
    await bus.aclose()
    await task

    assert spoken == [CONSENT_PROMPT, CONSENT_REPROMPT, CONSENT_DECLINE_ACK]
    manifest = _read_manifest(store, session_id)
    assert manifest["consent"] == "declined"


# ---------------------------------------------------------------------------
# Declined-session finalize: purge artifacts, skip synthesis (spec §3.1 / §11)
# ---------------------------------------------------------------------------


class _StubNotifier:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def send_sms(self, to: str, body: str) -> str:
        self.sent.append((to, body))
        return "sid-stub"


@pytest.mark.asyncio
async def test_finalize_declined_purges_audio_and_skips_synthesis(
    tmp_path: Path,
) -> None:
    store = LocalFilesystemStore(tmp_path, "https://example.test")
    notifier = _StubNotifier()
    orchestrator = SessionOrchestrator(
        store,
        synthesizer=SessionSynthesizer(),
        notifier=notifier,
    )
    session = Session(created_at=datetime.now(UTC), consent=ConsentState.DECLINED)
    session_dir = store.session_dir(session.id)
    (session_dir / "session.json").write_text(session.model_dump_json(indent=2))
    (session_dir / "audio.wav").write_bytes(b"RIFF....WAVEfmtfake")
    (session_dir / "prosody.jsonl").write_text('{"x":1}\n{"x":2}\n')
    transcript_lines = [
        '{"session_id":"x","utterance_id":"u1","ts_start":0.0,"ts_end":0.1,'
        '"speaker":"coach","phase":"intake","text":"prompt","is_interim":false}',
        '{"session_id":"x","utterance_id":"u2","ts_start":0.2,"ts_end":0.3,'
        '"speaker":"user","phase":"intake","text":"what","is_interim":false}',
        '{"session_id":"x","utterance_id":"u3","ts_start":0.4,"ts_end":0.5,'
        '"speaker":"coach","phase":"intake","text":"reprompt","is_interim":false}',
        '{"session_id":"x","utterance_id":"u4","ts_start":0.6,"ts_end":0.7,'
        '"speaker":"user","phase":"intake","text":"no thanks","is_interim":false}',
        '{"session_id":"x","utterance_id":"u5","ts_start":0.8,"ts_end":0.9,'
        '"speaker":"user","phase":"intake","text":"leaked content","is_interim":false}',
    ]
    (session_dir / "transcript.jsonl").write_text("\n".join(transcript_lines) + "\n")

    orchestrator._handles[session.id] = SessionHandle(
        session_id=session.id,
        session_dir=session_dir,
        started_at=session.created_at,
        consent=ConsentState.DECLINED,
        from_number_hash="hash",
        reply_to_number="+15555550100",
    )
    await orchestrator.finalize(session.id, "partial")

    assert not (session_dir / "audio.wav").exists()
    assert not (session_dir / "prosody.jsonl").exists()
    assert not (session_dir / "story.md").exists()
    assert not (session_dir / "feedback.md").exists()
    transcript = (session_dir / "transcript.jsonl").read_text()
    assert "leaked content" not in transcript
    assert notifier.sent == []  # no viewer SMS on declined sessions
    manifest = _read_manifest(store, session.id)
    assert manifest["completion_status"] == "partial"
    assert "audio" not in manifest["artifact_paths"]
    assert "prosody" not in manifest["artifact_paths"]


@pytest.mark.asyncio
async def test_finalize_promotes_pending_outcome_to_skipped(
    tmp_path: Path,
) -> None:
    store = LocalFilesystemStore(tmp_path, "https://example.test")
    notifier = _StubNotifier()
    orchestrator = SessionOrchestrator(
        store,
        synthesizer=SessionSynthesizer(),
        notifier=notifier,
    )
    session = Session(created_at=datetime.now(UTC), consent=ConsentState.GRANTED)
    session.outcome_probe_status = "asked"
    session_dir = store.session_dir(session.id)
    (session_dir / "session.json").write_text(session.model_dump_json(indent=2))
    orchestrator._handles[session.id] = SessionHandle(
        session_id=session.id,
        session_dir=session_dir,
        started_at=session.created_at,
        consent=ConsentState.GRANTED,
        from_number_hash="hash",
        reply_to_number=None,
    )
    await orchestrator.finalize(session.id, "complete")

    manifest = _read_manifest(store, session.id)
    assert manifest["outcome_probe_status"] == "skipped"
