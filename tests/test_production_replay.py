"""Tests for ProductionSessionsDataset + ProductionReplayEnvironment."""

from __future__ import annotations

import asyncio
import json
import wave
from pathlib import Path

import pytest

from rehearse.eval.datasets.production_sessions import ProductionSessionsDataset
from rehearse.eval.environments.production_replay import ProductionReplayEnvironment


def _make_session(
    root: Path,
    sid: str,
    *,
    consent: str = "granted",
    audio_duration_s: float = 10.0,
    transcript: list[dict] | None = None,
) -> Path:
    session_dir = root / sid
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text(
        json.dumps({"id": sid, "consent": consent, "created_at": "2026-05-06T00:00:00Z"})
    )

    transcript = transcript or [
        {"speaker": "coach", "text": "Hello.", "ts_start": 0.0, "ts_end": 0.0},
        {"speaker": "user", "text": "Hi.", "ts_start": 2.0, "ts_end": 4.0},
        {"speaker": "coach", "text": "Tell me more.", "ts_start": 5.0, "ts_end": 5.0},
        {"speaker": "user", "text": "Okay.", "ts_start": 7.0, "ts_end": 9.0},
    ]
    (session_dir / "transcript.jsonl").write_text(
        "\n".join(json.dumps(t) for t in transcript) + "\n"
    )

    sample_rate = 16_000
    n_frames = int(audio_duration_s * sample_rate)
    with wave.open(str(session_dir / "audio.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * n_frames)
    return session_dir


def test_dataset_filters_unconsented_by_default(tmp_path: Path) -> None:
    _make_session(tmp_path, "granted-1", consent="granted")
    _make_session(tmp_path, "declined-1", consent="declined")
    _make_session(tmp_path, "pending-1", consent="pending")

    examples = list(ProductionSessionsDataset(sessions_root=tmp_path).load())
    assert [e.id for e in examples] == ["granted-1"]


def test_dataset_consent_gate_bypassed_via_env(tmp_path: Path, monkeypatch) -> None:
    _make_session(tmp_path, "granted-1", consent="granted")
    _make_session(tmp_path, "pending-1", consent="pending")

    monkeypatch.setenv("REHEARSE_REQUIRE_CONSENT", "0")
    examples = list(ProductionSessionsDataset(sessions_root=tmp_path).load())
    assert sorted(e.id for e in examples) == ["granted-1", "pending-1"]


def test_dataset_can_skip_consent_gate(tmp_path: Path) -> None:
    _make_session(tmp_path, "granted-1", consent="granted")
    _make_session(tmp_path, "pending-1", consent="pending")

    examples = list(
        ProductionSessionsDataset(
            sessions_root=tmp_path, require_consent=False
        ).load()
    )
    assert sorted(e.id for e in examples) == ["granted-1", "pending-1"]


def test_dataset_skips_incomplete_bundles(tmp_path: Path) -> None:
    full = _make_session(tmp_path, "ok", consent="granted")
    assert (full / "audio.wav").exists()

    bad = tmp_path / "missing-audio"
    bad.mkdir()
    (bad / "session.json").write_text('{"consent":"granted"}')
    (bad / "transcript.jsonl").write_text("")

    examples = list(ProductionSessionsDataset(sessions_root=tmp_path).load())
    assert [e.id for e in examples] == ["ok"]


def test_dataset_skips_empty_transcripts(tmp_path: Path) -> None:
    _make_session(tmp_path, "ok", consent="granted")
    empty = _make_session(tmp_path, "early-disconnect", consent="granted")
    (empty / "transcript.jsonl").write_text("")

    examples = list(ProductionSessionsDataset(sessions_root=tmp_path).load())
    assert [e.id for e in examples] == ["ok"]


def test_replay_writes_per_turn_audio_and_timing(tmp_path: Path) -> None:
    session_root = tmp_path / "sessions"
    session_root.mkdir()
    _make_session(session_root, "s1", audio_duration_s=10.0)

    examples = list(ProductionSessionsDataset(sessions_root=session_root).load())
    assert len(examples) == 1
    example = examples[0]

    env = ProductionReplayEnvironment()
    run_dir = tmp_path / "run"
    result = asyncio.run(env.rollout(example, run_dir, rng_seed=0))

    assert result.status == "ok"
    assert result.artifacts_dir == run_dir
    assert (run_dir / "transcript.jsonl").exists()
    assert (run_dir / "audio" / "user" / "turn_0.wav").exists()
    assert (run_dir / "audio" / "user" / "turn_1.wav").exists()
    assert (run_dir / "audio" / "coach" / "turn_0.wav").exists()
    assert (run_dir / "audio" / "coach" / "turn_1.wav").exists()

    timing = [
        json.loads(line)
        for line in (run_dir / "timing.jsonl").read_text().splitlines()
        if line.strip()
    ]
    # 4 turns × 2 events each
    assert len(timing) == 8
    def _find(role: str, turn: int, event: str) -> dict:
        return next(
            e for e in timing
            if e["role"] == role and e["turn_index"] == turn and e["event"] == event
        )

    assert _find("user", 0, "audio_start")["t_ms"] == 2000
    assert _find("user", 0, "audio_end")["duration_ms"] == 2000
    # Coach turn end inferred from next turn's start (user turn 0 starts at 2.0).
    coach0_end = _find("coach", 0, "audio_end")
    assert coach0_end["duration_ms"] == 2000  # 2.0s - 0.0s


def test_replay_handles_trailing_coach_turn(tmp_path: Path) -> None:
    session_root = tmp_path / "sessions"
    session_root.mkdir()
    transcript = [
        {"speaker": "user", "text": "hi", "ts_start": 0.0, "ts_end": 1.0},
        {"speaker": "coach", "text": "bye", "ts_start": 2.0, "ts_end": 2.0},
    ]
    _make_session(
        session_root, "s1", audio_duration_s=5.0, transcript=transcript
    )

    examples = list(ProductionSessionsDataset(sessions_root=session_root).load())
    env = ProductionReplayEnvironment()
    result = asyncio.run(env.rollout(examples[0], tmp_path / "run", rng_seed=0))

    assert result.status == "ok"
    timing_path = result.artifacts_dir / "timing.jsonl"
    timing = [
        json.loads(line)
        for line in timing_path.read_text().splitlines()
        if line.strip()
    ]
    coach_end = next(
        e for e in timing if e["role"] == "coach" and e["event"] == "audio_end"
    )
    # Trailing coach turn extends to end of audio (5.0s).
    assert coach_end["t_ms"] == 5000


def test_replay_errors_when_bundle_missing(tmp_path: Path) -> None:
    from rehearse.eval.protocols import BenchmarkExample

    env = ProductionReplayEnvironment()
    example = BenchmarkExample(
        id="missing",
        benchmark="production-sessions",
        payload={"session_id": "missing", "session_dir": str(tmp_path / "nope")},
        expected={},
    )
    result = asyncio.run(env.rollout(example, tmp_path / "run", rng_seed=0))
    assert result.status == "error"
    assert "not found" in (result.error or "")



@pytest.mark.parametrize(
    "name",
    ["production-replay"],
)
def test_environment_registered(name: str) -> None:
    from rehearse.eval.environments import ENVIRONMENTS

    assert name in ENVIRONMENTS
