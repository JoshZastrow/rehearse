"""Tests for `AffectPerceptionJudgeScorer`.

Reads `transcript.jsonl` + per-turn user audio (`audio/user/turn_<N>.wav`)
from `rollout.artifacts_dir`, calls an audio judge, and emits a single
`affect_perception` `RubricScore` per rollout.
"""

from __future__ import annotations

import json
import wave
from datetime import datetime
from pathlib import Path

import pytest

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.scorers.audio_judges import AffectPerceptionJudgeScorer, AudioJudgeError, StubAudioJudge


def _silent_wav(path: Path, *, duration_s: float = 0.5) -> Path:
    n_samples = int(duration_s * 16_000)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * n_samples)
    return path


def _example() -> BenchmarkExample:
    return BenchmarkExample(
        id="ex-ap-1",
        benchmark="mme-sandbox-rollout",
        payload={
            "scenario": {
                "situation": "deliver hard feedback",
                "goal": "land the message clearly",
            }
        },
        expected={"opening_emotion": "anxious"},
    )


def _make_rollout(tmp_path: Path, *, with_user_audio: bool = True) -> RolloutResult:
    (tmp_path / "transcript.jsonl").write_text(
        json.dumps({"speaker": "user", "text": "I'm dreading this meeting."})
        + "\n"
        + json.dumps({"speaker": "coach", "text": "What worries you most about it?"})
        + "\n"
    )
    if with_user_audio:
        _silent_wav(tmp_path / "audio" / "user" / "turn_0.wav")
    return RolloutResult(
        example_id="ex-ap-1",
        target_name="voice-agent-sandbox",
        target_version="v0",
        status="ok",
        started_at=datetime(2026, 5, 6, 12, 0, 0),
        completed_at=datetime(2026, 5, 6, 12, 0, 5),
        duration_ms=5000,
        artifacts_dir=tmp_path,
    )


@pytest.mark.asyncio
async def test_affect_judge_emits_one_rubric_score(tmp_path: Path) -> None:
    judge = StubAudioJudge(
        response={
            "affect_perception": {
                "score": 0.72,
                "rationale": "coach acknowledged the user's worry",
            }
        }
    )
    scorer = AffectPerceptionJudgeScorer(judge=judge)

    rows = await scorer.score(_example(), _make_rollout(tmp_path), run_id="run-1")

    assert len(rows) == 1
    assert rows[0].dimension == "affect_perception"
    assert rows[0].value == pytest.approx(0.72)


@pytest.mark.asyncio
async def test_affect_judge_marks_modality_audio_text(tmp_path: Path) -> None:
    judge = StubAudioJudge(
        response={"affect_perception": {"score": 0.5, "rationale": ""}}
    )
    scorer = AffectPerceptionJudgeScorer(judge=judge)
    rows = await scorer.score(_example(), _make_rollout(tmp_path), run_id="r")
    assert rows[0].modality == "audio+text"


@pytest.mark.asyncio
async def test_affect_judge_populates_judge_prompt_version(tmp_path: Path) -> None:
    judge = StubAudioJudge(
        response={"affect_perception": {"score": 0.5, "rationale": ""}}
    )
    scorer = AffectPerceptionJudgeScorer(judge=judge, prompt_version="affect-v2")
    rows = await scorer.score(_example(), _make_rollout(tmp_path), run_id="r")
    assert rows[0].judge_prompt_version == "affect-v2"


@pytest.mark.asyncio
async def test_affect_judge_falls_back_text_only_without_user_audio(tmp_path: Path) -> None:
    judge = StubAudioJudge(
        response={"affect_perception": {"score": 0.4, "rationale": "text-only fallback"}}
    )
    scorer = AffectPerceptionJudgeScorer(judge=judge)

    rollout = _make_rollout(tmp_path, with_user_audio=False)
    rows = await scorer.score(_example(), rollout, run_id="r")

    assert "audio_missing" in rows[0].flags
    assert rows[0].modality == "text"  # degraded modality
    assert judge.last_audio_paths == []


@pytest.mark.asyncio
async def test_affect_judge_zeroes_when_rollout_failed(tmp_path: Path) -> None:
    judge = StubAudioJudge(
        response={"affect_perception": {"score": 0.9, "rationale": ""}}
    )
    scorer = AffectPerceptionJudgeScorer(judge=judge)
    rollout = _make_rollout(tmp_path).model_copy(update={"status": "error"})

    rows = await scorer.score(_example(), rollout, run_id="r")

    assert rows[0].value == 0.0
    assert "error" in (rows[0].rationale or "")


@pytest.mark.asyncio
async def test_affect_judge_zeroes_when_transcript_missing(tmp_path: Path) -> None:
    judge = StubAudioJudge(response={})
    scorer = AffectPerceptionJudgeScorer(judge=judge)

    # Skip transcript — empty artifacts dir.
    rollout = RolloutResult(
        example_id="ex-ap-1",
        target_name="t",
        target_version="v",
        status="ok",
        started_at=datetime(2026, 5, 6, 12, 0, 0),
        completed_at=datetime(2026, 5, 6, 12, 0, 5),
        duration_ms=5000,
        artifacts_dir=tmp_path,
    )

    rows = await scorer.score(_example(), rollout, run_id="r")

    assert rows[0].value == 0.0
    assert "transcript" in (rows[0].rationale or "").lower()


@pytest.mark.asyncio
async def test_affect_judge_zeroes_when_judge_raises(tmp_path: Path) -> None:
    judge = StubAudioJudge(error=AudioJudgeError("model failure"))
    scorer = AffectPerceptionJudgeScorer(judge=judge)

    rows = await scorer.score(_example(), _make_rollout(tmp_path), run_id="r")

    assert rows[0].value == 0.0
    assert "model failure" in (rows[0].rationale or "")


@pytest.mark.asyncio
async def test_affect_judge_zeroes_when_response_missing_dimension(tmp_path: Path) -> None:
    judge = StubAudioJudge(response={"unrelated_dim": 0.3})
    scorer = AffectPerceptionJudgeScorer(judge=judge)

    rows = await scorer.score(_example(), _make_rollout(tmp_path), run_id="r")

    assert rows[0].value == 0.0
    assert "missing" in (rows[0].rationale or "").lower()


@pytest.mark.asyncio
async def test_affect_judge_passes_user_audio_paths_to_judge(tmp_path: Path) -> None:
    judge = StubAudioJudge(
        response={"affect_perception": {"score": 0.6, "rationale": ""}}
    )
    scorer = AffectPerceptionJudgeScorer(judge=judge)

    # Stage two user-audio turns so we know it picks them up in order.
    _silent_wav(tmp_path / "audio" / "user" / "turn_0.wav")
    _silent_wav(tmp_path / "audio" / "user" / "turn_2.wav")
    (tmp_path / "transcript.jsonl").write_text(
        json.dumps({"speaker": "user", "text": "first"}) + "\n"
        + json.dumps({"speaker": "coach", "text": "ok"}) + "\n"
        + json.dumps({"speaker": "user", "text": "second"}) + "\n"
    )
    rollout = RolloutResult(
        example_id="ex-ap-1",
        target_name="t",
        target_version="v",
        status="ok",
        started_at=datetime(2026, 5, 6, 12, 0, 0),
        completed_at=datetime(2026, 5, 6, 12, 0, 5),
        duration_ms=5000,
        artifacts_dir=tmp_path,
    )

    await scorer.score(_example(), rollout, run_id="r")

    assert len(judge.last_audio_paths) == 2
    assert all("user" in str(p) for p in judge.last_audio_paths)
