"""Tests for `DeliveryJudgeScorer`.

Reads per-turn user audio (`audio/user/turn_<N>.wav`) AND per-turn coach
audio (`audio/coach/turn_<N>.wav`) from `rollout.artifacts_dir`, sends
both to a multimodal LLM judge, and emits a single `delivery_quality`
`RubricScore` covering attunement + expressiveness.

Without coach audio, the dimension is *not emitted* (degraded; downstream
data card filters this out).
"""

from __future__ import annotations

import json
import wave
from datetime import datetime
from pathlib import Path

import pytest

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.scorers.audio_judges import AudioJudgeError, DeliveryJudgeScorer, StubAudioJudge


def _silent_wav(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * 8000)
    return path


def _example() -> BenchmarkExample:
    return BenchmarkExample(
        id="ex-d-1",
        benchmark="mme-sandbox-rollout",
        payload={"scenario": {"situation": "talking through grief"}},
    )


def _stage_full_rollout(tmp_path: Path) -> RolloutResult:
    (tmp_path / "transcript.jsonl").write_text(
        json.dumps({"speaker": "user", "text": "I just lost my grandfather."})
        + "\n"
        + json.dumps({"speaker": "coach", "text": "I'm so sorry. Take your time."})
        + "\n"
    )
    _silent_wav(tmp_path / "audio" / "user" / "turn_0.wav")
    _silent_wav(tmp_path / "audio" / "coach" / "turn_0.wav")
    return RolloutResult(
        example_id="ex-d-1",
        target_name="voice-agent-sandbox",
        target_version="v0",
        status="ok",
        started_at=datetime(2026, 5, 6, 12, 0, 0),
        completed_at=datetime(2026, 5, 6, 12, 0, 5),
        duration_ms=5000,
        artifacts_dir=tmp_path,
    )


@pytest.mark.asyncio
async def test_delivery_judge_emits_one_rubric_score(tmp_path: Path) -> None:
    judge = StubAudioJudge(
        response={
            "delivery_quality": {
                "score": 0.81,
                "rationale": "warm, measured pacing",
            }
        }
    )
    scorer = DeliveryJudgeScorer(judge=judge)

    rows = await scorer.score(_example(), _stage_full_rollout(tmp_path), run_id="r")

    assert len(rows) == 1
    assert rows[0].dimension == "delivery_quality"
    assert rows[0].value == pytest.approx(0.81)
    assert rows[0].modality == "audio"


@pytest.mark.asyncio
async def test_delivery_judge_populates_judge_prompt_version(tmp_path: Path) -> None:
    judge = StubAudioJudge(
        response={"delivery_quality": {"score": 0.5, "rationale": ""}}
    )
    scorer = DeliveryJudgeScorer(judge=judge, prompt_version="delivery-v3")
    rows = await scorer.score(_example(), _stage_full_rollout(tmp_path), run_id="r")
    assert rows[0].judge_prompt_version == "delivery-v3"


@pytest.mark.asyncio
async def test_delivery_judge_zeroes_with_audio_missing_when_no_coach_audio(
    tmp_path: Path,
) -> None:
    """No coach audio → dimension not emittable; zero with audio_missing flag."""
    judge = StubAudioJudge(
        response={"delivery_quality": {"score": 0.7, "rationale": ""}}
    )
    scorer = DeliveryJudgeScorer(judge=judge)

    (tmp_path / "transcript.jsonl").write_text(
        json.dumps({"speaker": "coach", "text": "ok"}) + "\n"
    )
    _silent_wav(tmp_path / "audio" / "user" / "turn_0.wav")
    rollout = RolloutResult(
        example_id="ex-d-1",
        target_name="t",
        target_version="v",
        status="ok",
        started_at=datetime(2026, 5, 6, 12, 0, 0),
        completed_at=datetime(2026, 5, 6, 12, 0, 1),
        duration_ms=1000,
        artifacts_dir=tmp_path,
    )

    rows = await scorer.score(_example(), rollout, run_id="r")

    assert rows[0].value == 0.0
    assert "audio_missing" in rows[0].flags
    assert rows[0].judge_prompt_version  # still populated for traceability


@pytest.mark.asyncio
async def test_delivery_judge_zeroes_with_audio_missing_when_no_user_audio(
    tmp_path: Path,
) -> None:
    judge = StubAudioJudge(
        response={"delivery_quality": {"score": 0.5, "rationale": ""}}
    )
    scorer = DeliveryJudgeScorer(judge=judge)

    (tmp_path / "transcript.jsonl").write_text(
        json.dumps({"speaker": "coach", "text": "ok"}) + "\n"
    )
    _silent_wav(tmp_path / "audio" / "coach" / "turn_0.wav")
    rollout = RolloutResult(
        example_id="ex-d-1",
        target_name="t",
        target_version="v",
        status="ok",
        started_at=datetime(2026, 5, 6, 12, 0, 0),
        completed_at=datetime(2026, 5, 6, 12, 0, 1),
        duration_ms=1000,
        artifacts_dir=tmp_path,
    )

    rows = await scorer.score(_example(), rollout, run_id="r")
    assert rows[0].value == 0.0
    assert "audio_missing" in rows[0].flags


@pytest.mark.asyncio
async def test_delivery_judge_zeroes_when_rollout_failed(tmp_path: Path) -> None:
    judge = StubAudioJudge(
        response={"delivery_quality": {"score": 0.9, "rationale": ""}}
    )
    scorer = DeliveryJudgeScorer(judge=judge)
    rollout = _stage_full_rollout(tmp_path).model_copy(update={"status": "error"})

    rows = await scorer.score(_example(), rollout, run_id="r")
    assert rows[0].value == 0.0


@pytest.mark.asyncio
async def test_delivery_judge_zeroes_when_judge_raises(tmp_path: Path) -> None:
    judge = StubAudioJudge(error=AudioJudgeError("audio model timeout"))
    scorer = DeliveryJudgeScorer(judge=judge)

    rows = await scorer.score(_example(), _stage_full_rollout(tmp_path), run_id="r")
    assert rows[0].value == 0.0
    assert "timeout" in (rows[0].rationale or "")


@pytest.mark.asyncio
async def test_delivery_judge_passes_user_and_coach_audio(tmp_path: Path) -> None:
    judge = StubAudioJudge(
        response={"delivery_quality": {"score": 0.6, "rationale": ""}}
    )
    scorer = DeliveryJudgeScorer(judge=judge)

    rollout = _stage_full_rollout(tmp_path)
    await scorer.score(_example(), rollout, run_id="r")

    paths = judge.last_audio_paths
    assert len(paths) == 2
    # Both legs present; user-vs-coach order is implementation detail.
    parts = {part for p in paths for part in p.parts}
    assert "user" in parts
    assert "coach" in parts


@pytest.mark.asyncio
async def test_delivery_judge_zeroes_when_response_missing_dimension(
    tmp_path: Path,
) -> None:
    judge = StubAudioJudge(response={"unrelated": 1.0})
    scorer = DeliveryJudgeScorer(judge=judge)

    rows = await scorer.score(_example(), _stage_full_rollout(tmp_path), run_id="r")
    assert rows[0].value == 0.0
    assert "missing" in (rows[0].rationale or "").lower()
