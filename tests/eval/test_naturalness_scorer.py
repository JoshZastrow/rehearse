"""Tests for `NaturalnessScorer` — deterministic timing-derived metrics.

Reads `timing.jsonl` from `rollout.artifacts_dir`; emits banded scores
for `interruption_rate`, `silence_after_affect`, and `speech_rate_band`.
No LLM, no calibration ρ — pinned thresholds versioned on the scorer.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.scorers.naturalness import NaturalnessScorer


def _example() -> BenchmarkExample:
    return BenchmarkExample(id="ex-n-1", benchmark="b", payload={})


def _rollout(artifacts_dir: Path, status: str = "ok") -> RolloutResult:
    return RolloutResult(
        example_id="ex-n-1",
        target_name="t",
        target_version="v",
        status=status,  # type: ignore[arg-type]
        started_at=datetime(2026, 5, 6, 12, 0, 0),
        completed_at=datetime(2026, 5, 6, 12, 0, 5),
        duration_ms=5000,
        artifacts_dir=artifacts_dir,
    )


def _write_timing(dir_: Path, events: list[dict]) -> Path:
    p = dir_ / "timing.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return p


def _write_transcript(dir_: Path, frames: list[dict]) -> None:
    p = dir_ / "transcript.jsonl"
    p.write_text("\n".join(json.dumps(f) for f in frames) + "\n")


# ─────────────────────────────────────────────────────────────────────
# interruption_rate
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_interruption_rate_zero_when_no_overlap(tmp_path: Path) -> None:
    """User finishes; coach starts after — no overlap, ideal."""
    _write_timing(
        tmp_path,
        [
            {"turn_index": 0, "role": "user", "event": "audio_start", "t_ms": 0},
            {"turn_index": 0, "role": "user", "event": "audio_end", "t_ms": 2000},
            {"turn_index": 0, "role": "coach", "event": "audio_start", "t_ms": 2200},
            {"turn_index": 0, "role": "coach", "event": "audio_end", "t_ms": 3500},
        ],
    )
    scorer = NaturalnessScorer()
    rows = await scorer.score(_example(), _rollout(tmp_path), run_id="r")
    interruption = next(r for r in rows if r.dimension == "naturalness.interruption_rate")
    assert interruption.value == pytest.approx(1.0)  # 0 events / turn → ideal band


@pytest.mark.asyncio
async def test_interruption_rate_pathological_when_coach_cuts_in(tmp_path: Path) -> None:
    """Coach starts while user still speaking — pathological."""
    _write_timing(
        tmp_path,
        [
            {"turn_index": 0, "role": "user", "event": "audio_start", "t_ms": 0},
            {"turn_index": 0, "role": "coach", "event": "audio_start", "t_ms": 500},
            {"turn_index": 0, "role": "user", "event": "audio_end", "t_ms": 2000},
            {"turn_index": 0, "role": "coach", "event": "audio_end", "t_ms": 3000},
        ],
    )
    scorer = NaturalnessScorer()
    rows = await scorer.score(_example(), _rollout(tmp_path), run_id="r")
    interruption = next(r for r in rows if r.dimension == "naturalness.interruption_rate")
    # 1 interruption in 1 user turn = 1.0 events/turn → pathological band 0.0
    assert interruption.value == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_interruption_ignores_brief_overlap_under_threshold(tmp_path: Path) -> None:
    """Backchannels (<250ms overlap) don't count as interruption."""
    _write_timing(
        tmp_path,
        [
            {"turn_index": 0, "role": "user", "event": "audio_start", "t_ms": 0},
            # Coach mhm 100ms before user ends — backchannel, not interruption.
            {"turn_index": 0, "role": "coach", "event": "audio_start", "t_ms": 1900},
            {"turn_index": 0, "role": "user", "event": "audio_end", "t_ms": 2000},
            {"turn_index": 0, "role": "coach", "event": "audio_end", "t_ms": 2200},
        ],
    )
    scorer = NaturalnessScorer()
    rows = await scorer.score(_example(), _rollout(tmp_path), run_id="r")
    interruption = next(r for r in rows if r.dimension == "naturalness.interruption_rate")
    assert interruption.value == pytest.approx(1.0)  # ideal — short overlap excluded


# ─────────────────────────────────────────────────────────────────────
# silence_after_affect
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_silence_after_user_turn_in_ideal_band(tmp_path: Path) -> None:
    """2.5s silence after user turn — within ideal 1.5-4s band."""
    _write_timing(
        tmp_path,
        [
            {"turn_index": 0, "role": "user", "event": "audio_start", "t_ms": 0},
            {"turn_index": 0, "role": "user", "event": "audio_end", "t_ms": 2000},
            {"turn_index": 0, "role": "coach", "event": "audio_start", "t_ms": 4500},
            {"turn_index": 0, "role": "coach", "event": "audio_end", "t_ms": 6000},
        ],
    )
    scorer = NaturalnessScorer()
    rows = await scorer.score(_example(), _rollout(tmp_path), run_id="r")
    silence = next(r for r in rows if r.dimension == "naturalness.silence_after_affect")
    assert silence.value == pytest.approx(1.0)  # 2.5s in [1.5, 4.0] ideal


@pytest.mark.asyncio
async def test_silence_rushed_when_coach_jumps_immediately(tmp_path: Path) -> None:
    _write_timing(
        tmp_path,
        [
            {"turn_index": 0, "role": "user", "event": "audio_start", "t_ms": 0},
            {"turn_index": 0, "role": "user", "event": "audio_end", "t_ms": 2000},
            {"turn_index": 0, "role": "coach", "event": "audio_start", "t_ms": 2100},  # 100ms
            {"turn_index": 0, "role": "coach", "event": "audio_end", "t_ms": 3500},
        ],
    )
    scorer = NaturalnessScorer()
    rows = await scorer.score(_example(), _rollout(tmp_path), run_id="r")
    silence = next(r for r in rows if r.dimension == "naturalness.silence_after_affect")
    assert silence.value == pytest.approx(0.0)  # <1.0s rushed


@pytest.mark.asyncio
async def test_silence_dead_air_when_coach_takes_too_long(tmp_path: Path) -> None:
    _write_timing(
        tmp_path,
        [
            {"turn_index": 0, "role": "user", "event": "audio_start", "t_ms": 0},
            {"turn_index": 0, "role": "user", "event": "audio_end", "t_ms": 2000},
            {"turn_index": 0, "role": "coach", "event": "audio_start", "t_ms": 9000},  # 7s
            {"turn_index": 0, "role": "coach", "event": "audio_end", "t_ms": 11000},
        ],
    )
    scorer = NaturalnessScorer()
    rows = await scorer.score(_example(), _rollout(tmp_path), run_id="r")
    silence = next(r for r in rows if r.dimension == "naturalness.silence_after_affect")
    assert silence.value == pytest.approx(0.0)  # >6.0s dead air


# ─────────────────────────────────────────────────────────────────────
# speech_rate_band
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_speech_rate_ideal_band(tmp_path: Path) -> None:
    """~150 wpm coach delivery — paced ideal."""
    _write_transcript(
        tmp_path,
        [
            {"speaker": "user", "text": "i need help"},
            # 15 words spoken by coach in 6 seconds = 150 wpm
            {
                "speaker": "coach",
                "text": (
                    "Let's break that down into smaller steps "
                    "so you can act on the first one today."
                ),
            },
        ],
    )
    _write_timing(
        tmp_path,
        [
            {"turn_index": 0, "role": "user", "event": "audio_start", "t_ms": 0},
            {"turn_index": 0, "role": "user", "event": "audio_end", "t_ms": 1000},
            {"turn_index": 0, "role": "coach", "event": "audio_start", "t_ms": 2000},
            {"turn_index": 0, "role": "coach", "event": "audio_end", "t_ms": 8000},
        ],
    )
    scorer = NaturalnessScorer()
    rows = await scorer.score(_example(), _rollout(tmp_path), run_id="r")
    rate = next(r for r in rows if r.dimension == "naturalness.speech_rate_band")
    assert rate.value == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_speech_rate_pathological_when_too_fast(tmp_path: Path) -> None:
    _write_transcript(
        tmp_path,
        [
            # 10 words in 2 seconds = 300 wpm (rushed)
            {"speaker": "coach", "text": "this is way too fast for any normal listener really"},
        ],
    )
    _write_timing(
        tmp_path,
        [
            {"turn_index": 0, "role": "coach", "event": "audio_start", "t_ms": 0},
            {"turn_index": 0, "role": "coach", "event": "audio_end", "t_ms": 2000},
        ],
    )
    scorer = NaturalnessScorer()
    rows = await scorer.score(_example(), _rollout(tmp_path), run_id="r")
    rate = next(r for r in rows if r.dimension == "naturalness.speech_rate_band")
    assert rate.value == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────
# scorer-level invariants
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_naturalness_emits_three_rows(tmp_path: Path) -> None:
    _write_transcript(
        tmp_path, [{"speaker": "coach", "text": "hello"}]
    )
    _write_timing(
        tmp_path,
        [
            {"turn_index": 0, "role": "coach", "event": "audio_start", "t_ms": 0},
            {"turn_index": 0, "role": "coach", "event": "audio_end", "t_ms": 1000},
        ],
    )
    scorer = NaturalnessScorer()
    rows = await scorer.score(_example(), _rollout(tmp_path), run_id="r")

    dims = {r.dimension for r in rows}
    assert dims == {
        "naturalness.interruption_rate",
        "naturalness.silence_after_affect",
        "naturalness.speech_rate_band",
    }
    for row in rows:
        assert row.modality == "timing"
        assert row.scorer == "deterministic"
        assert row.judge_prompt_version  # thresholds_version stamped


@pytest.mark.asyncio
async def test_naturalness_handles_missing_timing_file(tmp_path: Path) -> None:
    """No timing.jsonl → emit zero rows + timing_missing flag."""
    scorer = NaturalnessScorer()
    rows = await scorer.score(_example(), _rollout(tmp_path), run_id="r")

    assert len(rows) == 3
    for row in rows:
        assert row.value == 0.0
        assert "timing_missing" in row.flags


@pytest.mark.asyncio
async def test_naturalness_handles_failed_rollout(tmp_path: Path) -> None:
    scorer = NaturalnessScorer()
    rows = await scorer.score(_example(), _rollout(tmp_path, status="error"), run_id="r")
    assert all(r.value == 0.0 for r in rows)


@pytest.mark.asyncio
async def test_naturalness_emits_thresholds_version(tmp_path: Path) -> None:
    _write_transcript(tmp_path, [{"speaker": "coach", "text": "hi"}])
    _write_timing(
        tmp_path,
        [
            {"turn_index": 0, "role": "coach", "event": "audio_start", "t_ms": 0},
            {"turn_index": 0, "role": "coach", "event": "audio_end", "t_ms": 1000},
        ],
    )
    scorer = NaturalnessScorer(thresholds_version="naturalness-v2")
    rows = await scorer.score(_example(), _rollout(tmp_path), run_id="r")
    assert all(r.judge_prompt_version == "naturalness-v2" for r in rows)


@pytest.mark.asyncio
async def test_naturalness_skips_malformed_timing_lines(tmp_path: Path) -> None:
    p = tmp_path / "timing.jsonl"
    p.write_text(
        json.dumps(
            {"turn_index": 0, "role": "coach", "event": "audio_start", "t_ms": 0}
        )
        + "\nnot-json\n"
        + json.dumps(
            {"turn_index": 0, "role": "coach", "event": "audio_end", "t_ms": 1000}
        )
        + "\n"
    )
    _write_transcript(tmp_path, [{"speaker": "coach", "text": "ok"}])
    scorer = NaturalnessScorer()
    rows = await scorer.score(_example(), _rollout(tmp_path), run_id="r")
    # Should not crash; still emits 3 rows.
    assert len(rows) == 3
