"""EQ-Bench adapter: example loading, prompt construction, correlation math,
output parsing. No anthropic calls — RawLLMTarget is exercised separately
in a manual smoke test."""

from __future__ import annotations

from datetime import datetime

import pytest

from rehearse.eval.benchmarks.eq_bench import EQBenchBenchmark
from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.scorers.deterministic import (
    EQBenchCorrelationScorer,
    _pearson,
    parse_eq_ratings,
)


def test_eq_bench_loads_sample_examples():
    bench = EQBenchBenchmark()
    examples = list(bench.load())
    assert len(examples) >= 3
    for ex in examples:
        assert ex.benchmark == "eq-bench"
        assert "prompt" in ex.payload
        assert "ratings" in ex.expected
        assert ex.metadata["character"]
        assert isinstance(ex.metadata["emotions"], list)


def test_prompt_includes_dialogue_and_emotion_keys():
    bench = EQBenchBenchmark()
    ex = next(iter(bench.load()))
    prompt = ex.payload["prompt"]
    for emotion in ex.metadata["emotions"]:
        assert emotion in prompt
    assert "JSON" in prompt


def test_pearson_perfect_correlation():
    assert _pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)


def test_pearson_inverse_correlation():
    assert _pearson([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_pearson_zero_variance_returns_zero():
    assert _pearson([5, 5, 5, 5], [1, 2, 3, 4]) == 0.0


def test_parse_eq_ratings_clean_json():
    out = '{"joy": 7, "sadness": 2, "anger": 1, "fear": 0}'
    parsed = parse_eq_ratings(out, ["joy", "sadness", "anger", "fear"])
    assert parsed == {"joy": 7.0, "sadness": 2.0, "anger": 1.0, "fear": 0.0}


def test_parse_eq_ratings_with_surrounding_prose():
    out = 'Sure, here are the ratings:\n{"joy": 5, "sadness": 8}\nLet me know if you need more.'
    parsed = parse_eq_ratings(out, ["joy", "sadness"])
    assert parsed == {"joy": 5.0, "sadness": 8.0}


def test_parse_eq_ratings_missing_keys_returns_none():
    out = '{"joy": 5}'
    assert parse_eq_ratings(out, ["joy", "sadness"]) is None


async def test_correlation_scorer_perfect_match():
    scorer = EQBenchCorrelationScorer()
    example = BenchmarkExample(
        id="ex1",
        benchmark="eq-bench",
        payload={},
        expected={"ratings": {"joy": 1, "sadness": 2, "anger": 3, "fear": 4}},
    )
    rollout = RolloutResult(
        example_id="ex1",
        target_name="raw-llm",
        target_version="v0",
        status="ok",
        started_at=datetime.now(),
        completed_at=datetime.now(),
        duration_ms=1,
        payload={"output": '{"joy": 1, "sadness": 2, "anger": 3, "fear": 4}'},
    )
    scores = await scorer.score(example, rollout, run_id="r")
    assert len(scores) == 1
    assert scores[0].dimension == "eq_bench_score"
    assert scores[0].value == pytest.approx(100.0)


async def test_correlation_scorer_unparseable_output_yields_zero():
    scorer = EQBenchCorrelationScorer()
    example = BenchmarkExample(
        id="ex1",
        benchmark="eq-bench",
        payload={},
        expected={"ratings": {"joy": 1, "sadness": 2, "anger": 3, "fear": 4}},
    )
    rollout = RolloutResult(
        example_id="ex1",
        target_name="raw-llm",
        target_version="v0",
        status="ok",
        started_at=datetime.now(),
        completed_at=datetime.now(),
        duration_ms=1,
        payload={"output": "I'm sorry, I can't help with that."},
    )
    scores = await scorer.score(example, rollout, run_id="r")
    assert scores[0].value == 0.0
    assert "could not parse" in (scores[0].rationale or "")


async def test_correlation_scorer_failed_rollout_yields_zero():
    scorer = EQBenchCorrelationScorer()
    example = BenchmarkExample(
        id="ex1", benchmark="eq-bench", payload={}, expected={"ratings": {"joy": 5}}
    )
    rollout = RolloutResult(
        example_id="ex1",
        target_name="raw-llm",
        target_version="v0",
        status="error",
        started_at=datetime.now(),
        completed_at=datetime.now(),
        duration_ms=1,
        error="boom",
    )
    scores = await scorer.score(example, rollout, run_id="r")
    assert scores[0].value == 0.0
