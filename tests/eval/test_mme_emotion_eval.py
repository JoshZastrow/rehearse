"""MME-Emotion dataset, eval, and deterministic scorer tests."""

from __future__ import annotations

from datetime import datetime

from rehearse.eval.datasets.mme_emotion import MMEEmotionDataset
from rehearse.eval.evals.mme_emotion import MMEEmotionEval
from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.scorers.deterministic import (
    MMERecognitionScorer,
    parse_json_object_with_keys,
)


def test_mme_emotion_dataset_loads_manifest_examples():
    dataset = MMEEmotionDataset()
    examples = list(dataset.load())
    assert len(examples) == 10
    for ex in examples:
        assert ex.benchmark == "mme-emotion"
        assert "video_path" in ex.payload
        assert "prompt" in ex.payload
        assert "label_set" in ex.payload
        assert "label" in ex.expected
        assert ex.metadata["subset"] == "ER_Lab"


def test_mme_emotion_eval_shape():
    eval_spec = MMEEmotionEval()
    assert eval_spec.dataset.name == "mme-emotion"
    assert eval_spec.preferred_environment == "multimodal-llm"
    assert "multimodal-llm" in eval_spec.supported_environments
    assert [s.name for s in eval_spec.scoring_plan()] == ["mme_recognition"]


def test_parse_json_object_with_keys_allows_surrounding_prose():
    out = 'Sure.\n{"label": "Frustration", "reasoning": "tense pacing"}\nDone.'
    parsed = parse_json_object_with_keys(out, ["label"])
    assert parsed == {"label": "Frustration", "reasoning": "tense pacing"}


def test_parse_json_object_with_keys_missing_key_returns_none():
    assert parse_json_object_with_keys('{"reasoning": "tense"}', ["label"]) is None


async def test_mme_recognition_scorer_exact_match():
    scorer = MMERecognitionScorer()
    example = BenchmarkExample(
        id="ex1",
        benchmark="mme-emotion",
        payload={},
        expected={"label": "Frustration"},
    )
    rollout = RolloutResult(
        example_id="ex1",
        target_name="multimodal-llm",
        target_version="v0",
        status="ok",
        started_at=datetime.now(),
        completed_at=datetime.now(),
        duration_ms=1,
        payload={"output": '{"label": "frustration", "reasoning": "tight tone"}'},
    )
    scores = await scorer.score(example, rollout, run_id="r")
    assert len(scores) == 1
    assert scores[0].dimension == "mme_recognition_accuracy"
    assert scores[0].value == 1.0


async def test_mme_recognition_scorer_unparseable_output_yields_zero():
    scorer = MMERecognitionScorer()
    example = BenchmarkExample(
        id="ex1",
        benchmark="mme-emotion",
        payload={},
        expected={"label": "Frustration"},
    )
    rollout = RolloutResult(
        example_id="ex1",
        target_name="multimodal-llm",
        target_version="v0",
        status="ok",
        started_at=datetime.now(),
        completed_at=datetime.now(),
        duration_ms=1,
        payload={"output": "Frustration"},
    )
    scores = await scorer.score(example, rollout, run_id="r")
    assert scores[0].value == 0.0
    assert "could not parse label" in (scores[0].rationale or "")


async def test_mme_recognition_scorer_failed_rollout_yields_zero():
    scorer = MMERecognitionScorer()
    example = BenchmarkExample(
        id="ex1", benchmark="mme-emotion", payload={}, expected={"label": "Fear"}
    )
    rollout = RolloutResult(
        example_id="ex1",
        target_name="multimodal-llm",
        target_version="v0",
        status="error",
        started_at=datetime.now(),
        completed_at=datetime.now(),
        duration_ms=1,
        error="boom",
    )
    scores = await scorer.score(example, rollout, run_id="r")
    assert scores[0].value == 0.0
