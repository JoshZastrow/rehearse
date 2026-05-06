"""Tests for converting (BenchmarkExample, RolloutResult) → DeepEval test cases.

Bridges our voice-shaped artifacts (transcript.jsonl, payload dicts) to the
input that DeepEval metrics expect.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from rehearse.eval.deepeval_adapter import to_conversational_test_case, to_llm_test_case
from rehearse.eval.protocols import BenchmarkExample, RolloutResult


def _make_example(**overrides) -> BenchmarkExample:
    base = dict(
        id="ex-1",
        benchmark="mme-sandbox-rollout",
        payload={
            "scenario": {
                "situation": "ask manager for raise",
                "goal": "get a number commitment",
                "counterparty_role": "manager",
            }
        },
        expected={"opening_emotion": "anxious"},
    )
    base.update(overrides)
    return BenchmarkExample(**base)


def _make_rollout(artifacts_dir: Path | None = None, **overrides) -> RolloutResult:
    base = dict(
        example_id="ex-1",
        target_name="voice-agent-sandbox",
        target_version="v0",
        status="ok",
        started_at=datetime(2026, 5, 6, 12, 0, 0),
        completed_at=datetime(2026, 5, 6, 12, 0, 5),
        duration_ms=5000,
        artifacts_dir=artifacts_dir,
        payload=None,
    )
    base.update(overrides)
    return RolloutResult(**base)


def _write_transcript(dir_: Path, frames: list[dict]) -> Path:
    p = dir_ / "transcript.jsonl"
    p.write_text("".join(json.dumps(f) + "\n" for f in frames))
    return p


def test_to_conversational_test_case_reads_transcript_artifacts(tmp_path: Path) -> None:
    _write_transcript(
        tmp_path,
        [
            {"speaker": "user", "text": "I'm nervous about asking."},
            {"speaker": "coach", "text": "What's the biggest worry?"},
            {"speaker": "user", "text": "That she'll say no."},
        ],
    )
    example = _make_example()
    rollout = _make_rollout(artifacts_dir=tmp_path)

    tc = to_conversational_test_case(example, rollout)

    assert len(tc.turns) == 3
    assert tc.turns[0].role == "user"
    assert tc.turns[0].content == "I'm nervous about asking."
    assert tc.turns[1].role == "assistant"  # coach maps to assistant
    assert tc.turns[1].content == "What's the biggest worry?"
    assert tc.turns[2].role == "user"


def test_to_conversational_test_case_includes_scenario_metadata(tmp_path: Path) -> None:
    _write_transcript(
        tmp_path, [{"speaker": "user", "text": "hi"}, {"speaker": "coach", "text": "hi"}]
    )
    example = _make_example()
    rollout = _make_rollout(artifacts_dir=tmp_path)

    tc = to_conversational_test_case(example, rollout)

    # scenario surfaces as test-case metadata for judge prompt grounding
    assert "ask manager for raise" in (tc.scenario or "")
    assert tc.chatbot_role  # coach role description present


def test_to_conversational_test_case_uses_payload_when_no_artifacts() -> None:
    payload = {
        "transcript": [
            {"speaker": "user", "text": "first"},
            {"speaker": "coach", "text": "second"},
        ]
    }
    example = _make_example()
    rollout = _make_rollout(artifacts_dir=None, payload=payload)

    tc = to_conversational_test_case(example, rollout)

    assert len(tc.turns) == 2
    assert tc.turns[0].content == "first"
    assert tc.turns[1].content == "second"


def test_to_conversational_test_case_raises_when_no_transcript(tmp_path: Path) -> None:
    example = _make_example()
    rollout = _make_rollout(artifacts_dir=tmp_path)  # no transcript.jsonl written

    with pytest.raises(ValueError, match="transcript"):
        to_conversational_test_case(example, rollout)


def test_to_conversational_test_case_skips_malformed_jsonl_lines(tmp_path: Path) -> None:
    p = tmp_path / "transcript.jsonl"
    p.write_text(
        json.dumps({"speaker": "user", "text": "good"})
        + "\nnot-json-garbage\n"
        + json.dumps({"speaker": "coach", "text": "also good"})
        + "\n"
    )
    example = _make_example()
    rollout = _make_rollout(artifacts_dir=tmp_path)

    tc = to_conversational_test_case(example, rollout)

    assert len(tc.turns) == 2  # bad line dropped, others kept


def test_to_llm_test_case_collapses_to_last_pair(tmp_path: Path) -> None:
    _write_transcript(
        tmp_path,
        [
            {"speaker": "user", "text": "earlier"},
            {"speaker": "coach", "text": "earlier reply"},
            {"speaker": "user", "text": "what should I say next?"},
            {"speaker": "coach", "text": "try this phrasing."},
        ],
    )
    example = _make_example()
    rollout = _make_rollout(artifacts_dir=tmp_path)

    tc = to_llm_test_case(example, rollout)

    # LLMTestCase is single-turn: input = last user message, actual_output = last coach reply
    assert tc.input == "what should I say next?"
    assert tc.actual_output == "try this phrasing."
