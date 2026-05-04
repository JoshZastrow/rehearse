"""MME-seeded RL sandbox rollout: dataset → LLM dialogue → judge → 3 metrics.

Two layers:
  1. Integration test with mocked Anthropic for both the sandbox agents and
     the trajectory judge. Verifies dataset shape, dialogue runs, judge
     produces 3 RubricScore rows, weighted_reward arithmetic is correct,
     judge.json artifact lands on disk. Runs in CI without an API key.
  2. Real-LLM smoke. Skipped without ANTHROPIC_API_KEY; otherwise produces
     a real coaching transcript + real judge scores on a single example.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from rehearse.eval.environments.voice_agent_sandbox import LLMSandboxAgent
from rehearse.eval.executors import InProcessExecutor as _InProcessExecutor
from rehearse.eval.runner import RunConfig, execute_run
from rehearse.eval.scorers.llm_judge import LLMJudge


@dataclass
class _MockTextBlock:
    text: str
    type: str = "text"


@dataclass
class _MockResponse:
    content: list[_MockTextBlock]
    stop_reason: str = "end_turn"
    usage: Any = field(
        default_factory=lambda: type("U", (), {"input_tokens": 0, "output_tokens": 0})()
    )


class _DialogueAnthropic:
    """Mock for the customer + coach LLMs. Cycles scripted lines per role."""

    def __init__(self) -> None:
        self._customer = [
            "I need to tell my partner I'm not okay with how this got decided.",
            "I just don't want it to turn into a fight.",
            "Yeah, that helps. Let me try framing it that way.",
        ] * 3  # extra room
        self._coach = [
            "What's the one thing you most want them to hear?",
            "Try opening with what you trust about them, then the concern.",
            "Notice you're already calmer. Bring that into the actual conversation.",
        ] * 3
        self._customer_idx = 0
        self._coach_idx = 0
        self.messages = self

    async def create(self, **kwargs: Any) -> _MockResponse:
        system = kwargs.get("system", "") or ""
        if "simulating a person on a coaching call" in system:
            text = self._customer[self._customer_idx]
            self._customer_idx += 1
        else:
            text = self._coach[self._coach_idx]
            self._coach_idx += 1
        return _MockResponse(content=[_MockTextBlock(text=text)])


class _JudgeAnthropic:
    """Mock for the trajectory judge. Returns scripted JSON."""

    def __init__(self, *, emotion_score: float = 0.8, trajectory_score: float = 0.7) -> None:
        self.emotion_score = emotion_score
        self.trajectory_score = trajectory_score
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    async def create(self, **kwargs: Any) -> _MockResponse:
        self.calls.append(kwargs)
        payload = {
            "emotion_responsiveness": {
                "score": self.emotion_score,
                "rationale": "Coach validated affect and adapted as user softened.",
                "key_moments": [1, 3],
            },
            "coaching_trajectory_quality": {
                "score": self.trajectory_score,
                "rationale": "Specific phrasing offered; user ends with a plan.",
                "key_moments": [3, 5],
            },
        }
        return _MockResponse(content=[_MockTextBlock(text=json.dumps(payload))])


@pytest.fixture
def mock_dialogue_and_judge(monkeypatch: pytest.MonkeyPatch):
    dialogue = _DialogueAnthropic()
    judge = _JudgeAnthropic()

    def _agent_client(self: LLMSandboxAgent) -> _DialogueAnthropic:
        return dialogue

    def _judge_client(self: LLMJudge) -> _JudgeAnthropic:
        return judge

    monkeypatch.setattr(LLMSandboxAgent, "_client_lazy", _agent_client)
    monkeypatch.setattr(LLMJudge, "_client_lazy", _judge_client)
    return dialogue, judge


async def test_mme_sandbox_rollout_integration_mocked(
    mock_dialogue_and_judge: tuple[_DialogueAnthropic, _JudgeAnthropic],
    tmp_path: Path,
):
    config = RunConfig(
        eval_name="mme-sandbox-rollout",
        environment="voice-agent-sandbox",
        limit=1,
        concurrency=1,
        runs_root=tmp_path,
    )
    outcome = await execute_run(config, executor=_InProcessExecutor())

    assert outcome.n_examples == 1
    assert outcome.n_ok == 1, "expected the rollout to complete cleanly"
    assert outcome.n_error == 0

    # Three dimensions emitted: emotion_responsiveness, coaching_trajectory_quality, weighted_reward
    aggs = outcome.aggregate_scores
    assert "emotion_responsiveness" in aggs
    assert "coaching_trajectory_quality" in aggs
    assert "weighted_reward" in aggs
    assert aggs["emotion_responsiveness"] == pytest.approx(0.8)
    assert aggs["coaching_trajectory_quality"] == pytest.approx(0.7)
    expected_reward = 0.45 * 0.8 + 0.55 * 0.7
    assert aggs["weighted_reward"] == pytest.approx(expected_reward, abs=1e-9)

    session_dir = outcome.run_dir / "sessions" / "mme-rollout-anger-001"
    judge_path = session_dir / "judge.json"
    assert judge_path.exists(), "judge.json artifact should be written"
    judge_artifact = json.loads(judge_path.read_text())
    assert judge_artifact["weights"]["emotion_responsiveness"] == pytest.approx(0.45)
    assert judge_artifact["weighted_reward"] == pytest.approx(expected_reward, abs=1e-9)
    assert "judge_output" in judge_artifact

    transcript = (session_dir / "transcript.jsonl").read_text().splitlines()
    speakers = [json.loads(line)["speaker"] for line in transcript]
    assert "user" in speakers and "coach" in speakers

    _dialogue, judge = mock_dialogue_and_judge
    assert len(judge.calls) == 1, "judge should be called exactly once per example"


async def test_mme_sandbox_rollout_judge_failure_yields_zeros(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """If the judge errors, scorer emits zero scores for all 3 dimensions."""
    dialogue = _DialogueAnthropic()
    monkeypatch.setattr(
        LLMSandboxAgent, "_client_lazy", lambda self: dialogue
    )

    class _BrokenJudgeClient:
        messages = None  # not used

    class _RaisingMessages:
        async def create(self, **_: Any):
            raise RuntimeError("simulated network failure")

    broken = _BrokenJudgeClient()
    broken.messages = _RaisingMessages()
    monkeypatch.setattr(LLMJudge, "_client_lazy", lambda self: broken)

    config = RunConfig(
        eval_name="mme-sandbox-rollout",
        environment="voice-agent-sandbox",
        limit=1,
        concurrency=1,
        runs_root=tmp_path,
    )
    outcome = await execute_run(config, executor=_InProcessExecutor())

    assert outcome.n_ok == 1, "rollout itself should succeed even if judge fails"
    assert outcome.aggregate_scores["emotion_responsiveness"] == 0.0
    assert outcome.aggregate_scores["coaching_trajectory_quality"] == 0.0
    assert outcome.aggregate_scores["weighted_reward"] == 0.0


@pytest.mark.live_api
async def test_mme_sandbox_rollout_real_llm(tmp_path: Path):
    """Real Anthropic call across customer + coach + judge. Skipped without key."""
    config = RunConfig(
        eval_name="mme-sandbox-rollout",
        environment="voice-agent-sandbox",
        limit=1,
        concurrency=1,
        runs_root=tmp_path,
    )
    outcome = await execute_run(config, executor=_InProcessExecutor())
    assert outcome.n_ok == 1, f"real-LLM run failed: {outcome.aggregate_scores}"

    aggs = outcome.aggregate_scores
    assert 0.0 <= aggs["emotion_responsiveness"] <= 1.0
    assert 0.0 <= aggs["coaching_trajectory_quality"] <= 1.0
    assert 0.0 <= aggs["weighted_reward"] <= 1.0

    session_dir = outcome.run_dir / "sessions" / "mme-rollout-anger-001"
    assert (session_dir / "transcript.jsonl").exists()
    assert (session_dir / "judge.json").exists()
