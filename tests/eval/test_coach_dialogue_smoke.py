"""Coach-dialogue smoke: LLM customer ⇄ LLM coach over the sandbox env.

Two layers:
  1. Integration test with a mocked Anthropic client. Verifies the env wires
     LLM agents end-to-end, multi-turn alternation works, transcript shape is
     correct, completeness scores green. Runs in CI without an API key.
  2. Real-LLM smoke. Skipped without ANTHROPIC_API_KEY; otherwise produces a
     real coaching transcript.
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


@dataclass
class _MockTextBlock:
    text: str
    type: str = "text"


@dataclass
class _MockResponse:
    content: list[_MockTextBlock]
    stop_reason: str = "end_turn"
    usage: Any = field(default_factory=lambda: type("U", (), {"input_tokens": 0, "output_tokens": 0})())


class _MockAnthropic:
    """Returns scripted responses keyed by which side is speaking.

    Cycles through `customer_lines` and `coach_lines` independently. Detects
    role from the system prompt prefix (the only fixed content per side).
    """

    def __init__(self, customer_lines: list[str], coach_lines: list[str]) -> None:
        self._customer = list(customer_lines)
        self._coach = list(coach_lines)
        self._customer_idx = 0
        self._coach_idx = 0
        self.calls: list[dict[str, Any]] = []
        self.messages = self  # mimic anthropic.AsyncAnthropic().messages.create

    async def create(self, **kwargs: Any) -> _MockResponse:
        system = kwargs.get("system", "") or ""
        self.calls.append({"system_prefix": system[:50], "messages": kwargs.get("messages")})
        if "simulating a person on a coaching call" in system:
            text = (
                self._customer[self._customer_idx]
                if self._customer_idx < len(self._customer)
                else ""
            )
            self._customer_idx += 1
        else:
            text = (
                self._coach[self._coach_idx]
                if self._coach_idx < len(self._coach)
                else ""
            )
            self._coach_idx += 1
        return _MockResponse(content=[_MockTextBlock(text=text)])


@pytest.fixture
def mock_anthropic_factory(monkeypatch: pytest.MonkeyPatch):
    """Patches LLMSandboxAgent._client_lazy to return our mock per role."""

    customer_lines = [
        "I want to revisit equity with my cofounder. I'm carrying more weight.",
        "I'm worried they'll take it as me being transactional.",
        "That helps. Let me try saying that.",
    ]
    coach_lines = [
        "What's the one sentence you'd want them to walk away understanding?",
        "Try: 'I want us to revisit the split because the work has shifted.'",
        "Notice you're already softer. Lead with the relationship, then the ask.",
    ]
    mock = _MockAnthropic(customer_lines, coach_lines)

    def _client_lazy(self: LLMSandboxAgent) -> _MockAnthropic:
        return mock

    monkeypatch.setattr(LLMSandboxAgent, "_client_lazy", _client_lazy)
    return mock


async def test_coach_dialogue_smoke_integration_mocked(
    mock_anthropic_factory: _MockAnthropic, tmp_path: Path
):
    config = RunConfig(
        eval_name="coach-dialogue-smoke",
        environment="voice-agent-sandbox",
        concurrency=1,
        runs_root=tmp_path,
    )
    outcome = await execute_run(config, executor=_InProcessExecutor())

    assert outcome.n_examples == 1
    assert outcome.n_ok == 1, "expected the rollout to complete cleanly"
    assert outcome.n_error == 0
    assert outcome.aggregate_scores["voice_agent_sandbox_completion"] == 1.0

    session_dir = (
        outcome.run_dir / "sessions" / "coach-dialogue-cofounder-equity-001"
    )
    transcript_path = session_dir / "transcript.jsonl"
    assert transcript_path.exists(), "transcript.jsonl should be written"

    rows = [json.loads(line) for line in transcript_path.read_text().splitlines()]
    assert len(rows) >= 4, "expected multiple turns in the transcript"

    speakers = [row["speaker"] for row in rows]
    assert "user" in speakers and "coach" in speakers, "both sides must speak"

    customer_count = sum(1 for s in speakers if s == "user")
    coach_count = sum(1 for s in speakers if s == "coach")
    assert customer_count >= 2 and coach_count >= 2, (
        f"expected ≥2 turns per side, got customer={customer_count} coach={coach_count}"
    )

    session_path = session_dir / "session.json"
    session = json.loads(session_path.read_text())
    assert session["completion_status"] == "complete"
    assert session["consent"] == "granted"

    customer_calls = [
        c for c in mock_anthropic_factory.calls
        if "simulating a person" in c["system_prefix"]
    ]
    coach_calls = [
        c for c in mock_anthropic_factory.calls
        if "simulating a person" not in c["system_prefix"]
    ]
    assert customer_calls, "customer agent should have called Anthropic"
    assert coach_calls, "coach agent should have called Anthropic"


async def test_voice_agent_smoke_still_uses_scripted_stub(tmp_path: Path):
    """Backward-compat: the existing eval keeps default scripted+stub agents."""
    config = RunConfig(
        eval_name="voice-agent-smoke",
        environment="voice-agent-sandbox",
        concurrency=1,
        runs_root=tmp_path,
    )
    outcome = await execute_run(config, executor=_InProcessExecutor())
    assert outcome.n_ok == 1
    assert outcome.aggregate_scores["voice_agent_sandbox_completion"] == 1.0


@pytest.mark.live_api
async def test_coach_dialogue_smoke_real_llm(tmp_path: Path):
    """Real Anthropic call. Skipped without an API key.

    Asserts only that the run completes and produces a multi-turn transcript.
    Does not assert dialogue quality — that's eyeballed manually.
    """
    config = RunConfig(
        eval_name="coach-dialogue-smoke",
        environment="voice-agent-sandbox",
        concurrency=1,
        runs_root=tmp_path,
    )
    outcome = await execute_run(config, executor=_InProcessExecutor())
    assert outcome.n_ok == 1, f"real-LLM run failed: {outcome.aggregate_scores}"

    session_dir = (
        outcome.run_dir / "sessions" / "coach-dialogue-cofounder-equity-001"
    )
    rows = [
        json.loads(line)
        for line in (session_dir / "transcript.jsonl").read_text().splitlines()
    ]
    customer_turns = [r for r in rows if r["speaker"] == "user"]
    coach_turns = [r for r in rows if r["speaker"] == "coach"]
    assert len(customer_turns) >= 2
    assert len(coach_turns) >= 2
    assert all(r["text"].strip() for r in rows), "every turn should have text"
