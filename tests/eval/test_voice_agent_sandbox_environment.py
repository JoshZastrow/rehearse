"""Voice-agent sandbox environment tests."""

from __future__ import annotations

import json
from pathlib import Path

from rehearse.eval.environments import get_environment, list_environments
from rehearse.eval.environments.voice_agent_sandbox import VoiceAgentSandboxEnvironment
from rehearse.eval.evals import get_eval, list_evals
from rehearse.eval.executors import LocalSubprocessExecutor
from rehearse.eval.protocols import BenchmarkExample
from rehearse.eval.runner import RunConfig, execute_run


def _example() -> BenchmarkExample:
    return BenchmarkExample(
        id="sandbox-ex1",
        benchmark="voice-agent-sandbox",
        payload={
            "customer_script": [
                "I need to tell my cofounder I want to revisit equity.",
                "I am worried they will get defensive.",
            ],
            "max_turns": 4,
        },
        expected={},
    )


async def test_voice_agent_sandbox_environment_rollout_writes_artifacts(tmp_path: Path):
    environment = VoiceAgentSandboxEnvironment(model_slots={})

    result = await environment.rollout(_example(), tmp_path / "rollout", rng_seed=0)

    assert result.status == "ok"
    assert result.artifacts_dir == tmp_path / "rollout"
    assert result.payload is not None
    assert result.payload["session_id"] == "sandbox-sandbox-ex1"
    assert result.payload["customer_output"] == {"turns_sent": 2}
    assert result.payload["runtime_output"] == {"turns_received": 2, "turns_sent": 2}

    session_json = json.loads((tmp_path / "rollout" / "session.json").read_text())
    assert session_json["completion_status"] == "complete"
    assert session_json["pipeline_version"] == "voice-agent-sandbox@v0"

    conversation_lines = (tmp_path / "rollout" / "conversation.jsonl").read_text().splitlines()
    assert len(conversation_lines) == 6
    assert json.loads(conversation_lines[0])["source"] == "customer"
    assert json.loads(conversation_lines[1])["source"] == "customer"

    transcript_lines = (tmp_path / "rollout" / "transcript.jsonl").read_text().splitlines()
    assert len(transcript_lines) == 4
    assert "cofounder" in json.loads(transcript_lines[0])["text"]


async def test_voice_agent_sandbox_environment_runs_through_subprocess(tmp_path: Path):
    executor = LocalSubprocessExecutor()

    result = await executor.submit(
        target_name="voice-agent-sandbox",
        target_version="v0",
        model_slots={},
        example=_example(),
        run_dir=tmp_path / "subprocess-rollout",
        timeout_s=10,
        rng_seed=0,
    )

    assert result.status == "ok"
    assert result.payload is not None
    assert result.payload["transport_events"] == 6
    assert (tmp_path / "subprocess-rollout" / "session.json").exists()


def test_voice_agent_sandbox_environment_is_registered():
    assert "voice-agent-sandbox" in list_environments()
    environment = get_environment("voice-agent-sandbox", model_slots={})
    assert environment.name == "voice-agent-sandbox"


async def test_voice_agent_smoke_eval_runs_via_runner(tmp_path: Path):
    assert "voice-agent-smoke" in list_evals()
    eval_spec = get_eval("voice-agent-smoke")
    assert eval_spec.preferred_environment == "voice-agent-sandbox"

    outcome = await execute_run(
        RunConfig(
            eval_name="voice-agent-smoke",
            environment="voice-agent-sandbox",
            runs_root=tmp_path,
        )
    )

    assert outcome.n_examples == 1
    assert outcome.n_ok == 1
    assert outcome.aggregate_scores["voice_agent_sandbox_completion"] == 1.0
    assert (outcome.run_dir / "sessions" / "voice-agent-smoke-001" / "session.json").exists()
