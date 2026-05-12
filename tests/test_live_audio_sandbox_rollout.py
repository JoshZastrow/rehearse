from __future__ import annotations

import json
from pathlib import Path

import pytest

from rehearse.eval.environments.live_audio_sandbox import LiveAudioSandboxEnvironment
from rehearse.eval.protocols import BenchmarkExample
from rehearse.runtime import (
    CoachAudioEvent,
    CoachTranscriptEvent,
    CoachTurnComplete,
    StubAudioCoachAdapter,
)


class _StubTTS:
    name = "stub-tts"

    async def synthesize(
        self,
        *,
        text: str,
        out_path: Path,
        description: str | None = None,
    ) -> float:
        raise AssertionError("live audio sandbox should stream PCM, not post-hoc WAV")

    async def synthesize_pcm(self, *, text: str, description: str | None = None) -> bytes:
        return b"\x01\x00" * 1600


class _StubLLM:
    async def complete(self, *, system_prompt: str, history: list[tuple[str, str]]) -> str:
        return "I need help with a hard conversation."


def _example() -> BenchmarkExample:
    return BenchmarkExample(
        id="live-audio-001",
        benchmark="voice-rollout-judges",
        payload={
            "scenario": {
                "situation": "asking for a raise",
                "goal": "get a raise",
                "stakes": "rent",
                "emotional_state": "anxious",
            }
        },
    )


def test_live_audio_sandbox_preflight_requires_keys_without_factory() -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("HUME_API_KEY", raising=False)
        mp.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="HUME_API_KEY"):
            LiveAudioSandboxEnvironment()


@pytest.mark.asyncio
async def test_live_audio_sandbox_completes_with_stub_adapter(tmp_path: Path) -> None:
    env = LiveAudioSandboxEnvironment(
        coach_adapter_factory=lambda: StubAudioCoachAdapter(
            scripted_events=[
                CoachTranscriptEvent(text="Let's work through it.", utterance_id="coach-1"),
                CoachAudioEvent(pcm16_16k=b"\x00\x00" * 1600),
                CoachTurnComplete(),
            ]
        ),
        tts_provider=_StubTTS(),
        llm_client=_StubLLM(),
        customer_max_turns=1,
    )

    result = await env.rollout(_example(), tmp_path / "run", rng_seed=0)

    assert result.status == "ok"
    assert result.artifacts_dir is not None
    assert (result.artifacts_dir / "session.json").exists()
    assert (result.artifacts_dir / "transcript.jsonl").exists()
    assert (result.artifacts_dir / "phase_timing.json").exists()
    assert (result.artifacts_dir / "audio" / "coach" / "turn_0.wav").exists()

    provenance = json.loads((result.artifacts_dir / "provenance.json").read_text())
    assert provenance["environment"] == "live-audio-sandbox"
    assert provenance["tier"] == "live-audio"
    assert provenance["audio_stub"] is False
