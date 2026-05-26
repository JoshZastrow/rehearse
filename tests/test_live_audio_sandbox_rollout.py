"""Integration tests for LiveAudioSandboxEnvironment using run_session().

These tests verify the eval runs through the real conversation pipeline
(run_session → PhaseProcessor → IntakeProcessor → writers) with a stub
backend that immediately signals end-of-call and a stub TTS provider.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rehearse.bus import FrameBus
from rehearse.eval.environments.live_audio_sandbox import LiveAudioSandboxEnvironment
from rehearse.eval.protocols import BenchmarkExample
from rehearse.frames import EndOfCall
from rehearse.session.session import utcnow


class _StubTTS:
    name = "stub-tts"

    async def synthesize(
        self,
        *,
        text: str,
        out_path: Path,
        description: str | None = None,
    ) -> float:
        raise AssertionError("live audio sandbox should use synthesize_pcm, not synthesize")

    async def synthesize_pcm(self, *, text: str, description: str | None = None) -> bytes:
        # Minimal 100ms chunk so the caller can stream something
        return b"\x01\x00" * 1600


class _StubLLM:
    """Stub LLM: one caller turn, then stop."""

    def __init__(self) -> None:
        self._calls = 0

    class _Messages:
        _calls = 0

        async def create(self, **kwargs):  # noqa: ANN001, ANN202
            self._calls += 1
            from unittest.mock import MagicMock
            resp = MagicMock()
            if self._calls <= 1:
                resp.content = [MagicMock(text="I need help with a hard conversation.")]
            else:
                resp.content = []
            resp.usage = MagicMock(input_tokens=10, output_tokens=5)
            return resp

    messages = _Messages()


class _EndingBackend:
    """Stub backend that immediately ends the call after start() is awaited."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    async def start(self, session_id: str, bus: FrameBus) -> None:
        # Give the caller one chance to send a turn, then end the call
        await asyncio.sleep(0.1)
        await bus.publish(
            EndOfCall(session_id=session_id, reason="hangup", ts=utcnow().timestamp())
        )

    async def send_caller_audio(self, pcm16_16k: bytes) -> None:
        pass

    async def inject_speech(self, text: str) -> None:
        pass

    async def swap_persona(self, persona) -> None:
        pass

    async def close(self) -> None:
        pass

    async def say(self, request) -> None:
        pass


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


def test_live_audio_sandbox_preflight_requires_keys_without_providers() -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("HUME_API_KEY", raising=False)
        mp.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError):
            LiveAudioSandboxEnvironment()


@pytest.mark.asyncio
async def test_live_audio_sandbox_completes_with_stub_providers(tmp_path: Path) -> None:
    """run_session() produces session artifacts via stub TTS + stub backend."""
    from unittest.mock import patch

    env = LiveAudioSandboxEnvironment(
        tts_provider=_StubTTS(),
        llm_client=_StubLLM(),
        customer_max_turns=1,
    )

    with patch(
        "rehearse.eval.environments.live_audio_sandbox.create_backend",
        return_value=_EndingBackend(),
    ):
        result = await env.rollout(_example(), tmp_path / "run", rng_seed=0)

    assert result.status == "ok", result.error
    assert result.artifacts_dir is not None
    assert (result.artifacts_dir / "session.json").exists()
    assert (result.artifacts_dir / "transcript.jsonl").exists()

    import json
    provenance = json.loads((result.artifacts_dir / "provenance.json").read_text())
    assert provenance["environment"] == "live-audio-sandbox"
    assert provenance["runtime_kernel"] == "real-run-session"
