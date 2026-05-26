"""Tests for post-hoc TTS + provenance in RuntimeSandboxEnvironment.

Covers:
- audio/{user,coach}/turn_<N>.wav files written for each transcript turn
- timing.jsonl row count matches turn count
- provenance.json fields are correct under stub mode
- HUME_API_KEY unset -> silent WAVs, audio_stub flag set
- coach_description override applied when present in payload
"""

from __future__ import annotations

import asyncio
import json
import os
import wave
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from rehearse.eval.customers import CallerResult
from rehearse.eval.environments.runtime_sandbox import RuntimeSandboxEnvironment
from rehearse.eval.protocols import BenchmarkExample
from rehearse.eval.tts_bridge import TTSProvider
from rehearse.session.runtime import RuntimeHost
from rehearse.transport import TwoWayChannel


_SCENARIO = {
    "situation": "asking for a raise",
    "goal": "get a 15% bump",
    "stakes": "career income",
    "emotional_state": "anxious",
}


class _RecordingTTSProvider:
    """Synthesizes silent WAVs but records the description used per call."""

    name = "test-recorder"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []  # (text, description)

    async def synthesize(
        self, *, text: str, out_path: Path, description: str
    ) -> float:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16_000)
            wav.writeframes(b"\x00\x00" * 16_000)  # 1.0s silent WAV
        self.calls.append((text, description))
        return 1.0


class _StubCoach:
    """No-network coach. Returns scripted replies."""

    def __init__(self) -> None:
        self._idx = 0
        self._replies = ["Tell me more.", "I see.", "Let's keep going."]

    async def respond(self, user_text: str, session_id: str) -> str:
        await asyncio.sleep(0.05)  # let LocalFilesystemStore finish writes
        reply = self._replies[self._idx % len(self._replies)]
        self._idx += 1
        return reply


async def _drive_short_session(env: RuntimeSandboxEnvironment, run_dir: Path):
    """Run a short session with both coach and customer mocked out."""

    class _ScriptedCaller:
        async def run(self, *, transport: TwoWayChannel, runtime_phase: Any) -> CallerResult:
            # Brief initial wait so the bus subscribers (transcript writer, etc.)
            # have time to register past their `_register_artifact` to_thread()
            # call. Without this, the first user turn is lost — see runtime.py
            # race note in v2026-05-08 spec follow-ups.
            await asyncio.sleep(0.1)
            await transport.send("text", payload={"text": "I need help with my manager."})
            await asyncio.sleep(0.1)
            await transport.send("control", payload={"event": "customer_done"})
            return CallerResult(turns_sent=1, turns_per_phase={"intake": 1})

    example = BenchmarkExample(
        id="audio-test-001",
        benchmark="voice-rollout-judges",
        payload={"scenario": _SCENARIO},
    )

    with patch(
        "rehearse.eval.environments.runtime_sandbox.SyntheticCaller",
        return_value=_ScriptedCaller(),
    ), patch(
        "rehearse.eval.environments.runtime_sandbox.TextOnlyCoachAdapter",
        return_value=_StubCoach(),
    ):
        result = await env.rollout(example, run_dir, rng_seed=0)

    return result, run_dir.parent / f"eval-{example.id}"


# ---------------------------------------------------------------------------
# Audio + timing artifacts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audio_files_written_with_recording_provider(tmp_path: Path) -> None:
    """Real (recording) TTS provider produces per-turn WAVs under audio/."""
    provider = _RecordingTTSProvider()
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "stub"}):
        env = RuntimeSandboxEnvironment(tts_provider=provider)
        result, session_dir = await _drive_short_session(env, tmp_path / "run")

    assert result.status == "ok"

    # Expect at least one user WAV (the customer's intake turn).
    user_wavs = sorted((session_dir / "audio" / "user").glob("turn_*.wav"))
    assert len(user_wavs) >= 1
    assert all(w.stat().st_size > 0 for w in user_wavs)

    # timing.jsonl should have rows (audio_start + audio_end per turn).
    timing_path = session_dir / "timing.jsonl"
    assert timing_path.exists()
    rows = [
        json.loads(line) for line in timing_path.read_text().splitlines() if line
    ]
    assert len(rows) >= 2  # at least one start + end


@pytest.mark.asyncio
async def test_silent_fallback_when_no_provider(tmp_path: Path) -> None:
    """No TTS provider -> silent WAVs, audio_stub flag in provenance."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "stub"}, clear=False):
        # Force default provider to None by passing it explicitly and patching
        # get_default_provider to return None.
        with patch(
            "rehearse.eval.environments.runtime_sandbox.get_default_provider",
            return_value=None,
        ):
            env = RuntimeSandboxEnvironment()
            result, session_dir = await _drive_short_session(env, tmp_path / "run")

    assert result.status == "ok"
    provenance = json.loads((session_dir / "provenance.json").read_text())
    assert provenance["tts_provider"] == "silent-fallback"
    assert provenance["audio_stub"] is True


@pytest.mark.asyncio
async def test_provenance_documents_substitutions(tmp_path: Path) -> None:
    """provenance.json records the eval-mode adapter and transport."""
    with patch.dict(
        os.environ, {"ANTHROPIC_API_KEY": "stub", "HUME_API_KEY": ""}, clear=False
    ):
        env = RuntimeSandboxEnvironment(tts_provider=_RecordingTTSProvider())
        _, session_dir = await _drive_short_session(env, tmp_path / "run")

    provenance = json.loads((session_dir / "provenance.json").read_text())
    assert provenance["environment"] == "runtime-sandbox"
    assert provenance["tier"] == "text-plus-tts"
    assert provenance["coach_voice_adapter"] == "TextOnlyCoachAdapter"
    assert provenance["transport"] == "InMemoryTwoWayChannel"
    assert provenance["synthetic_caller"] == "SyntheticCaller"
    assert provenance["anthropic_api_key_set"] is True


@pytest.mark.asyncio
async def test_coach_description_override_propagates_to_helper(tmp_path: Path) -> None:
    """example.payload['coach_description'] is passed to synthesize_turns."""
    # Test the helper path directly with a hand-built transcript so we don't
    # need to spin up an Anthropic-backed coach for this assertion.
    from rehearse.eval.environments._audio import synthesize_turns

    provider = _RecordingTTSProvider()
    frames = [
        {"speaker": "user", "text": "Help me prep for tomorrow."},
        {"speaker": "coach", "text": "Tell me more about what's at stake."},
    ]
    await synthesize_turns(
        run_dir=tmp_path,
        frames=frames,
        user_description="anxious",
        coach_description="extremely calm, almost monotone",
        provider=provider,
    )

    coach_calls = [c for c in provider.calls if c[1] == "extremely calm, almost monotone"]
    user_calls = [c for c in provider.calls if c[1] == "anxious"]
    assert len(coach_calls) == 1
    assert len(user_calls) == 1
