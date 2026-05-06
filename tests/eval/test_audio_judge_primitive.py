"""Tests for the `AudioJudge` primitive — Gemini-backed audio LLM judge.

Mirrors the `LLMJudge` pattern: a thin wrapper that calls a multimodal
LLM with audio inputs + a structured prompt, parses the response as
JSON, and surfaces parse failures cleanly.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import pytest

from rehearse.eval.scorers.audio_judge import AudioJudge, AudioJudgeError, StubAudioJudge


def _silent_wav(path: Path, *, duration_s: float = 0.5, sample_rate: int = 16_000) -> Path:
    """Write a small silent WAV at `path` for use as fixture audio."""
    n_samples = int(duration_s * sample_rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * n_samples)
    return path


@pytest.mark.asyncio
async def test_stub_audio_judge_returns_configured_payload() -> None:
    payload = {"affect_perception": {"score": 0.7, "rationale": "calm voice"}}
    judge = StubAudioJudge(response=payload)

    result = await judge.judge(system="sys", user="u", audio_paths=[])

    assert result == payload


@pytest.mark.asyncio
async def test_stub_audio_judge_raises_on_configured_error(tmp_path: Path) -> None:
    judge = StubAudioJudge(error=AudioJudgeError("simulated parse failure"))
    with pytest.raises(AudioJudgeError, match="simulated"):
        await judge.judge(system="s", user="u", audio_paths=[])


@pytest.mark.asyncio
async def test_stub_audio_judge_records_audio_paths_passed(tmp_path: Path) -> None:
    user_wav = _silent_wav(tmp_path / "user.wav")
    coach_wav = _silent_wav(tmp_path / "coach.wav")
    judge = StubAudioJudge(response={})

    await judge.judge(system="s", user="u", audio_paths=[user_wav, coach_wav])

    assert judge.last_audio_paths == [user_wav, coach_wav]
    assert judge.last_system == "s"
    assert judge.last_user == "u"


@pytest.mark.asyncio
async def test_audio_judge_parses_json_block_from_response_text(tmp_path: Path) -> None:
    """Real AudioJudge with a fake client extracts JSON from the model's text."""

    class _FakeClient:
        async def generate(
            self, *, model: str, system: str, user: str, audio_paths: list[Path]
        ) -> str:
            return (
                'Here is my analysis. {"affect_perception": '
                '{"score": 0.6, "rationale": "neutral tone"}} done.'
            )

    judge = AudioJudge(model="gemini-2.5-pro", client=_FakeClient())
    out = await judge.judge(system="s", user="u", audio_paths=[])
    assert out["affect_perception"]["score"] == 0.6
    assert "neutral" in out["affect_perception"]["rationale"]


@pytest.mark.asyncio
async def test_audio_judge_raises_on_unparseable_response() -> None:
    class _FakeClient:
        async def generate(self, *, model, system, user, audio_paths):  # type: ignore[no-untyped-def]
            return "no json here at all"

    judge = AudioJudge(model="gemini-2.5-pro", client=_FakeClient())
    with pytest.raises(AudioJudgeError, match="JSON"):
        await judge.judge(system="s", user="u", audio_paths=[])


@pytest.mark.asyncio
async def test_audio_judge_raises_on_client_failure() -> None:
    class _FakeClient:
        async def generate(self, *, model, system, user, audio_paths):  # type: ignore[no-untyped-def]
            raise RuntimeError("network down")

    judge = AudioJudge(model="gemini-2.5-pro", client=_FakeClient())
    with pytest.raises(AudioJudgeError, match="network"):
        await judge.judge(system="s", user="u", audio_paths=[])


@pytest.mark.asyncio
async def test_audio_judge_passes_audio_paths_to_client(tmp_path: Path) -> None:
    received: dict[str, Any] = {}

    class _FakeClient:
        async def generate(
            self, *, model, system, user, audio_paths
        ):  # type: ignore[no-untyped-def]
            received["paths"] = list(audio_paths)
            received["model"] = model
            return '{"x": 1}'

    user_wav = _silent_wav(tmp_path / "u.wav")
    coach_wav = _silent_wav(tmp_path / "c.wav")
    judge = AudioJudge(model="gemini-2.5-pro", client=_FakeClient())

    await judge.judge(system="s", user="u", audio_paths=[user_wav, coach_wav])

    assert received["paths"] == [user_wav, coach_wav]
    assert received["model"] == "gemini-2.5-pro"
