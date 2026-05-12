"""Eval-side TTS bridge — synthesize speakable audio for fixture rollouts.

The audio judges (`AffectPerceptionJudgeScorer`, `DeliveryJudgeScorer`)
need real speech to score. The `audio-fixture` environment uses this
module to synthesize per-turn WAVs from transcript text. When no
provider is configured (no `HUME_API_KEY`), callers fall back to silent
WAVs so CI smoke runs stay hermetic.

Default provider: Hume Octave. The `description` field on each
utterance carries an emotion/style cue (e.g. "anxious, hesitant") that
Octave conditions on, giving the audio judges signal to grade.
"""

from __future__ import annotations

import base64
import os
import tempfile
import wave
from pathlib import Path
from typing import Protocol


class TTSProvider(Protocol):
    """Synthesize text to a WAV file on disk; return actual duration in seconds."""

    name: str

    async def synthesize(
        self,
        *,
        text: str,
        out_path: Path,
        description: str | None = None,
    ) -> float: ...

    async def synthesize_pcm(
        self,
        *,
        text: str,
        description: str | None = None,
    ) -> bytes: ...


class HumeOctaveProvider:
    """Hume Octave TTS — emotion-aware, returns WAV bytes via `synthesize_json`."""

    name = "hume-octave"

    def __init__(self, api_key: str, *, voice: str | None = None) -> None:
        self._api_key = api_key
        self._voice = voice

    async def synthesize(
        self,
        *,
        text: str,
        out_path: Path,
        description: str | None = None,
    ) -> float:
        from hume import AsyncHumeClient
        from hume.tts.types.format_wav import FormatWav
        from hume.tts.types.posted_utterance import PostedUtterance

        client = AsyncHumeClient(api_key=self._api_key)
        utterance = PostedUtterance(text=text, description=description)
        result = await client.tts.synthesize_json(
            utterances=[utterance],
            format=FormatWav(),
        )
        if not result.generations:
            raise RuntimeError("hume octave returned no generations")
        audio_b64 = result.generations[0].audio
        audio_bytes = base64.b64decode(audio_b64)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(audio_bytes)
        return _wav_duration_s(out_path)

    async def synthesize_pcm(
        self,
        *,
        text: str,
        description: str | None = None,
    ) -> bytes:
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "utterance.wav"
            await self.synthesize(text=text, out_path=wav_path, description=description)
            return _read_wav_pcm16(wav_path)


def get_default_provider() -> TTSProvider | None:
    """Return the configured provider, or None if no credentials are set.

    Set `REHEARSE_TTS_PROVIDER=none` to force the silent-fallback path
    even when an API key is present (useful for CI smoke runs that
    happen to have credentials in the environment).
    """
    if os.environ.get("REHEARSE_TTS_PROVIDER", "").lower() == "none":
        return None
    api_key = os.environ.get("HUME_API_KEY")
    if not api_key:
        return None
    return HumeOctaveProvider(api_key=api_key)


def _wav_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
    return frames / float(rate) if rate else 0.0


def _read_wav_pcm16(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        return wav.readframes(wav.getnframes())
