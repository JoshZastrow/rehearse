"""End-to-end test for MoshiBackend with real model weights.

Skipped automatically if weights are absent or SKIP_MOSHI_E2E=1.

Run:
    uv run python -m pytest tests/integration/test_moshi_e2e.py -v -s
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import numpy as np
import pytest

_WEIGHTS_DIR = Path(__file__).parents[2] / "rehearse" / "models" / "kyutai" / "moshiko-pytorch-bf16"
_REQUIRED_FILES = [
    "model.safetensors",
    "tokenizer-e351c8d8-checkpoint125.safetensors",
    "tokenizer_spm_32k_3.model",
]
_WEIGHTS_PRESENT = all((_WEIGHTS_DIR / f).exists() for f in _REQUIRED_FILES)

skip_no_weights = pytest.mark.skipif(
    not _WEIGHTS_PRESENT or os.environ.get("SKIP_MOSHI_E2E") == "1",
    reason="Moshi weights not present — run scripts/download_moshi_weights.py first",
)


@skip_no_weights
@pytest.mark.asyncio
async def test_moshi_backend_produces_audio_and_transcript():
    """Feed 2 seconds of silence; assert AudioChunk(COACH) frames arrive within 5s."""
    from rehearse.backends.moshi import MoshiBackend
    from rehearse.bus import FrameBus
    from rehearse.frames import AudioChunk, TranscriptDelta
    from rehearse.types import Speaker

    backend = MoshiBackend(
        checkpoint_path=str(_WEIGHTS_DIR),
        hf_repo="kyutai/moshiko-pytorch-bf16",
        device="cpu",   # use CPU so the test runs without a GPU
        asr_model="tiny",
    )
    bus = FrameBus(session_id="e2e-test")

    collected: list[object] = []
    async def _collect():
        async for frame in bus.subscribe():
            collected.append(frame)

    collector = asyncio.create_task(_collect())

    async with backend:
        await backend.start("e2e-test", bus)

        # Feed 2 seconds of silence at 16 kHz (320 samples per 20ms chunk)
        silence_chunk = (np.zeros(320, dtype=np.int16)).tobytes()
        n_chunks = int(2.0 / 0.020)   # 100 chunks = 2s
        for _ in range(n_chunks):
            await backend.send_caller_audio(silence_chunk)
            await asyncio.sleep(0)   # yield to event loop

        # Wait up to 10s for inference output
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            audio_frames = [f for f in collected if isinstance(f, AudioChunk) and f.speaker == Speaker.COACH]
            if audio_frames:
                break
            await asyncio.sleep(0.1)

    collector.cancel()

    audio_frames = [f for f in collected if isinstance(f, AudioChunk) and f.speaker == Speaker.COACH]
    assert audio_frames, "No AudioChunk(COACH) frames received within 10s"

    # Each decoded frame is 1280 samples at 16kHz = 2560 bytes
    for f in audio_frames[:3]:
        assert isinstance(f.pcm16_16k, bytes)
        assert len(f.pcm16_16k) > 0

    print(f"\n  AudioChunk(COACH) frames received: {len(audio_frames)}")
    total_audio_ms = sum(len(f.pcm16_16k) // 2 for f in audio_frames) / 16  # samples → ms
    print(f"  Total coach audio: {total_audio_ms:.0f} ms")

    transcript = [f for f in collected if isinstance(f, TranscriptDelta) and f.speaker == Speaker.COACH]
    print(f"  TranscriptDelta(COACH) frames: {len(transcript)}")


@skip_no_weights
@pytest.mark.asyncio
async def test_moshi_backend_closes_cleanly():
    """Verify close() terminates the inference loop without hanging."""
    from rehearse.backends.moshi import MoshiBackend
    from rehearse.bus import FrameBus
    from rehearse.frames import EndOfCall

    backend = MoshiBackend(
        checkpoint_path=str(_WEIGHTS_DIR),
        hf_repo="kyutai/moshiko-pytorch-bf16",
        device="cpu",
        asr_model="tiny",
    )
    bus = FrameBus(session_id="e2e-close-test")

    collected: list[object] = []
    async def _collect():
        async for frame in bus.subscribe():
            collected.append(frame)

    collector = asyncio.create_task(_collect())

    t0 = time.monotonic()
    async with backend:
        await backend.start("e2e-close-test", bus)
        silence = (np.zeros(320, dtype=np.int16)).tobytes()
        for _ in range(10):
            await backend.send_caller_audio(silence)

    elapsed = time.monotonic() - t0
    collector.cancel()

    assert elapsed < 10.0, f"close() took {elapsed:.1f}s — likely hung"

    end_frames = [f for f in collected if isinstance(f, EndOfCall)]
    assert end_frames, "EndOfCall frame never published"
    print(f"\n  close() completed in {elapsed:.2f}s")
    print(f"  EndOfCall reason: {end_frames[0].reason}")
