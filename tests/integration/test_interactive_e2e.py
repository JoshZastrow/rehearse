"""End-to-end test for InteractiveBackend with real model weights.

Skipped automatically if weights are absent or SKIP_INTERACTIVE_E2E=1.

Run:
    uv run python -m pytest tests/integration/test_interactive_e2e.py -v -s

CPU note: Moshi inference on CPU takes ~20s per 80ms frame (250x slower than
real-time). test_produces_audio waits up to 90s for the first output frame.
Use a GPU machine for fast tests.
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
    not _WEIGHTS_PRESENT or os.environ.get("SKIP_INTERACTIVE_E2E") == "1",
    reason="Model weights not present — run scripts/download_moshi_weights.py first",
)


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@skip_no_weights
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_interactive_backend_produces_audio_and_transcript():
    """Feed silence until AudioChunk(COACH) frames arrive.

    On GPU: first output in <5s. On CPU: ~40s (two 20s forward passes).
    Test waits up to 90s to accommodate CPU-only CI.
    """
    from rehearse.backends.interactive import InteractiveBackend
    from rehearse.bus import FrameBus
    from rehearse.frames import AudioChunk, TranscriptDelta
    from rehearse.types import Speaker

    device = _device()
    backend = InteractiveBackend(
        checkpoint_path=str(_WEIGHTS_DIR),
        hf_repo="kyutai/moshiko-pytorch-bf16",
        device=device,
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

        # Feed silence continuously; the inference thread processes frames as fast
        # as it can. On CPU each 80ms frame takes ~20s, so we keep feeding.
        silence_chunk = (np.zeros(320, dtype=np.int16)).tobytes()
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline:
            await backend.send_caller_audio(silence_chunk)
            audio_frames = [f for f in collected if isinstance(f, AudioChunk) and f.speaker == Speaker.COACH]
            if audio_frames:
                break
            await asyncio.sleep(0.01)

    collector.cancel()

    audio_frames = [f for f in collected if isinstance(f, AudioChunk) and f.speaker == Speaker.COACH]
    assert audio_frames, "No AudioChunk(COACH) frames received within 90s"

    for f in audio_frames[:3]:
        assert isinstance(f.pcm16_16k, bytes)
        assert len(f.pcm16_16k) > 0

    print(f"\n  device={device}")
    print(f"  AudioChunk(COACH) frames received: {len(audio_frames)}")
    total_audio_ms = sum(len(f.pcm16_16k) // 2 for f in audio_frames) / 16  # samples → ms
    print(f"  Total coach audio: {total_audio_ms:.0f} ms")

    transcript = [f for f in collected if isinstance(f, TranscriptDelta) and f.speaker == Speaker.COACH]
    print(f"  TranscriptDelta(COACH) frames: {len(transcript)}")


@skip_no_weights
@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_interactive_backend_closes_cleanly():
    """Verify close() terminates the inference loop without hanging.

    Sends fewer chunks than one full Moshi frame (< 4 × 320 samples) so the
    inference thread stays in the queue-wait loop and can respond to the stop
    sentinel immediately — no long forward pass to wait out.
    """
    from rehearse.backends.interactive import InteractiveBackend
    from rehearse.bus import FrameBus
    from rehearse.frames import EndOfCall

    device = _device()
    backend = InteractiveBackend(
        checkpoint_path=str(_WEIGHTS_DIR),
        hf_repo="kyutai/moshiko-pytorch-bf16",
        device=device,
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
        # 3 chunks × 320 samples = 960 samples at 16kHz → 1440 at 24kHz
        # Frame size is 1920, so no complete frame is queued — thread stays idle.
        for _ in range(3):
            await backend.send_caller_audio(silence)

    elapsed = time.monotonic() - t0

    # Let event loop drain the run_coroutine_threadsafe-scheduled EndOfCall
    await asyncio.sleep(0.2)
    collector.cancel()

    assert elapsed < 10.0, f"close() took {elapsed:.1f}s — likely hung"

    end_frames = [f for f in collected if isinstance(f, EndOfCall)]
    assert end_frames, "EndOfCall frame never published"
    print(f"\n  device={device}, close() completed in {elapsed:.2f}s")
    print(f"  EndOfCall reason: {end_frames[0].reason}")
