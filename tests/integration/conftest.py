"""Fixtures for backend integration tests.

FrameCollector records all frames from a FrameBus in order.
SyntheticCaller produces a fixed sequence of PCM audio chunks.
"""

from __future__ import annotations

import os
import struct
from collections.abc import Iterator

import pytest

from rehearse.bus import FrameBus
from rehearse.frames import Frame


class FrameCollector:
    """Subscribe to a FrameBus and collect every frame published to it."""

    def __init__(self, bus: FrameBus) -> None:
        self.frames: list[Frame] = []
        self._bus = bus

    async def run(self) -> None:
        """Consume bus frames until the bus closes."""
        async for frame in self._bus.subscribe():
            self.frames.append(frame)


class SyntheticCaller:
    """Produce a fixed sequence of PCM16/16kHz audio chunks.

    Each chunk is 20ms of silence (640 bytes = 320 samples x 2 bytes).
    """

    def __init__(self, num_chunks: int = 50) -> None:
        self._num_chunks = num_chunks
        self._chunk = struct.pack("<320h", *([0] * 320))

    def audio_chunks(self) -> Iterator[bytes]:
        for _ in range(self._num_chunks):
            yield self._chunk


@pytest.fixture
def synthetic_caller() -> SyntheticCaller:
    return SyntheticCaller(num_chunks=50)


@pytest.fixture
def frame_collector_factory():
    """Return a factory that creates a FrameCollector for a given bus."""
    def _make(bus: FrameBus) -> FrameCollector:
        return FrameCollector(bus)
    return _make


class SpeechCaller:
    """Generate PCM16/16kHz caller audio from a text phrase using PocketTTS.

    Used for live pipeline tests where Whisper needs actual speech to transcribe.
    """

    def __init__(self, phrase: str = "Hello, this is a test. One two three.") -> None:
        self._phrase = phrase
        self._audio: bytes | None = None

    async def prepare(self) -> None:
        """Pre-generate the speech audio (loads PocketTTS model once)."""
        from rehearse.backends.tts import PocketTTSService
        svc = PocketTTSService(voice_ref="alba")
        self._audio = await svc.synthesize(self._phrase)

    def audio_chunks(self, chunk_size: int = 640) -> Iterator[bytes]:
        """Yield PCM16/16kHz chunks of the pre-generated speech audio."""
        if self._audio is None:
            raise RuntimeError("Call prepare() first")
        for i in range(0, len(self._audio), chunk_size):
            yield self._audio[i : i + chunk_size]


@pytest.fixture
async def speech_caller() -> SpeechCaller:
    caller = SpeechCaller()
    await caller.prepare()
    return caller


@pytest.fixture
def vllm_base_url() -> str:
    """Return VLLM_BASE_URL from the environment.

    Skips the test if the variable is not set.  Set it to the Modal-deployed
    endpoint, e.g.::

        export VLLM_BASE_URL=https://<workspace>--rehearse-gemma-judge-serve.modal.run/v1
    """
    url = os.environ.get("VLLM_BASE_URL", "")
    if not url:
        pytest.skip("VLLM_BASE_URL not set — skipping live Modal test")
    return url
