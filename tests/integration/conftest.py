"""Fixtures for backend integration tests.

FrameCollector records all frames from a FrameBus in order.
SyntheticCaller produces a fixed sequence of PCM audio chunks.
"""

from __future__ import annotations

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
