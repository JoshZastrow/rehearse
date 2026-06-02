"""ConversationBridge — cross-wires two (backend, bus) pairs for synthetic sessions.

Routes AudioChunk frames from each bus into send_caller_audio() on the opposite
backend, creating a full-duplex audio loop between two Moshi endpoints.
"""
from __future__ import annotations

import asyncio

from rehearse.bus import FrameBus
from rehearse.frames import AudioChunk, EndOfCall


class ConversationBridge:
    """Cross-wire two ConversationBackend instances via their FrameBuses.

    provider_bus → caller_backend.send_caller_audio()
    caller_bus   → provider_backend.send_caller_audio()
    """

    def __init__(
        self,
        *,
        provider_backend,
        provider_bus: FrameBus,
        caller_backend,
        caller_bus: FrameBus,
    ) -> None:
        self._provider_backend = provider_backend
        self._provider_bus = provider_bus
        self._caller_backend = caller_backend
        self._caller_bus = caller_bus
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Start routing tasks and yield so subscribers register before returning."""
        self._tasks = [
            asyncio.create_task(
                self._route(self._provider_bus, self._caller_backend),
                name="bridge-provider-to-caller",
            ),
            asyncio.create_task(
                self._route(self._caller_bus, self._provider_backend),
                name="bridge-caller-to-provider",
            ),
        ]
        # Yield twice: once to enter _route, once to reach queue.get() inside subscribe().
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    async def _route(self, source_bus: FrameBus, target_backend) -> None:
        async for frame in source_bus.subscribe():
            if isinstance(frame, AudioChunk):
                await target_backend.send_caller_audio(frame.pcm16_16k)
            elif isinstance(frame, EndOfCall):
                return

    async def close(self) -> None:
        """Cancel routing tasks and wait for them to finish."""
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
