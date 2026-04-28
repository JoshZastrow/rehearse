"""Async fanout bus for runtime frames."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from rehearse.frames import Frame

_SENTINEL = object()


class FrameBus:
    """Publish runtime frames to one or more independent subscribers."""

    def __init__(self, session_id: str, maxsize: int = 256) -> None:
        self.session_id = session_id
        self.maxsize = maxsize
        self._closed = False
        self._lock = asyncio.Lock()
        self._subscribers: list[asyncio.Queue[Frame | object]] = []

    async def publish(self, frame: Frame) -> None:
        """Fan a frame out to every current subscriber."""

        async with self._lock:
            if self._closed:
                return
            subscribers = list(self._subscribers)
        for queue in subscribers:
            await queue.put(frame)

    async def aclose(self) -> None:
        """Stop the bus and terminate all subscribers."""

        async with self._lock:
            if self._closed:
                return
            self._closed = True
            subscribers = list(self._subscribers)
            self._subscribers.clear()
        for queue in subscribers:
            await queue.put(_SENTINEL)

    async def subscribe(self) -> AsyncIterator[Frame]:
        """Yield frames until the bus is closed."""

        queue: asyncio.Queue[Frame | object] = asyncio.Queue(maxsize=self.maxsize)
        async with self._lock:
            if self._closed:
                await queue.put(_SENTINEL)
            else:
                self._subscribers.append(queue)
        try:
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    return
                yield item
        finally:
            async with self._lock:
                if queue in self._subscribers:
                    self._subscribers.remove(queue)
