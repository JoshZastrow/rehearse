from __future__ import annotations

import asyncio

import pytest

from rehearse.bus import FrameBus
from rehearse.frames import AudioChunk
from rehearse.types import Speaker


@pytest.mark.asyncio
async def test_bus_fans_out_frames_to_multiple_subscribers() -> None:
    bus = FrameBus("s1")
    sub_a = bus.subscribe()
    sub_b = bus.subscribe()

    async def next_item(subscription):
        return await anext(subscription)

    task_a = asyncio.create_task(next_item(sub_a))
    task_b = asyncio.create_task(next_item(sub_b))
    await asyncio.sleep(0)
    frame = AudioChunk(session_id="s1", speaker=Speaker.USER, pcm16_16k=b"abc", ts=0.1)

    await bus.publish(frame)

    assert await task_a == frame
    assert await task_b == frame

    await sub_a.aclose()
    await sub_b.aclose()


@pytest.mark.asyncio
async def test_bus_close_ends_existing_subscribers() -> None:
    bus = FrameBus("s1")
    subscription = bus.subscribe()

    await bus.aclose()

    with pytest.raises(StopAsyncIteration):
        await anext(subscription)


@pytest.mark.asyncio
async def test_bus_subscriber_after_close_finishes_immediately() -> None:
    bus = FrameBus("s1")
    await bus.aclose()
    subscription = bus.subscribe()

    with pytest.raises(StopAsyncIteration):
        await anext(subscription)
