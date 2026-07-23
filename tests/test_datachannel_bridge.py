"""Hermetic tests for the LiveKit datachannel bridge.

The bridge forwards runtime bus frames to the browser over the LiveKit data
channel. The cold-start warmup UX depends on it forwarding ProviderReady as a
``{"type": "provider_ready"}`` message so the frontend can leave its warming
state. No LiveKit server or model needed — a capture stream records the JSON.
"""

from __future__ import annotations

import asyncio
import json

from rehearse.bus import FrameBus
from rehearse.frames import EndOfCall, ProviderReady
from rehearse.session.livekit_session import _datachannel_bridge


class _CaptureStream:
    def __init__(self) -> None:
        self.published: list[dict] = []

    async def publish_data(self, data: bytes) -> None:
        self.published.append(json.loads(data.decode()))


async def _wait_subscribed(bus: FrameBus) -> None:
    for _ in range(50):
        if bus._subscribers:  # noqa: SLF001 — test needs the subscribe barrier
            return
        await asyncio.sleep(0)


async def test_bridge_forwards_provider_ready_before_end() -> None:
    bus = FrameBus("s1")
    stream = _CaptureStream()
    task = asyncio.create_task(_datachannel_bridge(stream, bus))
    await _wait_subscribed(bus)

    await bus.publish(ProviderReady(session_id="s1", ts=1.0))
    await bus.publish(EndOfCall(session_id="s1", reason="hangup", ts=2.0))
    await asyncio.wait_for(task, timeout=2.0)

    types = [m["type"] for m in stream.published]
    assert "provider_ready" in types, f"bridge dropped provider_ready — got {types}"
    assert types.index("provider_ready") < types.index("end_of_call")
