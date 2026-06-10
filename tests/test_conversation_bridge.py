"""Tests for ConversationBridge.

ConversationBridge cross-wires two (backend, bus) pairs:
  - AudioChunk from provider_bus → caller_backend.send_caller_audio()
  - AudioChunk from caller_bus   → provider_backend.send_caller_audio()
  - EndOfCall on either bus      → that routing task stops
"""
from __future__ import annotations

import asyncio

import pytest

from rehearse.bus import FrameBus
from rehearse.frames import AudioChunk, EndOfCall
from rehearse.types import Speaker


def _silence(n_bytes: int = 320) -> bytes:
    return b"\x00" * n_bytes


class _MockBackend:
    def __init__(self) -> None:
        self.received: list[bytes] = []

    async def send_caller_audio(self, pcm: bytes) -> None:
        self.received.append(pcm)


@pytest.mark.asyncio
async def test_bridge_routes_provider_audio_to_caller():
    """AudioChunk on provider_bus must reach caller_backend.send_caller_audio()."""
    from rehearse.backends.interactive.bridge import ConversationBridge

    provider_bus = FrameBus(session_id="p")
    caller_bus = FrameBus(session_id="c")
    provider_backend = _MockBackend()
    caller_backend = _MockBackend()

    bridge = ConversationBridge(
        provider_backend=provider_backend,
        provider_bus=provider_bus,
        caller_backend=caller_backend,
        caller_bus=caller_bus,
    )
    await bridge.start()

    pcm = _silence(640)
    await provider_bus.publish(
        AudioChunk(session_id="p", speaker=Speaker.GUIDE, pcm16_16k=pcm, ts=0.0)
    )
    await asyncio.sleep(0.01)

    assert caller_backend.received == [pcm]
    assert provider_backend.received == []

    await bridge.close()
    await provider_bus.aclose()
    await caller_bus.aclose()


@pytest.mark.asyncio
async def test_bridge_routes_caller_audio_to_provider():
    """AudioChunk on caller_bus must reach provider_backend.send_caller_audio()."""
    from rehearse.backends.interactive.bridge import ConversationBridge

    provider_bus = FrameBus(session_id="p")
    caller_bus = FrameBus(session_id="c")
    provider_backend = _MockBackend()
    caller_backend = _MockBackend()

    bridge = ConversationBridge(
        provider_backend=provider_backend,
        provider_bus=provider_bus,
        caller_backend=caller_backend,
        caller_bus=caller_bus,
    )
    await bridge.start()

    pcm = _silence(320)
    await caller_bus.publish(
        AudioChunk(session_id="c", speaker=Speaker.GUIDE, pcm16_16k=pcm, ts=0.0)
    )
    await asyncio.sleep(0.01)

    assert provider_backend.received == [pcm]
    assert caller_backend.received == []

    await bridge.close()
    await provider_bus.aclose()
    await caller_bus.aclose()


@pytest.mark.asyncio
async def test_bridge_stops_routing_on_end_of_call():
    """EndOfCall on provider_bus must stop the provider→caller routing task."""
    from rehearse.backends.interactive.bridge import ConversationBridge

    provider_bus = FrameBus(session_id="p")
    caller_bus = FrameBus(session_id="c")
    provider_backend = _MockBackend()
    caller_backend = _MockBackend()

    bridge = ConversationBridge(
        provider_backend=provider_backend,
        provider_bus=provider_bus,
        caller_backend=caller_backend,
        caller_bus=caller_bus,
    )
    await bridge.start()

    await provider_bus.publish(
        EndOfCall(session_id="p", reason="hangup", ts=0.0)
    )
    await asyncio.sleep(0.01)

    # Publish audio after EndOfCall — must NOT reach caller
    await provider_bus.publish(
        AudioChunk(session_id="p", speaker=Speaker.GUIDE, pcm16_16k=_silence(), ts=1.0)
    )
    await asyncio.sleep(0.01)

    assert caller_backend.received == []

    await bridge.close()
    await provider_bus.aclose()
    await caller_bus.aclose()
