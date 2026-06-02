"""Live Modal integration test: ConversationBridge with two real Moshi endpoints.

Requires both ProviderServer and CallerServer deployed and reachable.
Run with:
    INTERACTIVE_PROVIDER_ENDPOINT=wss://... \
    INTERACTIVE_CALLER_ENDPOINT=wss://... \
    pytest tests/integration/test_interactive_caller.py -m live_modal -v
"""
from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from rehearse.backends.interactive.bridge import ConversationBridge
from rehearse.backends.interactive.modal_backend import ModalInteractiveBackend
from rehearse.bus import FrameBus
from rehearse.frames import AudioChunk, EndOfCall, TranscriptDelta


def _endpoints() -> tuple[str, str]:
    provider = os.environ.get("INTERACTIVE_PROVIDER_ENDPOINT", "")
    caller = os.environ.get("INTERACTIVE_CALLER_ENDPOINT", "")
    if not provider or not caller:
        pytest.skip(
            "INTERACTIVE_PROVIDER_ENDPOINT and INTERACTIVE_CALLER_ENDPOINT must be set"
        )
    return provider, caller


@pytest.mark.live_modal
@pytest.mark.asyncio
async def test_bridge_audio_flows_both_ways():
    """Both endpoints must generate audio within 10 seconds of seeding."""
    provider_url, caller_url = _endpoints()

    session_id = str(uuid.uuid4())
    provider_bus = FrameBus(session_id=session_id)
    caller_bus = FrameBus(session_id=session_id + "-caller")

    provider_chunks: list[AudioChunk] = []
    caller_chunks: list[AudioChunk] = []

    async def _collect_provider() -> None:
        async for frame in provider_bus.subscribe():
            if isinstance(frame, AudioChunk):
                provider_chunks.append(frame)
            elif isinstance(frame, EndOfCall):
                return

    async def _collect_caller() -> None:
        async for frame in caller_bus.subscribe():
            if isinstance(frame, AudioChunk):
                caller_chunks.append(frame)
            elif isinstance(frame, EndOfCall):
                return

    provider_backend = ModalInteractiveBackend(endpoint=provider_url)
    caller_backend = ModalInteractiveBackend(endpoint=caller_url)
    bridge = ConversationBridge(
        provider_backend=provider_backend,
        provider_bus=provider_bus,
        caller_backend=caller_backend,
        caller_bus=caller_bus,
    )

    collect_p = asyncio.create_task(_collect_provider())
    collect_c = asyncio.create_task(_collect_caller())

    await provider_backend.start(session_id, provider_bus)
    await caller_backend.start(session_id + "-caller", caller_bus)
    await bridge.start()

    # Seed caller to start the loop
    await caller_backend.send_caller_audio(b"\x00" * 3200)

    await asyncio.sleep(10.0)

    # Cancel collectors first so they are not blocked on bus.subscribe() during teardown
    collect_p.cancel()
    collect_c.cancel()
    await asyncio.gather(collect_p, collect_c, return_exceptions=True)

    await bridge.close()
    await caller_backend.close()
    await provider_backend.close()
    await provider_bus.aclose()
    await caller_bus.aclose()

    assert len(provider_chunks) > 0, "Provider endpoint generated no audio in 10s"
    assert len(caller_chunks) > 0, "Caller endpoint generated no audio in 10s"


@pytest.mark.live_modal
@pytest.mark.asyncio
async def test_bridge_transcript_appears_within_30s():
    """A final TranscriptDelta must appear on the provider bus within 30 seconds."""
    provider_url, caller_url = _endpoints()

    session_id = str(uuid.uuid4())
    provider_bus = FrameBus(session_id=session_id)
    caller_bus = FrameBus(session_id=session_id + "-caller")

    transcripts: list[TranscriptDelta] = []

    async def _collect() -> None:
        async for frame in provider_bus.subscribe():
            if isinstance(frame, TranscriptDelta) and frame.is_final:
                transcripts.append(frame)
                return
            elif isinstance(frame, EndOfCall):
                return

    provider_backend = ModalInteractiveBackend(endpoint=provider_url)
    caller_backend = ModalInteractiveBackend(endpoint=caller_url)
    bridge = ConversationBridge(
        provider_backend=provider_backend,
        provider_bus=provider_bus,
        caller_backend=caller_backend,
        caller_bus=caller_bus,
    )

    collect_task = asyncio.create_task(_collect())

    await provider_backend.start(session_id, provider_bus)
    await caller_backend.start(session_id + "-caller", caller_bus)
    await bridge.start()

    await caller_backend.send_caller_audio(b"\x00" * 3200)

    try:
        await asyncio.wait_for(collect_task, timeout=30.0)
    except asyncio.TimeoutError:
        collect_task.cancel()
        try:
            await collect_task
        except asyncio.CancelledError:
            pass

    await bridge.close()
    await caller_backend.close()
    await provider_backend.close()
    await provider_bus.aclose()
    await caller_bus.aclose()

    assert len(transcripts) >= 1, "No final transcript from provider within 30s"
