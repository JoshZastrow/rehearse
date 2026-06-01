"""Tests for InteractiveBackend — WebSocket proxy to Modal interactive server."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_interactive_backend_start_sends_handshake():
    """start() should connect to the endpoint and send the session handshake."""
    from rehearse.backends.interactive import InteractiveBackend
    from rehearse.bus import FrameBus

    async def _empty_iter():
        return
        yield

    mock_ws = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=False)
    mock_ws.__aiter__ = MagicMock(return_value=_empty_iter())

    with patch("rehearse.backends.interactive.websockets.connect", return_value=mock_ws):
        backend = InteractiveBackend(endpoint="wss://example.modal.run/ws")
        bus = FrameBus(session_id="sess-1")
        await backend.start("sess-1", bus)
        await asyncio.sleep(0.05)
        await backend.close()

    sent = mock_ws.send.call_args_list
    assert len(sent) >= 1
    handshake = json.loads(sent[0].args[0])
    assert handshake["type"] == "start"
    assert handshake["session_id"] == "sess-1"


@pytest.mark.asyncio
async def test_send_caller_audio_forwards_bytes():
    """send_caller_audio() should send raw bytes over the WebSocket."""
    from rehearse.backends.interactive import InteractiveBackend
    from rehearse.bus import FrameBus

    done = asyncio.Event()

    class _BlockingIter:
        def __aiter__(self):
            return self

        async def __anext__(self):
            await done.wait()
            raise StopAsyncIteration

    mock_ws = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=False)
    mock_ws.__aiter__ = MagicMock(return_value=_BlockingIter())

    with patch("rehearse.backends.interactive.websockets.connect", return_value=mock_ws):
        backend = InteractiveBackend(endpoint="wss://example.modal.run/ws")
        bus = FrameBus(session_id="sess-2")
        await backend.start("sess-2", bus)

        pcm = b"\x00" * 640
        await backend.send_caller_audio(pcm)
        await asyncio.sleep(0.05)  # let sender drain queue
        done.set()
        await backend.close()

    binary_sends = [c for c in mock_ws.send.call_args_list if isinstance(c.args[0], bytes)]
    assert any(c.args[0] == pcm for c in binary_sends)


@pytest.mark.asyncio
async def test_binary_message_published_as_audio_chunk():
    """Binary WebSocket messages should be published as AudioChunk(COACH) frames."""
    from rehearse.backends.interactive import InteractiveBackend
    from rehearse.bus import FrameBus
    from rehearse.frames import AudioChunk
    from rehearse.types import Speaker

    pcm_payload = b"\x01\x02" * 320

    async def _fake_iter():
        yield pcm_payload

    mock_ws = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=False)
    mock_ws.__aiter__ = MagicMock(return_value=_fake_iter())

    received: list[AudioChunk] = []

    with patch("rehearse.backends.interactive.websockets.connect", return_value=mock_ws):
        backend = InteractiveBackend(endpoint="wss://example.modal.run/ws")
        bus = FrameBus(session_id="sess-3")

        async def _collect():
            async for frame in bus.subscribe():
                if isinstance(frame, AudioChunk):
                    received.append(frame)
                    return

        collector = asyncio.create_task(_collect())
        await backend.start("sess-3", bus)
        await asyncio.sleep(0.1)
        await backend.close()
        collector.cancel()

    assert any(f.speaker == Speaker.COACH and f.pcm16_16k == pcm_payload for f in received)


@pytest.mark.asyncio
async def test_transcript_event_published_as_transcript_delta():
    """JSON transcript events should be published as TranscriptDelta(COACH) frames."""
    from rehearse.backends.interactive import InteractiveBackend
    from rehearse.bus import FrameBus
    from rehearse.frames import TranscriptDelta
    from rehearse.types import Speaker

    event = json.dumps({
        "type": "transcript",
        "utterance_id": "u-1",
        "speaker": "coach",
        "text": "Hello",
        "is_final": True,
    })

    async def _fake_iter():
        yield event

    mock_ws = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=False)
    mock_ws.__aiter__ = MagicMock(return_value=_fake_iter())

    received: list[TranscriptDelta] = []

    with patch("rehearse.backends.interactive.websockets.connect", return_value=mock_ws):
        backend = InteractiveBackend(endpoint="wss://example.modal.run/ws")
        bus = FrameBus(session_id="sess-4")

        async def _collect():
            async for frame in bus.subscribe():
                if isinstance(frame, TranscriptDelta):
                    received.append(frame)
                    return

        collector = asyncio.create_task(_collect())
        await backend.start("sess-4", bus)
        await asyncio.sleep(0.1)
        await backend.close()
        collector.cancel()

    assert any(f.speaker == Speaker.COACH and f.text == "Hello" and f.is_final for f in received)


@pytest.mark.asyncio
async def test_session_stored_event_published():
    """JSON session_stored events should be published as SessionStoredEvent frames."""
    from rehearse.backends.interactive import InteractiveBackend
    from rehearse.bus import FrameBus
    from rehearse.frames import SessionStoredEvent

    event = json.dumps({
        "type": "session_stored",
        "session_id": "sess-5",
        "volume_path": "/mnt/sessions/sess-5",
        "artifacts": ["caller_stream.pcm", "provider_stream.pcm"],
    })

    async def _fake_iter():
        yield event

    mock_ws = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=False)
    mock_ws.__aiter__ = MagicMock(return_value=_fake_iter())

    received: list[SessionStoredEvent] = []

    with patch("rehearse.backends.interactive.websockets.connect", return_value=mock_ws):
        backend = InteractiveBackend(endpoint="wss://example.modal.run/ws")
        bus = FrameBus(session_id="sess-5")

        async def _collect():
            async for frame in bus.subscribe():
                if isinstance(frame, SessionStoredEvent):
                    received.append(frame)
                    return

        collector = asyncio.create_task(_collect())
        await backend.start("sess-5", bus)
        await asyncio.sleep(0.1)
        await backend.close()
        collector.cancel()

    assert any(f.volume_path == "/mnt/sessions/sess-5" for f in received)
