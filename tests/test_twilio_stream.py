from __future__ import annotations

import base64
import struct

import pytest

from rehearse.audio.mulaw import encode_pcm16
from rehearse.audio.twilio_stream import TwilioStream


class FakeWebSocket:
    def __init__(self, inbound_events: list[object]) -> None:
        self._inbound = list(inbound_events)
        self.sent: list[object] = []

    async def receive_json(self) -> object:
        if not self._inbound:
            raise RuntimeError("no more inbound events")
        return self._inbound.pop(0)

    async def send_json(self, payload: object) -> None:
        self.sent.append(payload)


@pytest.mark.asyncio
async def test_twilio_stream_handshake_uses_custom_session_id() -> None:
    ws = FakeWebSocket(
        [
            {"event": "connected"},
            {
                "event": "start",
                "start": {
                    "streamSid": "MZ123",
                    "callSid": "CA123",
                    "customParameters": {"session_id": "session-1"},
                },
            },
        ]
    )

    async with TwilioStream(ws) as stream:
        assert stream.stream_sid == "MZ123"
        assert stream.session_id == "session-1"


@pytest.mark.asyncio
async def test_twilio_stream_handshake_falls_back_to_call_sid() -> None:
    ws = FakeWebSocket(
        [{"event": "start", "start": {"streamSid": "MZ123", "callSid": "CA123"}}]
    )

    async with TwilioStream(ws) as stream:
        assert stream.session_id == "CA123"


@pytest.mark.asyncio
async def test_twilio_stream_properties_require_handshake() -> None:
    stream = TwilioStream(FakeWebSocket([]))

    with pytest.raises(RuntimeError):
        _ = stream.stream_sid
    with pytest.raises(RuntimeError):
        _ = stream.session_id


@pytest.mark.asyncio
async def test_twilio_stream_rejects_missing_start_payload() -> None:
    ws = FakeWebSocket([{"event": "start", "start": None}])

    with pytest.raises(ValueError, match="missing start payload"):
        async with TwilioStream(ws):
            pass


@pytest.mark.asyncio
async def test_twilio_stream_rejects_missing_stream_sid() -> None:
    ws = FakeWebSocket([{"event": "start", "start": {"callSid": "CA123"}}])

    with pytest.raises(ValueError, match="missing streamSid"):
        async with TwilioStream(ws):
            pass


@pytest.mark.asyncio
async def test_twilio_stream_inbound_decodes_media_and_stops() -> None:
    pcm8k = struct.pack("<4h", 0, 1000, -1000, 0)
    payload = base64.b64encode(encode_pcm16(pcm8k)).decode("ascii")
    ws = FakeWebSocket(
        [
            {"event": "start", "start": {"streamSid": "MZ123", "callSid": "CA123"}},
            {"event": "media", "media": {"payload": payload}},
            {"event": "stop"},
        ]
    )

    async with TwilioStream(ws) as stream:
        chunks = [chunk async for chunk in stream.inbound()]

    assert len(chunks) == 1
    assert len(chunks[0]) == len(pcm8k) * 2


@pytest.mark.asyncio
async def test_twilio_stream_inbound_skips_non_dict_and_non_media_events() -> None:
    pcm8k = struct.pack("<2h", 0, 1000)
    payload = base64.b64encode(encode_pcm16(pcm8k)).decode("ascii")
    ws = FakeWebSocket(
        [
            {"event": "start", "start": {"streamSid": "MZ123", "callSid": "CA123"}},
            "not-a-dict",
            {"event": "mark"},
            {"event": "media", "media": {"payload": payload}},
            {"event": "stop"},
        ]
    )

    async with TwilioStream(ws) as stream:
        chunks = [chunk async for chunk in stream.inbound()]

    assert len(chunks) == 1


@pytest.mark.asyncio
async def test_twilio_stream_inbound_skips_invalid_media_payload() -> None:
    ws = FakeWebSocket(
        [
            {"event": "start", "start": {"streamSid": "MZ123", "callSid": "CA123"}},
            {"event": "media", "media": {"payload": "%%%not-base64%%%"}} ,
            {"event": "stop"},
        ]
    )

    async with TwilioStream(ws) as stream:
        chunks = [chunk async for chunk in stream.inbound()]

    assert chunks == []


@pytest.mark.asyncio
async def test_twilio_stream_inbound_skips_missing_media_fields() -> None:
    ws = FakeWebSocket(
        [
            {"event": "start", "start": {"streamSid": "MZ123", "callSid": "CA123"}},
            {"event": "media", "media": None},
            {"event": "media", "media": {"payload": None}},
            {"event": "stop"},
        ]
    )

    async with TwilioStream(ws) as stream:
        chunks = [chunk async for chunk in stream.inbound()]

    assert chunks == []


@pytest.mark.asyncio
async def test_twilio_stream_send_encodes_media_event() -> None:
    ws = FakeWebSocket(
        [{"event": "start", "start": {"streamSid": "MZ123", "callSid": "CA123"}}]
    )
    pcm16 = struct.pack("<8h", 0, 100, 200, 300, 400, 500, 600, 700)

    async with TwilioStream(ws) as stream:
        await stream.send(pcm16)

    assert ws.sent
    event = ws.sent[0]
    assert event["event"] == "media"
    assert event["streamSid"] == "MZ123"
    assert isinstance(event["media"]["payload"], str)


@pytest.mark.asyncio
async def test_twilio_stream_send_mark() -> None:
    ws = FakeWebSocket(
        [{"event": "start", "start": {"streamSid": "MZ123", "callSid": "CA123"}}]
    )

    async with TwilioStream(ws) as stream:
        await stream.send_mark("assistant-finished")

    assert ws.sent == [
        {
            "event": "mark",
            "streamSid": "MZ123",
            "mark": {"name": "assistant-finished"},
        }
    ]
