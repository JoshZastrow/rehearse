from __future__ import annotations

import asyncio
import base64
import io
import struct
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from hume.empathic_voice.types.assistant_message import AssistantMessage
from hume.empathic_voice.types.audio_output import AudioOutput
from hume.empathic_voice.types.user_message import UserMessage
from hume.empathic_voice.types.web_socket_error import WebSocketError

from rehearse.bus import FrameBus
from rehearse.frames import AudioChunk, EndOfCall, ProsodyEvent, TranscriptDelta
from rehearse.services.hume_evi import HumeEVIClient


class FakeHumeSocket:
    def __init__(self, events: list[object]) -> None:
        self._events = list(events)
        self.sent_audio_inputs: list[object] = []

    def __aiter__(self) -> AsyncIterator[object]:
        return self

    async def __anext__(self) -> object:
        if not self._events:
            raise StopAsyncIteration
        event = self._events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event

    async def send_audio_input(self, message: object) -> None:
        self.sent_audio_inputs.append(message)


def _connect_factory(sockets: list[FakeHumeSocket]):
    @asynccontextmanager
    async def connect(**_kwargs):
        yield sockets.pop(0)

    return connect


def _wav_payload(samples: bytes, sample_rate: int = 48_000) -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


async def _drain(bus: FrameBus) -> list[object]:
    frames: list[object] = []
    async for frame in bus.subscribe():
        frames.append(frame)
    return frames


@pytest.mark.asyncio
async def test_hume_client_send_audio_uses_audio_input_message() -> None:
    socket = FakeHumeSocket([])
    client = HumeEVIClient(
        api_key="api",
        config_id="cfg",
        bus=FrameBus("s1"),
        session_id="s1",
        connect_fn=_connect_factory([socket]),
    )

    async with client:
        await client.send_audio(struct.pack("<4h", 1, 2, 3, 4))

    assert len(socket.sent_audio_inputs) == 1
    assert socket.sent_audio_inputs[0].type == "audio_input"


@pytest.mark.asyncio
async def test_hume_client_maps_user_assistant_and_audio_events() -> None:
    pcm48k = struct.pack("<6h", 0, 100, 200, 300, 400, 500)
    socket = FakeHumeSocket(
        [
            UserMessage.model_validate(
                {
                    "type": "user_message",
                    "from_text": False,
                    "interim": False,
                    "message": {"role": "user", "content": "hello"},
                    "models": {
                        "prosody": {
                            "scores": {
                                "Anger": 0.1,
                                "Joy": 0.8,
                                **{name: 0.0 for name in _OTHER_EMOTIONS},
                            }
                        }
                    },
                    "time": {"begin": 100, "end": 200},
                }
            ),
            AssistantMessage.model_validate(
                {
                    "type": "assistant_message",
                    "from_text": False,
                    "is_quick_response": False,
                    "message": {"role": "assistant", "content": "hi there"},
                    "models": {},
                }
            ),
            AudioOutput.model_validate(
                {
                    "type": "audio_output",
                    "id": "ao1",
                    "index": 0,
                    "data": _wav_payload(pcm48k),
                }
            ),
        ]
    )
    bus = FrameBus("s1")
    client = HumeEVIClient(
        api_key="api",
        config_id="cfg",
        bus=bus,
        session_id="s1",
        connect_fn=_connect_factory([socket]),
    )

    async with client:
        consume = asyncio.create_task(_drain(bus))
        await asyncio.sleep(0)
        await client.run_event_loop()
        await bus.aclose()
        frames = await consume

    assert [type(frame) for frame in frames] == [
        TranscriptDelta,
        ProsodyEvent,
        TranscriptDelta,
        AudioChunk,
    ]
    assert frames[0].text == "hello"
    assert frames[1].scores.emotions["joy"] == 0.8
    assert frames[2].text == "hi there"
    assert len(frames[3].pcm16_16k) > 0


@pytest.mark.asyncio
async def test_hume_client_reconnects_once_then_emits_end_of_call() -> None:
    first = FakeHumeSocket([RuntimeError("socket drop")])
    second = FakeHumeSocket(
        [
            WebSocketError.model_validate(
                {"type": "error", "code": "x", "message": "bad", "slug": "bad"}
            )
        ]
    )
    bus = FrameBus("s1")
    client = HumeEVIClient(
        api_key="api",
        config_id="cfg",
        bus=bus,
        session_id="s1",
        connect_fn=_connect_factory([first, second]),
        reconnect_backoff_s=0.0,
    )

    async with client:
        consume = asyncio.create_task(_drain(bus))
        await asyncio.sleep(0)
        await client.run_event_loop()
        await bus.aclose()
        frames = await consume

    assert isinstance(frames[-1], EndOfCall)
    assert frames[-1].reason == "error"


_OTHER_EMOTIONS = [
    "Admiration",
    "Adoration",
    "Aesthetic Appreciation",
    "Amusement",
    "Anxiety",
    "Awe",
    "Awkwardness",
    "Boredom",
    "Calmness",
    "Concentration",
    "Confusion",
    "Contemplation",
    "Contempt",
    "Contentment",
    "Craving",
    "Desire",
    "Determination",
    "Disappointment",
    "Disgust",
    "Distress",
    "Doubt",
    "Ecstasy",
    "Embarrassment",
    "Empathic Pain",
    "Entrancement",
    "Envy",
    "Excitement",
    "Fear",
    "Guilt",
    "Horror",
    "Interest",
    "Love",
    "Nostalgia",
    "Pain",
    "Pride",
    "Realization",
    "Relief",
    "Romance",
    "Sadness",
    "Satisfaction",
    "Shame",
    "Surprise (negative)",
    "Surprise (positive)",
    "Sympathy",
    "Tiredness",
    "Triumph",
]
