from __future__ import annotations

import asyncio
import base64
import io
import json
import struct
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from hume.empathic_voice.types.assistant_end import AssistantEnd
from hume.empathic_voice.types.assistant_message import AssistantMessage
from hume.empathic_voice.types.audio_output import AudioOutput
from hume.empathic_voice.types.user_message import UserMessage
from hume.empathic_voice.types.web_socket_error import WebSocketError

from rehearse.bus import FrameBus
from rehearse.frames import AudioChunk, EndOfCall, ProsodyEvent, TranscriptDelta
from rehearse.services.hume_evi import HumeEVIClient


class _Sleep:
    """Test marker: sleeping inside the event stream to advance the clock."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds


class FakeHumeSocket:
    def __init__(self, events: list[object]) -> None:
        self._events = list(events)
        self.sent_audio_inputs: list[object] = []

    def __aiter__(self) -> AsyncIterator[object]:
        return self

    async def __anext__(self) -> object:
        while self._events:
            event = self._events.pop(0)
            if isinstance(event, _Sleep):
                await asyncio.sleep(event.seconds)
                continue
            if isinstance(event, Exception):
                raise event
            return event
        raise StopAsyncIteration

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
async def test_hume_client_send_audio_requires_connection() -> None:
    client = HumeEVIClient(
        api_key="api",
        config_id="cfg",
        bus=FrameBus("s1"),
        session_id="s1",
        connect_fn=_connect_factory([FakeHumeSocket([])]),
    )

    with pytest.raises(RuntimeError, match="not connected"):
        await client.send_audio(b"\x00\x00")


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
            AssistantEnd.model_validate({"type": "assistant_end"}),
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

    # Coach TranscriptDelta is buffered until assistant_end fires, so
    # it lands AFTER the AudioChunk emitted by intervening audio_output.
    assert [type(frame) for frame in frames] == [
        TranscriptDelta,
        ProsodyEvent,
        AudioChunk,
        TranscriptDelta,
    ]
    assert frames[0].text == "hello"
    assert frames[1].scores.emotions["joy"] == 0.8
    assert len(frames[2].pcm16_16k) > 0
    assert frames[3].text == "hi there"
    # Coach turn now carries a real ts_end >= ts_start (was equal before).
    assert frames[3].ts_end >= frames[3].ts_start


@pytest.mark.asyncio
async def test_coach_transcript_records_real_duration_from_assistant_end() -> None:
    """ts_end of the coach TranscriptDelta reflects time elapsed before assistant_end."""
    socket = FakeHumeSocket(
        [
            AssistantMessage.model_validate(
                {
                    "type": "assistant_message",
                    "from_text": False,
                    "is_quick_response": False,
                    "message": {"role": "assistant", "content": "hi"},
                    "models": {},
                }
            ),
            # Sleep before assistant_end to grow the elapsed clock.
            _Sleep(0.05),
            AssistantEnd.model_validate({"type": "assistant_end"}),
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

    coach = [f for f in frames if isinstance(f, TranscriptDelta)]
    assert len(coach) == 1
    assert coach[0].text == "hi"
    assert coach[0].ts_end - coach[0].ts_start >= 0.04


@pytest.mark.asyncio
async def test_two_assistant_messages_without_end_flush_previous() -> None:
    """A second assistant_message before assistant_end flushes the first turn."""
    socket = FakeHumeSocket(
        [
            AssistantMessage.model_validate(
                {
                    "type": "assistant_message",
                    "from_text": False,
                    "is_quick_response": False,
                    "message": {"role": "assistant", "content": "first"},
                    "models": {},
                }
            ),
            _Sleep(0.02),
            AssistantMessage.model_validate(
                {
                    "type": "assistant_message",
                    "from_text": False,
                    "is_quick_response": False,
                    "message": {"role": "assistant", "content": "second"},
                    "models": {},
                }
            ),
            AssistantEnd.model_validate({"type": "assistant_end"}),
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

    coach = [f for f in frames if isinstance(f, TranscriptDelta)]
    assert [f.text for f in coach] == ["first", "second"]
    # First turn's ts_end is bounded by the second turn's ts_start.
    assert coach[0].ts_end <= coach[1].ts_start


@pytest.mark.asyncio
async def test_hume_client_swap_config_not_implemented() -> None:
    client = HumeEVIClient(
        api_key="api",
        config_id="cfg",
        bus=FrameBus("s1"),
        session_id="s1",
        connect_fn=_connect_factory([FakeHumeSocket([])]),
    )

    with pytest.raises(NotImplementedError):
        await client.swap_config("cfg-2", "prompt")


@pytest.mark.asyncio
async def test_reconnect_schedule_exhausted_emits_end_of_call() -> None:
    """All retries fail → publishes EndOfCall(error) only after schedule exhausted."""
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
        reconnect_backoff_schedule_s=(0.0,),
        reconnect_budget_s=1.0,
    )

    async with client:
        consume = asyncio.create_task(_drain(bus))
        await asyncio.sleep(0)
        await client.run_event_loop()
        await bus.aclose()
        frames = await consume

    assert isinstance(frames[-1], EndOfCall)
    assert frames[-1].reason == "error"


@pytest.mark.asyncio
async def test_reconnect_within_budget_recovers_without_end_of_call() -> None:
    """connect_fn that fails twice then succeeds yields one continuous loop."""
    first = FakeHumeSocket([RuntimeError("blip 1")])
    second = FakeHumeSocket([RuntimeError("blip 2")])
    third = FakeHumeSocket([])  # clean close
    bus = FrameBus("s1")
    client = HumeEVIClient(
        api_key="api",
        config_id="cfg",
        bus=bus,
        session_id="s1",
        connect_fn=_connect_factory([first, second, third]),
        reconnect_backoff_schedule_s=(0.0, 0.0, 0.0, 0.0),
        reconnect_budget_s=2.0,
    )

    async with client:
        consume = asyncio.create_task(_drain(bus))
        await asyncio.sleep(0)
        await client.run_event_loop()
        await bus.aclose()
        frames = await consume

    error_frames = [f for f in frames if isinstance(f, EndOfCall) and f.reason == "error"]
    assert error_frames == []


@pytest.mark.asyncio
async def test_reconnect_state_resets_after_successful_event() -> None:
    """A successful event after reconnect re-arms the budget for later failures.

    Sequence: socket A fails → reconnect to B → B yields one valid event,
    then fails → reconnect to C → C closes cleanly. Without reset, B's
    failure would count as attempt #2 against A's budget; with reset it's
    a fresh attempt. With a 1-entry schedule both reconnects must succeed
    independently or this test would fall through to EndOfCall.
    """
    first = FakeHumeSocket([RuntimeError("disconnect 1")])
    second = FakeHumeSocket(
        [
            UserMessage.model_validate(
                {
                    "type": "user_message",
                    "from_text": False,
                    "interim": False,
                    "message": {"role": "user", "content": "after first reconnect"},
                    "models": {
                        "prosody": {
                            "scores": {
                                "Anger": 0.0,
                                "Joy": 0.5,
                                **{name: 0.0 for name in _OTHER_EMOTIONS},
                            }
                        }
                    },
                    "time": {"begin": 0, "end": 100},
                }
            ),
            RuntimeError("disconnect 2"),
        ]
    )
    third = FakeHumeSocket([])
    bus = FrameBus("s1")
    client = HumeEVIClient(
        api_key="api",
        config_id="cfg",
        bus=bus,
        session_id="s1",
        connect_fn=_connect_factory([first, second, third]),
        reconnect_backoff_schedule_s=(0.0,),
        reconnect_budget_s=1.0,
    )

    async with client:
        consume = asyncio.create_task(_drain(bus))
        await asyncio.sleep(0)
        await client.run_event_loop()
        await bus.aclose()
        frames = await consume

    error_frames = [f for f in frames if isinstance(f, EndOfCall) and f.reason == "error"]
    assert error_frames == []
    transcripts = [f for f in frames if isinstance(f, TranscriptDelta)]
    assert any(f.text == "after first reconnect" for f in transcripts)


@pytest.mark.asyncio
async def test_reconnect_budget_caps_total_window() -> None:
    """Budget exhaustion ends reconnect attempts even if schedule has slots left."""
    sockets = [FakeHumeSocket([RuntimeError("fail")]) for _ in range(10)]
    bus = FrameBus("s1")
    client = HumeEVIClient(
        api_key="api",
        config_id="cfg",
        bus=bus,
        session_id="s1",
        connect_fn=_connect_factory(sockets),
        reconnect_backoff_schedule_s=(0.05, 0.05, 0.05, 0.05),
        reconnect_budget_s=0.06,
    )

    async with client:
        consume = asyncio.create_task(_drain(bus))
        await asyncio.sleep(0)
        await client.run_event_loop()
        await bus.aclose()
        frames = await consume

    assert isinstance(frames[-1], EndOfCall)
    assert frames[-1].reason == "error"


@pytest.mark.asyncio
async def test_connect_uses_mapped_config_id_for_persona(tmp_path, monkeypatch):
    from rehearse.bus import FrameBus
    from rehearse.services import hume_configs

    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps({"relationship_coach": "cfg_relationship", "default": "cfg_default"})
    )
    monkeypatch.setattr(hume_configs, "MAPPING_PATH_DEFAULT", mapping)

    captured: dict = {}

    class _FakeSocket:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    def _connect_fn(**kwargs):
        captured.update(kwargs)
        return _FakeSocket()

    bus = FrameBus("sess_test")
    client = HumeEVIClient(
        api_key="k",
        config_id="cfg_env_fallback",
        persona_key="relationship_coach",
        bus=bus,
        session_id="sess_test",
        connect_fn=_connect_fn,
    )
    async with client:
        pass

    assert captured["config_id"] == "cfg_relationship"


@pytest.mark.asyncio
async def test_connect_falls_back_to_env_config_id_when_no_mapping(tmp_path, monkeypatch):
    from rehearse.bus import FrameBus
    from rehearse.services import hume_configs

    monkeypatch.setattr(hume_configs, "MAPPING_PATH_DEFAULT", tmp_path / "missing.json")

    captured: dict = {}

    class _FakeSocket:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    def _connect_fn(**kwargs):
        captured.update(kwargs)
        return _FakeSocket()

    bus = FrameBus("sess_test")
    client = HumeEVIClient(
        api_key="k",
        config_id="cfg_env_fallback",
        persona_key="relationship_coach",
        bus=bus,
        session_id="sess_test",
        connect_fn=_connect_fn,
    )
    async with client:
        pass

    assert captured["config_id"] == "cfg_env_fallback"


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
