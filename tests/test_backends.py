"""Unit tests for ConversationBackend protocol, ManagedBackend, and factory."""

from __future__ import annotations

import asyncio

import pytest


def test_persona_spec_is_constructable():
    from rehearse.backends.base import PersonaSpec
    spec = PersonaSpec(
        name="Alex",
        gender="male",
        system_prompt="You are Alex.",
        voice_ref="voice-123",
    )
    assert spec["name"] == "Alex"
    assert spec["gender"] == "male"
    assert spec["voice_ref"] == "voice-123"


def test_persona_spec_voice_ref_is_optional():
    from rehearse.backends.base import PersonaSpec
    spec = PersonaSpec(name="Sam", gender="female", system_prompt="You are Sam.", voice_ref=None)
    assert spec["voice_ref"] is None


def test_conversation_backend_protocol_exists():
    from rehearse.backends.base import ConversationBackend
    # Structural check: protocol has the five required methods
    assert hasattr(ConversationBackend, "start")
    assert hasattr(ConversationBackend, "send_caller_audio")
    assert hasattr(ConversationBackend, "inject_speech")
    assert hasattr(ConversationBackend, "swap_persona")
    assert hasattr(ConversationBackend, "close")


@pytest.mark.asyncio
async def test_null_prosody_service_returns_zeroed_scores():
    from rehearse.backends.prosody import NullProsodyService
    from rehearse.types import ProsodyScores
    svc = NullProsodyService()
    scores = await svc.score(b"\x00" * 320)
    assert isinstance(scores, ProsodyScores)
    assert scores.arousal == 0.0
    assert scores.valence == 0.0
    assert scores.emotions == {}


def test_prosody_service_protocol_exists():
    from rehearse.backends.prosody import ProsodyService
    assert hasattr(ProsodyService, "score")


@pytest.mark.asyncio
async def test_managed_backend_satisfies_conversation_backend_protocol():
    from rehearse.backends.base import ConversationBackend
    from rehearse.backends.managed import ManagedBackend
    backend = ManagedBackend(api_key="k", config_id="c")
    assert isinstance(backend, ConversationBackend)


@pytest.mark.asyncio
async def test_managed_backend_start_delegates_to_hume_client():
    """start() opens the HumeEVIClient and launches run_event_loop as a task."""
    from rehearse.backends.managed import ManagedBackend
    from rehearse.bus import FrameBus

    started: list[str] = []
    closed: list[str] = []

    class FakeHumeEVIClient:
        async def __aenter__(self):
            started.append("entered")
            return self

        async def __aexit__(self, *_):
            closed.append("exited")

        async def run_event_loop(self, bus: FrameBus) -> None:
            import asyncio
            await asyncio.Event().wait()  # block until cancelled

        async def send_audio(self, pcm: bytes) -> None:
            pass

        async def say(self, text: str) -> None:
            pass

        async def send_session_settings(self, *, voice_id: str, system_prompt: str | None) -> None:
            pass

    backend = ManagedBackend(api_key="k", config_id="c")
    backend._client = FakeHumeEVIClient()   # inject before start() so lazy creation is skipped

    bus = FrameBus("s1")
    async with backend:
        await backend.start("s1", bus)
        assert started == ["entered"]

    assert closed == ["exited"]


@pytest.mark.asyncio
async def test_managed_backend_send_caller_audio_delegates():
    from rehearse.backends.managed import ManagedBackend
    from rehearse.bus import FrameBus

    received: list[bytes] = []

    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *_): pass
        async def run_event_loop(self, bus): import asyncio; await asyncio.Event().wait()
        async def send_audio(self, pcm: bytes) -> None: received.append(pcm)
        async def say(self, text): pass
        async def send_session_settings(self, **_): pass

    backend = ManagedBackend(api_key="k", config_id="c")
    backend._client = FakeClient()
    bus = FrameBus("s1")

    async with backend:
        await backend.start("s1", bus)
        await backend.send_caller_audio(b"\x00\x01")
        assert received == [b"\x00\x01"]


@pytest.mark.asyncio
async def test_managed_backend_inject_speech_calls_say():
    from rehearse.backends.managed import ManagedBackend
    from rehearse.bus import FrameBus

    spoken: list[str] = []

    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *_): pass
        async def run_event_loop(self, bus): import asyncio; await asyncio.Event().wait()
        async def send_audio(self, pcm): pass
        async def say(self, text: str) -> None: spoken.append(text)
        async def send_session_settings(self, **_): pass

    backend = ManagedBackend(api_key="k", config_id="c")
    backend._client = FakeClient()
    bus = FrameBus("s1")

    async with backend:
        await backend.start("s1", bus)
        await backend.inject_speech("Hello caller.")
        assert spoken == ["Hello caller."]


@pytest.mark.asyncio
async def test_managed_backend_swap_persona_calls_session_settings():
    from rehearse.backends.base import PersonaSpec
    from rehearse.backends.managed import ManagedBackend
    from rehearse.bus import FrameBus

    settings_calls: list[dict] = []

    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *_): pass
        async def run_event_loop(self, bus): import asyncio; await asyncio.Event().wait()
        async def send_audio(self, pcm): pass
        async def say(self, text): pass
        async def send_session_settings(self, *, voice_id: str, system_prompt: str | None) -> None:
            settings_calls.append({"voice_id": voice_id, "system_prompt": system_prompt})

    backend = ManagedBackend(api_key="k", config_id="c")
    backend._client = FakeClient()
    bus = FrameBus("s1")

    persona: PersonaSpec = {
        "name": "Alex",
        "gender": "male",
        "system_prompt": "You are Alex.",
        "voice_ref": "voice-alex-123",
    }

    async with backend:
        await backend.start("s1", bus)
        await backend.swap_persona(persona)
        assert settings_calls == [{"voice_id": "voice-alex-123", "system_prompt": "You are Alex."}]


@pytest.mark.asyncio
async def test_managed_backend_say_delegates_to_inject_speech():
    """say(SpeakRequest) satisfies VoiceSpeaker protocol."""
    from rehearse.backends.managed import ManagedBackend
    from rehearse.bus import FrameBus
    from rehearse.participants import SpeakRequest

    spoken: list[str] = []

    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *_): pass
        async def run_event_loop(self, bus): import asyncio; await asyncio.Event().wait()
        async def send_audio(self, pcm): pass
        async def say(self, text): spoken.append(text)
        async def send_session_settings(self, **_): pass

    backend = ManagedBackend(api_key="k", config_id="c")
    backend._client = FakeClient()
    bus = FrameBus("s1")

    async with backend:
        await backend.start("s1", bus)
        await backend.say(SpeakRequest(text="Hi there."))
        assert spoken == ["Hi there."]


def test_create_backend_managed_returns_managed_backend():
    from pathlib import Path
    from rehearse.backends.factory import create_backend
    from rehearse.backends.managed import ManagedBackend
    from rehearse.config import RuntimeConfig
    cfg = RuntimeConfig(
        twilio_account_sid="x", twilio_auth_token="x",
        twilio_from_number="+1", public_base_url="https://x.com",
        hume_api_key="k", hume_config_id="c", session_root=Path("/tmp"),
        backend_type="managed",
    )
    backend = create_backend(cfg)
    assert isinstance(backend, ManagedBackend)


def test_create_backend_unknown_raises():
    from pathlib import Path
    from rehearse.backends.factory import create_backend
    from rehearse.config import RuntimeConfig
    cfg = RuntimeConfig(
        twilio_account_sid="x", twilio_auth_token="x",
        twilio_from_number="+1", public_base_url="https://x.com",
        hume_api_key="k", hume_config_id="c", session_root=Path("/tmp"),
        backend_type="unknown",
    )
    with pytest.raises(ValueError, match="Unknown backend_type"):
        create_backend(cfg)


@pytest.mark.asyncio
async def test_bus_publisher_translates_transcription_frame():
    """Pipecat TranscriptionFrame → Rehearse TranscriptDelta(is_final=True)."""
    from rehearse.backends.bus_publisher import BusPublisher
    from rehearse.bus import FrameBus
    from rehearse.frames import TranscriptDelta
    from rehearse.types import Speaker

    bus = FrameBus("s1")
    frames: list = []

    async def collect():
        async for f in bus.subscribe():
            frames.append(f)

    collect_task = asyncio.create_task(collect())
    await asyncio.sleep(0.01)  # Let subscriber attach

    publisher = BusPublisher(session_id="s1", bus=bus)

    class TranscriptionFrame:
        text = "hello there"
        user_id = "user"

    await publisher.on_transcription(TranscriptionFrame())
    await bus.aclose()
    await collect_task

    assert len(frames) == 1
    assert isinstance(frames[0], TranscriptDelta)
    assert frames[0].speaker == Speaker.USER
    assert frames[0].is_final is True
    assert frames[0].text == "hello there"


@pytest.mark.asyncio
async def test_bus_publisher_translates_interim_frame():
    """Pipecat InterimTranscriptionFrame → TranscriptDelta(is_final=False)."""
    from rehearse.backends.bus_publisher import BusPublisher
    from rehearse.bus import FrameBus
    from rehearse.frames import TranscriptDelta
    from rehearse.types import Speaker

    bus = FrameBus("s1")
    frames: list = []

    async def collect():
        async for f in bus.subscribe():
            frames.append(f)

    collect_task = asyncio.create_task(collect())
    await asyncio.sleep(0.01)  # Let subscriber attach

    publisher = BusPublisher(session_id="s1", bus=bus)

    class InterimTranscriptionFrame:
        text = "hel"

    await publisher.on_interim_transcription(InterimTranscriptionFrame())
    await bus.aclose()
    await collect_task

    assert frames[0].is_final is False
    assert frames[0].speaker == Speaker.USER


@pytest.mark.asyncio
async def test_pipeline_backend_satisfies_protocol():
    """PipelineBackend satisfies ConversationBackend structural protocol."""
    from rehearse.backends.base import ConversationBackend
    from rehearse.backends.pipeline import PipelineBackend
    backend = PipelineBackend(
        speech_mode="modular",
        stt_model="whisper-tiny",
        tts_model="kokoro",
        clm_url="http://localhost:0/chat/completions",
    )
    assert isinstance(backend, ConversationBackend)
    assert hasattr(backend, "start")
    assert hasattr(backend, "send_caller_audio")
    assert hasattr(backend, "inject_speech")
    assert hasattr(backend, "swap_persona")
    assert hasattr(backend, "close")


def test_create_backend_pipeline_returns_pipeline_backend():
    from pathlib import Path
    from rehearse.backends.factory import create_backend
    from rehearse.backends.pipeline import PipelineBackend
    from rehearse.config import RuntimeConfig
    cfg = RuntimeConfig(
        twilio_account_sid="x", twilio_auth_token="x",
        twilio_from_number="+1", public_base_url="https://x.com",
        hume_api_key="k", hume_config_id="c", session_root=Path("/tmp"),
        backend_type="pipeline",
    )
    backend = create_backend(cfg)
    assert isinstance(backend, PipelineBackend)


def test_tts_service_protocol_exists():
    from rehearse.backends.tts import TTSService
    assert hasattr(TTSService, "synthesize")
    assert hasattr(TTSService, "set_voice")


@pytest.mark.asyncio
async def test_silence_tts_returns_pcm16_bytes():
    from rehearse.backends.tts import SilenceTTSService
    svc = SilenceTTSService()
    result = await svc.synthesize("Hello.")
    assert isinstance(result, bytes)
    assert len(result) > 0
    assert len(result) % 2 == 0  # PCM16 = 2 bytes per sample


@pytest.mark.asyncio
async def test_silence_tts_set_voice_is_noop():
    from rehearse.backends.tts import SilenceTTSService
    svc = SilenceTTSService()
    await svc.set_voice("any-voice")
    await svc.set_voice(None)
