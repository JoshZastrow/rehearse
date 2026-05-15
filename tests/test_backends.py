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
