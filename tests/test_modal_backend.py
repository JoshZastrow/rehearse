"""Tests for ModalInteractiveBackend.

modal_backend.py has no torch/faster_whisper dependencies, so it can be
imported cleanly. We stub rehearse.backends.interactive at the package level
so that the __init__.py (which imports the torch-dependent InteractiveBackend)
is never executed.
"""
from __future__ import annotations

import sys
import types

import pytest

# Prevent rehearse.backends.interactive.__init__ from importing InteractiveBackend,
# which pulls in torch and faster_whisper. We insert a lightweight package stub
# before any test code imports the submodule.
if "rehearse.backends.interactive" not in sys.modules:
    import importlib.util
    _pkg = types.ModuleType("rehearse.backends.interactive")
    _spec = importlib.util.find_spec("rehearse.backends.interactive")
    _pkg.__path__ = list(_spec.submodule_search_locations) if _spec else []  # type: ignore[attr-defined]
    _pkg.__package__ = "rehearse.backends.interactive"
    sys.modules["rehearse.backends.interactive"] = _pkg

from rehearse.backends.interactive.modal_backend import ModalInteractiveBackend  # noqa: E402
from rehearse.audio.participants import SpeakRequest  # noqa: E402
from rehearse.bus import FrameBus  # noqa: E402
from rehearse.frames import EndOfCall  # noqa: E402


@pytest.mark.asyncio
async def test_modal_backend_say_delegates_to_inject_speech():
    """ModalInteractiveBackend.say() must call inject_speech with the request text.

    Regression: PersonaSwapCoordinator calls speaker.say(SpeakRequest) at every
    phase transition (conversation.py passes speaker=backend). If ModalInteractiveBackend
    lacks a say() method the call raises AttributeError and the persona swap is silently
    skipped for the whole session.
    """
    backend = ModalInteractiveBackend(endpoint="ws://fake")

    injected: list[str] = []

    async def _fake_inject(text: str) -> None:
        injected.append(text)

    backend.inject_speech = _fake_inject  # type: ignore[method-assign]

    await backend.say(SpeakRequest(text="Okay, let's run it."))

    assert injected == ["Okay, let's run it."]


@pytest.mark.asyncio
async def test_modal_backend_start_publishes_end_of_call_on_connect_failure():
    """start() must not raise when the Modal endpoint is unreachable.

    Regression: websockets.connect() raises InvalidStatus (HTTP 404) when the Modal
    app isn't running. That exception escaped through backend_task in conversation.py
    and crashed the ASGI WebSocket handler. The fix retries _CONNECT_ATTEMPTS times,
    then publishes EndOfCall so the session tears down through the normal path.
    """
    import asyncio
    from unittest.mock import patch, AsyncMock

    backend = ModalInteractiveBackend(endpoint="wss://does-not-exist.example.com/ws")
    backend._CONNECT_RETRY_DELAY = 0.0  # skip sleeps in tests
    bus = FrameBus(session_id="test-connect-fail")

    frames: list = []

    async def _collect() -> None:
        async for frame in bus.subscribe():
            frames.append(frame)

    collect_task = asyncio.create_task(_collect())
    await asyncio.sleep(0)  # let _collect() advance to queue.get() before publish

    with patch(
        "rehearse.backends.interactive.modal_backend.websockets.connect",
        side_effect=OSError("connection refused"),
    ) as mock_connect:
        await backend.start("test-connect-fail", bus)

    await bus.aclose()
    await collect_task

    assert mock_connect.call_count == backend._CONNECT_ATTEMPTS
    assert any(isinstance(f, EndOfCall) for f in frames)
    assert frames[-1].reason == "error"


@pytest.mark.asyncio
async def test_modal_backend_maps_provider_label_to_coach():
    """'provider' wire label from the refactored server must map to Speaker.COACH."""
    import asyncio

    backend = ModalInteractiveBackend(endpoint="ws://fake")
    bus = FrameBus(session_id="test-provider-label")
    frames: list = []

    async def collect():
        async for frame in bus.subscribe():
            frames.append(frame)

    collect_task = asyncio.create_task(collect())
    await asyncio.sleep(0)

    backend._session_id = "test-provider-label"
    backend._bus = bus

    await backend._handle_event({
        "type": "transcript",
        "utterance_id": "u1",
        "speaker": "provider",
        "text": "Hello",
        "is_final": True,
    })
    await bus.aclose()
    await collect_task

    from rehearse.frames import TranscriptDelta
    from rehearse.types import Speaker
    assert any(
        isinstance(f, TranscriptDelta) and f.speaker == Speaker.COACH
        for f in frames
    ), f"Expected Speaker.COACH for 'provider' label, got {frames}"


@pytest.mark.asyncio
async def test_modal_backend_maps_caller_label_to_user():
    """'caller' wire label from the refactored server must map to Speaker.USER."""
    import asyncio

    backend = ModalInteractiveBackend(endpoint="ws://fake")
    bus = FrameBus(session_id="test-caller-label")
    frames: list = []

    async def collect():
        async for frame in bus.subscribe():
            frames.append(frame)

    collect_task = asyncio.create_task(collect())
    await asyncio.sleep(0)

    backend._session_id = "test-caller-label"
    backend._bus = bus

    await backend._handle_event({
        "type": "transcript",
        "utterance_id": "u2",
        "speaker": "caller",
        "text": "Hi",
        "is_final": True,
    })
    await bus.aclose()
    await collect_task

    from rehearse.frames import TranscriptDelta
    from rehearse.types import Speaker
    assert any(
        isinstance(f, TranscriptDelta) and f.speaker == Speaker.USER
        for f in frames
    ), f"Expected Speaker.USER for 'caller' label, got {frames}"
