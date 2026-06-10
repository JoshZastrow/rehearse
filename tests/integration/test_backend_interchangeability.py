"""Backend equivalence integration test.

Verifies that the managed backend, given a synthetic caller, emits the
required frame types. Skipped unless MANAGED_API_KEY is set (live_api marker).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from rehearse.backends.factory import create_backend
from rehearse.bus import FrameBus
from rehearse.config import RuntimeConfig
from rehearse.frames import AudioChunk, TranscriptDelta
from rehearse.types import Speaker

from tests.integration.conftest import FrameCollector, SyntheticCaller


def _build_managed_config(session_root: Path) -> RuntimeConfig:
    api_key = os.environ.get("MANAGED_API_KEY", "")
    if not api_key:
        pytest.skip("MANAGED_API_KEY not set")
    return RuntimeConfig(
        twilio_account_sid="x",
        twilio_auth_token="x",
        twilio_from_number="+1",
        public_base_url="https://example.com",
        hume_api_key=api_key,
        hume_config_id=os.environ.get("MANAGED_CONFIG_ID", ""),
        session_root=session_root,
        backend_type="managed",
    )


@pytest.mark.live_api
@pytest.mark.asyncio
async def test_managed_backend_produces_required_frames(tmp_path: Path) -> None:
    """Managed backend emits the required frame types for a synthetic call."""
    config = _build_managed_config(tmp_path)
    backend = create_backend(config)
    bus = FrameBus("test-managed")
    collector = FrameCollector(bus)
    collect_task = asyncio.create_task(collector.run())

    caller = SyntheticCaller(num_chunks=50)

    async with backend:
        await backend.start("test-managed", bus)
        for chunk in caller.audio_chunks():
            await backend.send_caller_audio(chunk)
            await asyncio.sleep(0)
        await backend.close()

    await bus.aclose()
    await collect_task

    frame_types = {type(f).__name__ for f in collector.frames}
    assert "TranscriptDelta" in frame_types, f"No TranscriptDelta. Got: {frame_types}"
    assert "AudioChunk" in frame_types, f"No AudioChunk. Got: {frame_types}"

    user_finals = [
        f for f in collector.frames
        if isinstance(f, TranscriptDelta) and f.speaker == Speaker.USER and f.is_final
    ]
    assert len(user_finals) >= 1


@pytest.mark.asyncio
async def test_pipeline_backend_frame_contract(tmp_path: Path) -> None:
    """PipelineBackend emits required frame types.

    Uses a scripted override of _build_and_run_pipecat_pipeline so
    no ML models are loaded. Verifies bus wiring and frame publishing.
    """
    import time
    import uuid

    from rehearse.backends.pipeline import PipelineBackend
    from rehearse.backends.tts import SilenceTTSService

    class ScriptedPipelineBackend(PipelineBackend):
        async def _build_and_run_pipecat_pipeline(self, bus: FrameBus) -> None:
            """Publish a canned frame sequence without running any model."""
            now = time.time()
            uid = uuid.uuid4().hex[:8]
            await bus.publish(TranscriptDelta(
                session_id=self._session_id,
                utterance_id=uid,
                speaker=Speaker.USER,
                text="scripted user utterance",
                is_final=True,
                ts_start=now,
                ts_end=now,
            ))
            await bus.publish(AudioChunk(
                session_id=self._session_id,
                speaker=Speaker.GUIDE,
                pcm16_16k=b"\x00" * 640,
                ts=now,
            ))
            await asyncio.Event().wait()

    backend = ScriptedPipelineBackend(
        stt_model="whisper-tiny",
        tts_service=SilenceTTSService(),
    )
    bus = FrameBus("test-pipeline-contract")
    collector = FrameCollector(bus)
    collect_task = asyncio.create_task(collector.run())

    caller = SyntheticCaller(num_chunks=5)

    async with backend:
        await backend.start("test-pipeline-contract", bus)
        for chunk in caller.audio_chunks():
            await backend.send_caller_audio(chunk)
            await asyncio.sleep(0)
        await asyncio.sleep(0.05)

    await bus.aclose()
    await collect_task

    frame_types = {type(f).__name__ for f in collector.frames}
    assert "TranscriptDelta" in frame_types, f"Missing TranscriptDelta. Got: {frame_types}"
    assert "AudioChunk" in frame_types, f"Missing AudioChunk. Got: {frame_types}"

    user_finals = [
        f for f in collector.frames
        if isinstance(f, TranscriptDelta)
        and f.speaker == Speaker.USER
        and f.is_final
    ]
    assert len(user_finals) >= 1


@pytest.mark.slow
@pytest.mark.pipeline
@pytest.mark.asyncio
async def test_pipeline_backend_produces_required_frames(tmp_path: Path) -> None:
    """PipelineBackend with whisper-tiny + SilenceTTS produces required frames
    when given real speech audio.

    Marks: @slow (loads whisper-tiny ~150MB, PocketTTS ~200MB on first run).
    Run explicitly: pytest -m pipeline tests/integration/
    """
    from rehearse.backends.pipeline import PipelineBackend
    from rehearse.backends.tts import SilenceTTSService
    from rehearse.bus import FrameBus
    from rehearse.frames import AudioChunk, TranscriptDelta
    from rehearse.types import Speaker

    from tests.integration.conftest import FrameCollector, SpeechCaller

    caller = SpeechCaller(phrase="Hello, this is a test. One two three.")
    await caller.prepare()

    backend = PipelineBackend(
        stt_model="whisper-tiny",
        tts_service=SilenceTTSService(),
        clm_url="http://localhost:0/chat/completions",
    )
    bus = FrameBus("test-pipeline-live")
    collector = FrameCollector(bus)
    collect_task = asyncio.create_task(collector.run())

    async with backend:
        await backend.start("test-pipeline-live", bus)
        for chunk in caller.audio_chunks():
            await backend.send_caller_audio(chunk)
            await asyncio.sleep(0)
        # 600ms of silence to trigger VAD turn boundary
        silence_chunk = b"\x00" * 640
        for _ in range(30):
            await backend.send_caller_audio(silence_chunk)
            await asyncio.sleep(0.02)
        await asyncio.sleep(2.0)

    await bus.aclose()
    await collect_task

    frame_types = {type(f).__name__ for f in collector.frames}
    assert "TranscriptDelta" in frame_types, (
        f"No TranscriptDelta. Got: {frame_types}. "
        "Check that faster-whisper is installed."
    )
    assert "AudioChunk" in frame_types, f"No AudioChunk. Got: {frame_types}"

    user_finals = [
        f for f in collector.frames
        if isinstance(f, TranscriptDelta)
        and f.speaker == Speaker.USER
        and f.is_final
    ]
    assert len(user_finals) >= 1
    all_text = " ".join(f.text for f in user_finals).lower()
    assert any(word in all_text for word in ("hello", "test", "one", "two", "three")), (
        f"Transcription seems wrong: {all_text!r}"
    )
