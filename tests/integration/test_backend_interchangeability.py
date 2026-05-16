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
