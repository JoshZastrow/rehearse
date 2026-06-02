"""Two-backend session runner for synthetic caller/provider conversations.

Wires two ModalInteractiveBackend instances together via ConversationBridge
and runs until EndOfCall or max_duration_sec, whichever comes first.
"""
from __future__ import annotations

import asyncio
import dataclasses
import time
from pathlib import Path

from rehearse.backends.interactive.bridge import ConversationBridge
from rehearse.backends.interactive.modal_backend import ModalInteractiveBackend
from rehearse.bus import FrameBus
from rehearse.frames import EndOfCall


@dataclasses.dataclass
class SessionResult:
    session_id: str
    duration_sec: float
    end_reason: str
    run_dir: Path | None = None


async def run_interactive_session(
    *,
    session_id: str,
    provider_endpoint: str,
    caller_endpoint: str,
    max_duration_sec: float = 120.0,
    run_dir: Path | None = None,
) -> SessionResult:
    """Run one synthetic session between a provider and caller Moshi endpoint.

    Seeds the caller with 100ms of silence to start the loop, then waits for
    EndOfCall on the provider bus or max_duration_sec timeout.
    """
    provider_bus = FrameBus(session_id=session_id)
    caller_bus = FrameBus(session_id=session_id + "-caller")

    provider_backend = ModalInteractiveBackend(endpoint=provider_endpoint)
    caller_backend = ModalInteractiveBackend(endpoint=caller_endpoint)

    bridge = ConversationBridge(
        provider_backend=provider_backend,
        provider_bus=provider_bus,
        caller_backend=caller_backend,
        caller_bus=caller_bus,
    )

    t_start = time.monotonic()
    end_reason = "timeout"

    await provider_backend.start(session_id, provider_bus)
    await caller_backend.start(session_id + "-caller", caller_bus)
    await bridge.start()

    # Seed caller with silence so Moshi starts generating audio
    silence_100ms = b"\x00" * 3200  # 100ms at 16kHz PCM16 (3200 bytes)
    await caller_backend.send_caller_audio(silence_100ms)

    try:
        async with asyncio.timeout(max_duration_sec):
            async for frame in provider_bus.subscribe():
                if isinstance(frame, EndOfCall):
                    end_reason = frame.reason
                    break
    except asyncio.TimeoutError:
        pass
    finally:
        await bridge.close()
        await caller_backend.close()
        await provider_backend.close()
        await provider_bus.aclose()
        await caller_bus.aclose()

    return SessionResult(
        session_id=session_id,
        duration_sec=time.monotonic() - t_start,
        end_reason=end_reason,
        run_dir=run_dir,
    )
