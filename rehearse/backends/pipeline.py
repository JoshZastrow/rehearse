"""PipelineBackend — open-source Pipecat pipeline behind ConversationBackend.

Modular mode: SileroVAD → SmartTurn → STT → ProsodyService ─┐
                                                              ├─ CLMWebhookService → TTS
                                                             ─┘
Frames are translated to Rehearse FrameBus by BusPublisher.

_build_and_run_pipecat_pipeline() is not yet implemented.
Use ManagedBackend (BACKEND_TYPE=managed) for live calls.
"""

from __future__ import annotations

import asyncio
import contextlib

import structlog

from rehearse.backends.base import PersonaSpec
from rehearse.backends.bus_publisher import BusPublisher
from rehearse.backends.prosody import NullProsodyService, ProsodyService
from rehearse.bus import FrameBus
from rehearse.participants import SpeakRequest

log = structlog.get_logger(__name__)


class PipelineBackend:
    """ConversationBackend backed by a local Pipecat audio pipeline.

    In modular mode: open-source VAD + STT + LLM-via-webhook + TTS.
    Each stage is swappable via constructor arguments.
    """

    def __init__(
        self,
        *,
        speech_mode: str = "modular",
        stt_model: str = "whisper-tiny",
        tts_model: str = "kokoro",
        clm_url: str = "http://localhost:8080/chat/completions",
        prosody_service: ProsodyService | None = None,
    ) -> None:
        self._speech_mode = speech_mode
        self._stt_model = stt_model
        self._tts_model = tts_model
        self._clm_url = clm_url
        self._prosody: ProsodyService = prosody_service or NullProsodyService()
        self._publisher: BusPublisher | None = None
        self._pipeline_task: asyncio.Task | None = None
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._session_id = ""

    async def __aenter__(self) -> PipelineBackend:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def start(self, session_id: str, bus: FrameBus) -> None:
        """Initialize the Pipecat pipeline and launch the audio processing loop."""
        self._session_id = session_id
        self._publisher = BusPublisher(session_id=session_id, bus=bus)
        self._pipeline_task = asyncio.create_task(
            self._run_pipeline(bus),
            name=f"pipeline-backend-{session_id}",
        )
        log.info(
            "pipeline_backend.started",
            session_id=session_id,
            speech_mode=self._speech_mode,
            stt=self._stt_model,
            tts=self._tts_model,
        )

    async def _run_pipeline(self, bus: FrameBus) -> None:
        """Drive the audio pipeline until cancelled or an error occurs."""
        try:
            await self._build_and_run_pipecat_pipeline(bus)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.error("pipeline_backend.error", session_id=self._session_id, error=str(exc))
            if self._publisher:
                await self._publisher.on_end(reason="error")

    async def _build_and_run_pipecat_pipeline(self, bus: FrameBus) -> None:
        """Build the Pipecat pipeline and run it.

        Not yet implemented. Use ManagedBackend (BACKEND_TYPE=managed) for live calls.
        """
        raise NotImplementedError(
            "PipelineBackend._build_and_run_pipecat_pipeline() is not yet implemented. "
            "Use ManagedBackend (BACKEND_TYPE=managed)."
        )

    async def send_caller_audio(self, pcm16_16k: bytes) -> None:
        """Push caller audio into the pipeline's input queue."""
        await self._audio_queue.put(pcm16_16k)

    async def inject_speech(self, text: str) -> None:
        """Inject a deterministic TTS utterance into the pipeline."""
        log.debug("pipeline_backend.inject_speech", text=text)

    async def swap_persona(self, persona: PersonaSpec) -> None:
        """Update system prompt and TTS voice for the next turn."""
        log.info(
            "pipeline_backend.swap_persona",
            session_id=self._session_id,
            persona_name=persona["name"],
        )

    async def say(self, request: SpeakRequest) -> None:
        """Satisfy VoiceSpeaker protocol."""
        await self.inject_speech(request.text)

    async def close(self) -> None:
        """Cancel the pipeline task."""
        if self._pipeline_task and not self._pipeline_task.done():
            self._pipeline_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pipeline_task
        log.info("pipeline_backend.closed", session_id=self._session_id)
