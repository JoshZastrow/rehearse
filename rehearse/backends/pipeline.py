"""PipelineBackend — open-source audio pipeline behind ConversationBackend.

Modular mode audio loop (no full Pipecat runner required):
  Caller audio → energy VAD → faster-whisper STT → TTSService → AudioChunk(COACH)

Swap the STT model with stt_model="whisper-tiny"|"whisper-medium" etc.
Swap the TTS implementation by passing tts_service=YourTTSService().

_build_and_run_pipecat_pipeline() runs until cancelled.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid

import structlog

from rehearse.backends.base import PersonaSpec
from rehearse.backends.bus_publisher import BusPublisher
from rehearse.backends.prosody import NullProsodyService, ProsodyService
from rehearse.backends.tts import SilenceTTSService, TTSService
from rehearse.bus import FrameBus
from rehearse.frames import AudioChunk, TranscriptDelta
from rehearse.participants import SpeakRequest
from rehearse.types import Speaker

log = structlog.get_logger(__name__)

# Energy-based VAD thresholds
_SILENCE_RMS = 150          # int16 RMS below this = silence
_TURN_SILENCE_CHUNKS = 25   # 25 × 20ms = 500ms silence → turn boundary


def _rms(pcm16: bytes) -> float:
    """Return RMS amplitude of a PCM16 chunk."""
    import numpy as np
    samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    return float(np.sqrt((samples ** 2).mean())) if len(samples) else 0.0


class PipelineBackend:
    """ConversationBackend backed by a local audio processing loop.

    In modular mode: energy VAD + faster-whisper STT + swappable TTSService.
    Pass tts_service= to inject any TTSService implementation.
    """

    def __init__(
        self,
        *,
        speech_mode: str = "modular",
        stt_model: str = "whisper-tiny",
        tts_model: str = "pocket-tts",
        clm_url: str = "http://localhost:8080/chat/completions",
        prosody_service: ProsodyService | None = None,
        tts_service: TTSService | None = None,
    ) -> None:
        self._speech_mode = speech_mode
        self._stt_model = stt_model
        self._clm_url = clm_url
        self._prosody: ProsodyService = prosody_service or NullProsodyService()
        self._tts: TTSService = tts_service or self._default_tts(tts_model)
        self._publisher: BusPublisher | None = None
        self._pipeline_task: asyncio.Task | None = None
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._session_id = ""

    @staticmethod
    def _default_tts(tts_model: str) -> TTSService:
        if tts_model == "silence":
            return SilenceTTSService()
        if tts_model in ("pocket-tts", "kokoro"):
            from rehearse.backends.tts import PocketTTSService
            return PocketTTSService()
        return SilenceTTSService()

    async def __aenter__(self) -> PipelineBackend:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def start(self, session_id: str, bus: FrameBus) -> None:
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
        )
        # Yield once so any pending subscribers (e.g. test collect tasks) can
        # attach to the bus before the first publish happens.
        await asyncio.sleep(0)

    async def _run_pipeline(self, bus: FrameBus) -> None:
        try:
            await self._build_and_run_pipecat_pipeline(bus)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.error("pipeline_backend.error", session_id=self._session_id, error=str(exc))
            if self._publisher:
                await self._publisher.on_end(reason="error")

    async def _build_and_run_pipecat_pipeline(self, bus: FrameBus) -> None:
        """Energy VAD → faster-whisper STT → TTSService audio loop.

        Reads PCM16/16kHz chunks from self._audio_queue, detects speech
        boundaries using RMS energy, transcribes with faster-whisper, then
        synthesizes a response with the injected TTSService.
        """
        import numpy as np
        from faster_whisper import WhisperModel

        whisper = WhisperModel(self._stt_model, device="cpu", compute_type="int8")
        log.info("pipeline_backend.whisper_loaded", model=self._stt_model)

        audio_buffer: list[bytes] = []
        silence_count = 0
        is_speaking = False

        while True:
            try:
                chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            rms = _rms(chunk)

            if rms > _SILENCE_RMS:
                silence_count = 0
                is_speaking = True
                audio_buffer.append(chunk)
            elif is_speaking:
                silence_count += 1
                audio_buffer.append(chunk)

                if silence_count >= _TURN_SILENCE_CHUNKS and audio_buffer:
                    full_pcm = b"".join(audio_buffer)
                    audio_buffer = []
                    is_speaking = False
                    silence_count = 0

                    audio_f32 = (
                        np.frombuffer(full_pcm, dtype=np.int16)
                        .astype(np.float32) / 32768.0
                    )
                    segments, _ = whisper.transcribe(audio_f32, language="en")
                    text = " ".join(s.text for s in segments).strip()

                    if not text:
                        continue

                    now = time.time()
                    uid = uuid.uuid4().hex[:8]
                    await bus.publish(TranscriptDelta(
                        session_id=self._session_id,
                        utterance_id=uid,
                        speaker=Speaker.USER,
                        text=text,
                        is_final=True,
                        ts_start=now,
                        ts_end=now,
                    ))
                    log.info("pipeline_backend.transcribed", text=text)

                    try:
                        pcm_response = await self._tts.synthesize(text)
                        await bus.publish(AudioChunk(
                            session_id=self._session_id,
                            speaker=Speaker.COACH,
                            pcm16_16k=pcm_response,
                            ts=time.time(),
                        ))
                    except Exception as exc:
                        log.warning("pipeline_backend.tts_error", error=str(exc))

    async def send_caller_audio(self, pcm16_16k: bytes) -> None:
        await self._audio_queue.put(pcm16_16k)

    async def inject_speech(self, text: str) -> None:
        """Speak a deterministic line: synthesize and publish AudioChunk immediately."""
        if self._publisher is None:
            return
        try:
            pcm = await self._tts.synthesize(text)
            await self._publisher.on_bot_audio(pcm)
        except Exception as exc:
            log.warning("pipeline_backend.inject_speech_error", text=text, error=str(exc))

    async def swap_persona(self, persona: PersonaSpec) -> None:
        await self._tts.set_voice(persona.get("voice_ref"))
        log.info(
            "pipeline_backend.persona_swapped",
            session_id=self._session_id,
            persona_name=persona["name"],
        )

    async def say(self, request: SpeakRequest) -> None:
        await self.inject_speech(request.text)

    async def close(self) -> None:
        if self._pipeline_task and not self._pipeline_task.done():
            self._pipeline_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pipeline_task
        log.info("pipeline_backend.closed", session_id=self._session_id)
