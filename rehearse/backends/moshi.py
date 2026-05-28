"""MoshiBackend — ConversationBackend backed by Moshi full-duplex speech model.

Audio pipeline:
  send_caller_audio(PCM16 16kHz) → resample to 24kHz → mimi.encode()
    → lm_gen.step() → text tokens + audio tokens
    → mimi.decode() → resample to 16kHz → AudioChunk(COACH)

User transcription: faster-whisper runs on buffered user audio after each
coach turn, emitting TranscriptDelta(USER, is_final=True) + ProsodyEvent(USER).

The inference loop runs in a ThreadPoolExecutor (sync streaming context
managers + GPU calls). Frames are published back to the asyncio event loop
via asyncio.run_coroutine_threadsafe.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import queue
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
import torch
import torchaudio.functional as F

from rehearse.backends.base import PersonaSpec
from rehearse.backends.moshi_asr import MoshiASR
from rehearse.backends.moshi_loader import load_models
from rehearse.bus import FrameBus
from rehearse.frames import AudioChunk, EndOfCall, ProsodyEvent, TranscriptDelta
from rehearse.types import ProsodyScores, Speaker

if TYPE_CHECKING:
    pass

log = structlog.get_logger(__name__)

_STOP_SENTINEL = object()
_MOSHI_SAMPLE_RATE = 24_000
_REHEARSE_SAMPLE_RATE = 16_000
_SILENCE_FRAMES_TO_FLUSH = 10  # 10 × 80ms = 800ms of silence triggers turn flush


class MoshiBackend:
    """ConversationBackend backed by Moshi full-duplex speech model."""

    def __init__(
        self,
        *,
        checkpoint_path: str,
        hf_repo: str,
        device: str,
        asr_model: str = "base",
    ) -> None:
        self._mimi, self._lm_gen, self._tokenizer = load_models(
            checkpoint_path=checkpoint_path,
            hf_repo=hf_repo,
            device=device,
        )
        self._asr = MoshiASR(model_size=asr_model)
        self._device = device
        self._frame_size: int = self._mimi.frame_size  # 1920 at 24 kHz

        self._audio_q: queue.Queue[bytes | object] = queue.Queue()
        self._stop = threading.Event()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="moshi-infer"
        )
        self._task: asyncio.Task | None = None
        self._session_id = ""
        self._bus: FrameBus | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def __aenter__(self) -> MoshiBackend:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def start(self, session_id: str, bus: FrameBus) -> None:
        self._session_id = session_id
        self._bus = bus
        loop = asyncio.get_running_loop()
        self._loop = loop
        future = loop.run_in_executor(self._executor, self._sync_inference_loop)
        self._task = asyncio.ensure_future(future)
        log.info("moshi_backend.started", session_id=session_id)

    async def send_caller_audio(self, pcm16_16k: bytes) -> None:
        self._audio_q.put_nowait(pcm16_16k)

    async def inject_speech(self, text: str) -> None:
        log.warning("moshi_backend.inject_speech_not_supported", text=text[:50])

    async def swap_persona(self, persona: PersonaSpec) -> None:
        log.warning("moshi_backend.swap_persona_not_supported", name=persona["name"])

    async def close(self) -> None:
        self._stop.set()
        self._audio_q.put_nowait(_STOP_SENTINEL)
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=30.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self._task.cancel()
        if self._task is not None and self._task.done() and not self._task.cancelled():
            exc = self._task.exception()
            if exc:
                log.error("moshi_backend.inference_loop_error", exc_info=exc, session_id=self._session_id)
        self._executor.shutdown(wait=False)
        log.info("moshi_backend.closed", session_id=self._session_id)

    # ------------------------------------------------------------------
    # Sync inference loop — runs in ThreadPoolExecutor
    # ------------------------------------------------------------------

    def _sync_inference_loop(self) -> None:
        pcm_buf = np.array([], dtype=np.float32)
        coach_utt_id = str(uuid.uuid4())
        coach_text_pieces: list[str] = []
        silence_streak = 0

        with self._mimi.streaming(1), self._lm_gen.streaming(1), torch.no_grad():
            while not self._stop.is_set():
                try:
                    raw = self._audio_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if raw is _STOP_SENTINEL:
                    break

                assert isinstance(raw, bytes)
                self._asr.push_audio(raw)

                # PCM16 16kHz → float32 24kHz
                pcm16 = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                pcm_float = pcm16 / 32768.0
                pcm_24k = self._resample(pcm_float, _REHEARSE_SAMPLE_RATE, _MOSHI_SAMPLE_RATE)
                pcm_buf = np.concatenate([pcm_buf, pcm_24k])

                while len(pcm_buf) >= self._frame_size:
                    frame = pcm_buf[: self._frame_size]
                    pcm_buf = pcm_buf[self._frame_size :]

                    for tokens_out in self._infer_frame(frame):
                        text_id = int(tokens_out[0, 0, 0].item())
                        pad_id = self._lm_gen.lm_model.text_padding_token_id
                        if text_id != pad_id:
                            piece = self._tokenizer.id_to_piece(text_id)
                            coach_text_pieces.append(piece)
                            self._publish(TranscriptDelta(
                                session_id=self._session_id,
                                utterance_id=coach_utt_id,
                                speaker=Speaker.COACH,
                                text=piece,
                                is_final=False,
                                ts_start=time.time(),
                            ))
                            silence_streak = 0
                        else:
                            silence_streak += 1

                        audio_codes = tokens_out[:, 1:, :]
                        wav = self._mimi.decode(audio_codes)
                        pcm16_out = self._wav_to_pcm16_16k(wav)
                        self._publish(AudioChunk(
                            session_id=self._session_id,
                            speaker=Speaker.COACH,
                            pcm16_16k=pcm16_out,
                            ts=time.time(),
                        ))

                        if silence_streak >= _SILENCE_FRAMES_TO_FLUSH and coach_text_pieces:
                            self._flush_coach_turn(coach_utt_id, coach_text_pieces)
                            coach_text_pieces = []
                            coach_utt_id = str(uuid.uuid4())
                            silence_streak = 0
                            self._flush_user_turn()

        if coach_text_pieces:
            self._flush_coach_turn(coach_utt_id, coach_text_pieces)
        self._flush_user_turn()
        self._publish(EndOfCall(
            session_id=self._session_id,
            reason="hangup",
            ts=time.time(),
        ))

    def _infer_frame(self, frame: np.ndarray) -> list[torch.Tensor]:
        """Encode one audio frame and step the LM for each code timestep.

        mimi.encode may return multiple time steps; lm_gen.step takes one at a
        time and may return None during warmup — those are filtered out.
        """
        frame_t = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0)
        if self._device != "cpu":
            frame_t = frame_t.to(self._device)
        codes = self._mimi.encode(frame_t)  # [1, K, T]
        results: list[torch.Tensor] = []
        for c in range(codes.shape[-1]):
            tokens = self._lm_gen.step(codes[:, :, c : c + 1])
            if tokens is not None:
                results.append(tokens)
        return results

    def _flush_coach_turn(self, utt_id: str, pieces: list[str]) -> None:
        text = "".join(pieces).replace("▁", " ").strip()
        self._publish(TranscriptDelta(
            session_id=self._session_id,
            utterance_id=utt_id,
            speaker=Speaker.COACH,
            text=text,
            is_final=True,
            ts_start=time.time(),
            ts_end=time.time(),
        ))

    def _flush_user_turn(self) -> None:
        text = self._asr.transcribe_and_reset()
        if not text:
            return
        utt_id = str(uuid.uuid4())
        now = time.time()
        self._publish(TranscriptDelta(
            session_id=self._session_id,
            utterance_id=utt_id,
            speaker=Speaker.USER,
            text=text,
            is_final=True,
            ts_start=now,
            ts_end=now,
        ))
        self._publish(ProsodyEvent(
            session_id=self._session_id,
            utterance_id=utt_id,
            speaker=Speaker.USER,
            scores=ProsodyScores(arousal=0.0, valence=0.0),
            ts_start=now,
            ts_end=now,
        ))

    def _publish(self, frame: Any) -> None:
        if self._bus is None or self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._bus.publish(frame), self._loop)

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        t = torch.from_numpy(audio).float().unsqueeze(0)
        resampled = F.resample(t, orig_sr, target_sr)
        return resampled.squeeze(0).numpy()

    @staticmethod
    def _wav_to_pcm16_16k(wav: torch.Tensor) -> bytes:
        audio = wav.squeeze(0).squeeze(0)
        audio_16k = F.resample(audio.unsqueeze(0), _MOSHI_SAMPLE_RATE, _REHEARSE_SAMPLE_RATE)
        audio_16k = audio_16k.squeeze(0).clamp(-1.0, 1.0)
        pcm16 = (audio_16k * 32767).to(torch.int16)
        return pcm16.numpy().tobytes()
