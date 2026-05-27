"""Scripted VoiceParticipant for offline eval rollouts.

Generates caller turns via LLM + TTS synthesis, streams PCM16 audio to the
backend, and listens to the FrameBus for provider responses and phase
transitions. Implements VoiceParticipant so the eval can call run_session()
with the same interface as production callers (Twilio, WebRTC).

Provider audio received via receive_audio() is discarded — the eval does not
play it back anywhere. The bus subscription is used only to know when to
generate the next caller turn.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any

import structlog

from rehearse.bus import FrameBus
from rehearse.eval.drivers.llm_customer import _PHASE_PROMPTS
from rehearse.eval.environments.tts_bridge import TTSProvider
from rehearse.frames import EndOfCall, PhaseSignal, TranscriptDelta
from rehearse.audio.participants import SpeakRequest, VoiceParticipant
from rehearse.types import ParticipantConfig, Phase, Speaker

log = structlog.get_logger(__name__)

_MAX_TURNS = 12
_CHUNK_SIZE = 3200       # 100 ms at 16 kHz × 2 bytes
_SILENCE_500MS = b"\x00" * 16000
_PACE_SLEEP = 0.05       # stream at ~2× realtime


class EvalCallerParticipant(VoiceParticipant):
    """LLM-driven, TTS-synthesized caller for offline eval rollouts.

    Reuses AudioCustomerDriver's LLM-completion and TTS logic but implements
    the VoiceParticipant interface so it can be passed directly to run_session().
    """

    def __init__(
        self,
        scenario: dict[str, Any],
        tts: TTSProvider,
        *,
        llm_client: Any = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 256,
        temperature: float = 0.7,
        max_turns: int = _MAX_TURNS,
    ) -> None:
        self._scenario = scenario
        self._tts = tts
        self._llm_client = llm_client
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_turns = max_turns
        self._usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    @property
    def config(self) -> ParticipantConfig:
        return ParticipantConfig(
            participant_id="eval-caller",
            role="caller",
            backend="eval_scripted",
        )

    async def receive_audio(self, pcm16_16k: bytes) -> None:
        pass  # provider audio is not replayed in eval

    async def say(self, request: SpeakRequest) -> None:
        pass  # eval caller does not receive injected speech

    def audio_stream(self, bus: FrameBus) -> AsyncIterator[bytes]:
        return self._generate_stream(bus)

    async def _generate_stream(self, bus: FrameBus) -> AsyncIterator[bytes]:
        chunk_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=500)

        async def _bus_watcher() -> None:
            phase = Phase.INTAKE
            history: list[tuple[str, str]] = []
            turns = 0

            # Send first caller turn immediately
            await _enqueue_turn(phase, history, chunk_queue, self)
            turns += 1

            async for frame in bus.subscribe():
                if isinstance(frame, PhaseSignal):
                    phase = frame.to_phase
                elif (
                    isinstance(frame, TranscriptDelta)
                    and frame.speaker != Speaker.USER
                    and frame.is_final
                    and frame.text.strip()
                ):
                    history.append(("provider", frame.text))
                    if turns >= self._max_turns:
                        await chunk_queue.put(None)
                        return
                    await _enqueue_turn(phase, history, chunk_queue, self)
                    turns += 1
                elif isinstance(frame, EndOfCall):
                    await chunk_queue.put(None)
                    return

            await chunk_queue.put(None)

        watcher = asyncio.create_task(_bus_watcher())
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(chunk_queue.get(), timeout=120)
                except asyncio.TimeoutError:
                    log.warning("eval_caller.response_timeout")
                    break
                if chunk is None:
                    break
                yield chunk
                await asyncio.sleep(_PACE_SLEEP)
        finally:
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher

    def _system_prompt(self, phase: Phase) -> str:
        template = _PHASE_PROMPTS.get(phase, _PHASE_PROMPTS[Phase.INTAKE])
        return template.format(
            situation=self._scenario.get("situation", ""),
            goal=self._scenario.get("goal", ""),
            stakes=self._scenario.get("stakes", ""),
            emotional_state=self._scenario.get("emotional_state", ""),
        )

    def _get_client(self) -> Any:
        if self._llm_client is None:
            from anthropic import AsyncAnthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            self._llm_client = AsyncAnthropic(api_key=api_key)
        return self._llm_client

    async def _complete(
        self,
        phase: Phase,
        history: list[tuple[str, str]],
    ) -> str:
        client = self._get_client()
        if hasattr(client, "complete"):
            return await client.complete(
                system_prompt=self._system_prompt(phase), history=history
            )
        messages = [
            {"role": "user" if role == "provider" else "assistant", "content": text}
            for role, text in history
        ]
        if not messages or messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": "."})
        resp = await client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=self._system_prompt(phase),
            messages=messages,
        )
        usage = getattr(resp, "usage", None)
        if usage:
            self._usage["prompt_tokens"] += getattr(usage, "input_tokens", 0)
            self._usage["completion_tokens"] += getattr(usage, "output_tokens", 0)
        return resp.content[0].text.strip() if resp.content else ""


async def _enqueue_turn(
    phase: Phase,
    history: list[tuple[str, str]],
    queue: asyncio.Queue[bytes | None],
    caller: EvalCallerParticipant,
) -> None:
    """Generate caller text, synthesize PCM, and put chunks in the queue."""
    text = await caller._complete(phase, history)  # noqa: SLF001
    if not text:
        return
    history.append(("caller", text))
    description = caller._scenario.get("emotional_state") or "natural conversational tone"  # noqa: SLF001
    pcm = await caller._tts.synthesize_pcm(text=text, description=description)  # noqa: SLF001
    for i in range(0, len(pcm), _CHUNK_SIZE):
        await queue.put(pcm[i : i + _CHUNK_SIZE])
    await queue.put(_SILENCE_500MS)
