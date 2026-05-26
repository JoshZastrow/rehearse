"""TTS-backed synthetic caller for live-audio sandbox rollouts."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rehearse.eval.customers import CallerResult
from rehearse.eval.customers.llm_customer import _PHASE_PROMPTS
from rehearse.eval.tts_bridge import TTSProvider
from rehearse.backends.transport import TwoWayChannel
from rehearse.types import Phase

_MAX_TURNS = 12


class AudioCustomerDriver:
    """Generate caller text, synthesize it, and stream PCM16 audio to runtime."""

    name = "audio-customer"
    version = "v1"

    def __init__(
        self,
        scenario: dict[str, Any],
        tts: TTSProvider,
        *,
        llm_client: Any = None,
        run_dir: Path | None = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 256,
        temperature: float = 0.7,
        max_turns: int = _MAX_TURNS,
    ) -> None:
        self._scenario = scenario
        self._tts = tts
        self._llm_client = llm_client
        self._run_dir = run_dir
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_turns = max_turns
        self._usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    def _system_prompt(self, phase: Phase) -> str:
        template = _PHASE_PROMPTS.get(phase, _PHASE_PROMPTS[Phase.INTAKE])
        return template.format(
            situation=self._scenario.get("situation", ""),
            goal=self._scenario.get("goal", ""),
            stakes=self._scenario.get("stakes", ""),
            emotional_state=self._scenario.get("emotional_state", ""),
        )

    def _description(self, phase: Phase) -> str:
        base = self._scenario.get("emotional_state") or "natural conversational tone"
        if phase == Phase.PRACTICE:
            return str(base)
        if phase == Phase.FEEDBACK:
            return str(base)
        return str(base)

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
        *,
        system_prompt: str,
        history: list[tuple[str, str]],
    ) -> str:
        client = self._get_client()
        if hasattr(client, "complete"):
            return await client.complete(system_prompt=system_prompt, history=history)
        messages = [
            {"role": "user" if role == "coach" else "assistant", "content": text}
            for role, text in history
        ]
        if not messages or messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": "."})
        resp = await client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system_prompt,
            messages=messages,
        )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self._usage["prompt_tokens"] += getattr(usage, "input_tokens", 0)
            self._usage["completion_tokens"] += getattr(usage, "output_tokens", 0)
        return resp.content[0].text.strip() if resp.content else ""

    async def _send_turn(
        self,
        *,
        transport: TwoWayChannel,
        phase: Phase,
        text: str,
    ) -> None:
        pcm = await self._tts.synthesize_pcm(
            text=text,
            description=self._description(phase),
        )
        # Stream in 100ms chunks (1600 bytes at 16kHz PCM16) so EVI's VAD
        # sees a natural stream rather than one large blob.
        chunk_size = 3200  # 100ms at 16kHz × 2 bytes
        for i in range(0, len(pcm), chunk_size):
            await transport.send("audio", data=pcm[i : i + chunk_size])
            await asyncio.sleep(0.05)  # pace to ~2× real-time
        # Trailing silence: 500ms lets EVI's VAD detect end of speech.
        silence_500ms = b"\x00" * 16000
        await transport.send("audio", data=silence_500ms)
        await asyncio.sleep(0.5)
        await transport.send("control", payload={"event": "end_of_turn"})

    async def run(
        self,
        *,
        transport: TwoWayChannel,
        runtime_phase: Callable[[], Phase],
    ) -> CallerResult:
        phase = Phase.INTAKE
        history: list[tuple[str, str]] = []
        turns_per_phase: dict[str, int] = {}
        total_turns = 0
        error: str | None = None

        async def _send_next() -> bool:
            nonlocal total_turns
            if total_turns >= self._max_turns:
                return False
            text = await self._complete(
                system_prompt=self._system_prompt(phase),
                history=history,
            )
            if not text:
                return False
            history.append(("customer", text))
            await self._send_turn(transport=transport, phase=phase, text=text)
            total_turns += 1
            turns_per_phase[phase.value] = turns_per_phase.get(phase.value, 0) + 1
            return True

        try:
            await _send_next()
            while total_turns < self._max_turns:
                event = await transport.receive()
                if event.kind == "control":
                    evt = event.payload.get("event", "")
                    if evt == "phase_transition":
                        phase_name = str(event.payload.get("phase", ""))
                        try:
                            phase = Phase[phase_name]
                        except KeyError:
                            pass
                        await _send_next()
                    elif evt in ("customer_done", "end_of_call", "runtime_done"):
                        break
                    continue
                if event.kind == "text":
                    coach_text = str(event.payload.get("text", ""))
                    if coach_text:
                        history.append(("coach", coach_text))
                    await _send_next()
        except Exception as exc:
            error = str(exc)
        finally:
            await transport.send("control", payload={"event": "customer_done"})

        result = CallerResult(
            turns_sent=total_turns,
            turns_per_phase=turns_per_phase,
            error=error,
            token_usage=dict(self._usage),
        )
        if self._run_dir is not None:
            _write_result(self._run_dir, result)
        return result


def _write_result(run_dir: Path, result: CallerResult) -> None:
    out = {
        "driver": AudioCustomerDriver.name,
        "version": AudioCustomerDriver.version,
        "turns_sent": result.turns_sent,
        "turns_per_phase": result.turns_per_phase,
        "error": result.error,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "customer_driver.json").write_text(json.dumps(out, indent=2))
