from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from rehearse.eval.customers.audio_customer import AudioCustomerDriver
from rehearse.transport import InMemoryTwoWayChannel
from rehearse.types import Phase


class _StubTTS:
    name = "stub-tts"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    async def synthesize(
        self,
        *,
        text: str,
        out_path: Path,
        description: str | None = None,
    ) -> float:
        raise AssertionError("audio driver should use synthesize_pcm")

    async def synthesize_pcm(self, *, text: str, description: str | None = None) -> bytes:
        self.calls.append((text, description))
        return b"\x01\x00\x02\x00"


class _StubLLM:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)

    async def complete(self, *, system_prompt: str, history: list[tuple[str, str]]) -> str:
        return self._replies.pop(0) if self._replies else ""


@pytest.mark.asyncio
async def test_audio_customer_sends_audio_before_waiting_for_runtime(tmp_path: Path) -> None:
    transport = InMemoryTwoWayChannel()
    tts = _StubTTS()
    driver = AudioCustomerDriver(
        {
            "situation": "asking for a raise",
            "goal": "earn more",
            "stakes": "rent",
            "emotional_state": "anxious",
        },
        tts,
        llm_client=_StubLLM(["I need help asking for a raise."]),
        run_dir=tmp_path,
        max_turns=1,
    )

    run_task = asyncio.create_task(
        driver.run(transport=transport.customer, runtime_phase=lambda: Phase.INTAKE)
    )
    event = await transport.runtime.receive()

    assert event.kind == "audio"
    assert event.data == b"\x01\x00\x02\x00"
    eot = await transport.runtime.receive()
    assert eot.kind == "control"
    assert eot.payload["event"] == "end_of_turn"

    result = await run_task
    assert result.turns_sent == 1
    assert tts.calls[0][1] == "anxious"

    payload = json.loads((tmp_path / "customer_driver.json").read_text())
    assert payload["driver"] == "audio-customer"
    assert payload["turns_per_phase"] == {"intake": 1}


@pytest.mark.asyncio
async def test_audio_customer_uses_new_phase_description(tmp_path: Path) -> None:
    transport = InMemoryTwoWayChannel()
    tts = _StubTTS()
    driver = AudioCustomerDriver(
        {
            "situation": "asking for a raise",
            "goal": "earn more",
            "stakes": "rent",
            "emotional_state": "nervous but determined",
        },
        tts,
        llm_client=_StubLLM(["first turn", "practice turn"]),
        run_dir=tmp_path,
        max_turns=2,
    )

    run_task = asyncio.create_task(
        driver.run(transport=transport.customer, runtime_phase=lambda: Phase.INTAKE)
    )
    await transport.runtime.receive()
    await transport.runtime.receive()
    await transport.runtime.send(
        "control", payload={"event": "phase_transition", "phase": "PRACTICE"}
    )
    await transport.runtime.receive()
    await transport.runtime.receive()

    result = await run_task

    assert result.turns_per_phase == {"intake": 1, "practice": 1}
    assert tts.calls[1][1] == "nervous but determined"
