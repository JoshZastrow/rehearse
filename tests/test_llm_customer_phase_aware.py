"""Verify SyntheticCaller: phase switching, turn cap, end-of-call, init protocol."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from rehearse.eval.drivers import CallerResult
from rehearse.eval.drivers.llm_customer import SyntheticCaller, _MAX_TURNS
from rehearse.backends.transport import InMemoryTwoWayChannel
from rehearse.types import Phase


# ---------------------------------------------------------------------------
# Minimal Anthropic client stub
# ---------------------------------------------------------------------------


@dataclass
class _TextBlock:
    text: str
    type: str = "text"


@dataclass
class _Response:
    content: list[_TextBlock]
    stop_reason: str = "end_turn"


class _StubClient:
    """Return scripted lines in order; fall back to a default reply."""

    def __init__(self, lines: list[str] | None = None) -> None:
        self._lines = list(lines or [])
        self._idx = 0
        self.messages = self

    async def create(self, **_: Any) -> _Response:
        if self._idx < len(self._lines):
            text = self._lines[self._idx]
        else:
            text = "I hear you."
        self._idx += 1
        return _Response(content=[_TextBlock(text=text)])


def _make_driver(
    lines: list[str] | None = None,
    run_dir: Path | None = None,
) -> SyntheticCaller:
    return SyntheticCaller(
        scenario={
            "situation": "salary negotiation",
            "goal": "get a 20% raise",
            "stakes": "career and finances",
            "emotional_state": "anxious",
        },
        client=_StubClient(lines),
        run_dir=run_dir,
    )


# ---------------------------------------------------------------------------
# Customer sends first frame before any runtime message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_customer_sends_first_without_waiting() -> None:
    """Driver sends the first turn without awaiting any runtime greeting."""
    driver = _make_driver(["Hello, I need help."])
    transport = InMemoryTwoWayChannel()

    sent_before_runtime: bool = False

    async def _runtime() -> None:
        nonlocal sent_before_runtime
        # Runtime has not sent anything yet — but the customer should have.
        evt = await asyncio.wait_for(transport.runtime.receive(), timeout=3.0)
        sent_before_runtime = evt.kind == "text"
        # Ack with customer_done so the driver exits.
        await transport.runtime.send("control", payload={"event": "end_of_call"})

    await asyncio.gather(
        driver.run(transport=transport.customer, runtime_phase=lambda: Phase.INTAKE),
        _runtime(),
    )

    assert sent_before_runtime


# ---------------------------------------------------------------------------
# Phase transition → driver switches prompt and sends new phase's first turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_phase_transition_triggers_prompt_switch() -> None:
    """On phase_transition control event, driver sends first turn of new phase."""
    driver = _make_driver(["Intake turn.", "Practice turn.", "Feedback turn."])
    transport = InMemoryTwoWayChannel()

    turns_received: list[str] = []

    async def _runtime() -> None:
        rt = transport.runtime
        # Receive intake first turn
        evt = await asyncio.wait_for(rt.receive(), timeout=3.0)
        assert evt.kind == "text"
        turns_received.append("intake-first")

        # Emit phase_transition → PRACTICE
        await rt.send(
            "control",
            payload={"event": "phase_transition", "phase": "PRACTICE"},
        )

        # Receive practice first turn
        evt = await asyncio.wait_for(rt.receive(), timeout=3.0)
        assert evt.kind == "text"
        turns_received.append("practice-first")

        # End the session
        await rt.send("control", payload={"event": "end_of_call"})

    await asyncio.gather(
        driver.run(transport=transport.customer, runtime_phase=lambda: Phase.INTAKE),
        _runtime(),
    )

    assert turns_received == ["intake-first", "practice-first"]


# ---------------------------------------------------------------------------
# Hard turn cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hard_turn_cap_stops_driver() -> None:
    """Driver stops after _MAX_TURNS and sends customer_done."""
    driver = _make_driver()  # default stub returns "I hear you." indefinitely
    transport = InMemoryTwoWayChannel()

    events_from_customer: list[Any] = []

    async def _runtime() -> None:
        rt = transport.runtime
        while True:
            evt = await asyncio.wait_for(rt.receive(), timeout=5.0)
            events_from_customer.append(evt)
            if evt.kind == "control":
                break
            # Echo a text reply so driver can keep going
            await rt.send("text", payload={"text": "Good, tell me more."})

    await asyncio.gather(
        driver.run(transport=transport.customer, runtime_phase=lambda: Phase.INTAKE),
        _runtime(),
    )

    text_events = [e for e in events_from_customer if e.kind == "text"]
    control_events = [e for e in events_from_customer if e.kind == "control"]

    assert len(text_events) <= _MAX_TURNS
    assert len(control_events) >= 1
    assert control_events[-1].payload.get("event") == "customer_done"


# ---------------------------------------------------------------------------
# end_of_call control event → driver exits cleanly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_of_call_exits_cleanly() -> None:
    """driver.run() returns CallerResult on end_of_call."""
    driver = _make_driver(["Hello, I need help."])
    transport = InMemoryTwoWayChannel()

    async def _runtime() -> None:
        await asyncio.wait_for(transport.runtime.receive(), timeout=3.0)
        await transport.runtime.send("control", payload={"event": "end_of_call"})

    result, _ = await asyncio.gather(
        driver.run(transport=transport.customer, runtime_phase=lambda: Phase.INTAKE),
        _runtime(),
    )

    assert isinstance(result, CallerResult)
    assert result.turns_sent >= 1
    assert result.error is None


# ---------------------------------------------------------------------------
# customer_driver.json written when run_dir is supplied
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_customer_driver_json_written(tmp_path: Path) -> None:
    """Result JSON is persisted to run_dir when supplied."""
    driver = _make_driver(["Hello!"], run_dir=tmp_path)
    transport = InMemoryTwoWayChannel()

    async def _runtime() -> None:
        await asyncio.wait_for(transport.runtime.receive(), timeout=3.0)
        await transport.runtime.send("control", payload={"event": "end_of_call"})

    await asyncio.gather(
        driver.run(transport=transport.customer, runtime_phase=lambda: Phase.INTAKE),
        _runtime(),
    )

    artifact = tmp_path / "customer_driver.json"
    assert artifact.exists()
    import json
    data = json.loads(artifact.read_text())
    assert data["driver"] == "synthetic-caller"
    assert data["turns_sent"] >= 1
