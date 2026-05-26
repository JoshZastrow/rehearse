"""Tests for RuntimeSandboxEnvironment (Phase 4a).

Covers:
- Missing ANTHROPIC_API_KEY → RuntimeError at __init__
- Full rollout with stub CustomerDriver → 5 artifact files in artifacts_dir
- voice-agent-sandbox shim → DeprecationWarning to stderr
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from rehearse.eval.customers import CallerResult
from rehearse.eval.environments import get_environment
from rehearse.eval.environments.runtime_sandbox import RuntimeSandboxEnvironment
from rehearse.eval.protocols import BenchmarkExample
from rehearse.phases.phases import PhaseBudgets
from rehearse.session.runtime import RuntimeHost, TextOnlyCoachAdapter
from rehearse.storage import LocalFilesystemStore
from rehearse.backends.transport import InMemoryTwoWayChannel, TwoWayChannel
from rehearse.types import Phase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCENARIO = {
    "situation": "salary negotiation",
    "goal": "get a 20% raise",
    "stakes": "career finances",
    "emotional_state": "anxious",
}

_EXAMPLE = BenchmarkExample(
    id="test-example-001",
    benchmark="voice-rollout-judges",
    payload={"scenario": _SCENARIO},
)

_ZERO_DWELL = PhaseBudgets(
    intake_seconds=120,
    practice_seconds=120,
    feedback_seconds=120,
    intake_min_dwell_seconds=0,
    practice_min_dwell_seconds=0,
)


class _ScriptedCustomerDriver:
    """Sends a few intake turns, then sends customer_done."""

    name = "stub-customer"
    version = "v0"

    def __init__(self, turns: list[str] | None = None) -> None:
        self._turns = turns or [
            "I need help negotiating my salary.",
            "I want at least a 20% raise.",
            "I'm ready to practice.",
        ]

    async def run(
        self,
        *,
        transport: TwoWayChannel,
        runtime_phase: Any,
    ) -> CallerResult:
        for turn in self._turns:
            await transport.send("text", payload={"text": turn})
            await asyncio.sleep(0.05)  # let host process turn

        # Drain text events; stop when we get a control event.
        for _ in range(30):
            try:
                evt = await asyncio.wait_for(transport.receive(), timeout=1.0)
                if evt.kind == "control":
                    break
            except asyncio.TimeoutError:
                break

        await transport.send("control", payload={"event": "customer_done"})
        return CallerResult(
            turns_sent=len(self._turns),
            turns_per_phase={"intake": len(self._turns)},
        )


class _ScriptedCoach:
    """Fixed responses, with short sleeps so I/O can complete."""

    def __init__(self) -> None:
        self._idx = 0
        self._replies = [
            "Tell me more about that.",
            "What's your goal?",
            "Great. Let's start practicing.",
            "Good try. Keep going.",
            "Here's what I noticed.",
        ]

    async def respond(self, user_text: str, session_id: str) -> str:
        await asyncio.sleep(0.05)
        reply = self._replies[self._idx % len(self._replies)]
        self._idx += 1
        return reply


# ---------------------------------------------------------------------------
# Missing ANTHROPIC_API_KEY → raises at __init__
# ---------------------------------------------------------------------------


def test_missing_api_key_raises_at_init() -> None:
    """RuntimeSandboxEnvironment must raise RuntimeError when API key absent."""
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            RuntimeSandboxEnvironment()


# ---------------------------------------------------------------------------
# Full rollout with stub customer → all 5 artifact files present
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_rollout_produces_artifacts(tmp_path: Path) -> None:
    """Rollout with a stub customer and scripted coach writes all five artifacts."""
    session_id = "eval-test-example-001"
    session_dir = tmp_path / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    store = LocalFilesystemStore(tmp_path, public_base_url="http://localhost")
    coach = _ScriptedCoach()
    host = RuntimeHost(store, coach, budgets=_ZERO_DWELL, phase_timeout_s=15.0)
    customer = _ScriptedCustomerDriver()
    transport = InMemoryTwoWayChannel()

    await asyncio.gather(
        host.run(session_id=session_id, transport=transport.runtime),
        customer.run(
            transport=transport.customer,
            runtime_phase=lambda: host.current_phase,
        ),
    )

    # session.json + transcript.jsonl + phase_timing.json are always written
    assert (session_dir / "session.json").exists(), "session.json missing"
    assert (session_dir / "transcript.jsonl").exists(), "transcript.jsonl missing"
    assert (session_dir / "phase_timing.json").exists(), "phase_timing.json missing"

    # intake.json and persona.json are written when INTAKE completes
    assert (session_dir / "intake.json").exists(), "intake.json missing"
    assert (session_dir / "persona.json").exists(), "persona.json missing"


# ---------------------------------------------------------------------------
# Stub customer doesn't deadlock (no hang under turn cap)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stub_customer_no_deadlock(tmp_path: Path) -> None:
    """A stub customer that sends customer_done immediately exits without hanging."""
    session_id = "eval-quick"
    session_dir = tmp_path / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    store = LocalFilesystemStore(tmp_path, public_base_url="http://localhost")
    coach = _ScriptedCoach()
    host = RuntimeHost(store, coach, budgets=_ZERO_DWELL, phase_timeout_s=5.0)
    transport = InMemoryTwoWayChannel()

    async def _quick_customer() -> CallerResult:
        await transport.customer.send("control", payload={"event": "customer_done"})
        return CallerResult(turns_sent=0, turns_per_phase={})

    results = await asyncio.wait_for(
        asyncio.gather(
            host.run(session_id=session_id, transport=transport.runtime),
            _quick_customer(),
        ),
        timeout=10.0,
    )
    assert results is not None


# ---------------------------------------------------------------------------
# voice-agent-sandbox shim → deprecation warning to stderr
# ---------------------------------------------------------------------------


def test_voice_agent_sandbox_deprecation_warning(capsys: Any) -> None:
    """get_environment('voice-agent-sandbox') prints a DeprecationWarning to stderr."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-stub"}):
        get_environment("voice-agent-sandbox", model_slots={})
    captured = capsys.readouterr()
    assert "DeprecationWarning" in captured.err
    assert "voice-agent-sandbox" in captured.err
    assert "runtime-sandbox" in captured.err
