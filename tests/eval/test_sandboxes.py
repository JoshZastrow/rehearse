"""Sandbox lifecycle tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from rehearse.eval.protocols import BenchmarkExample
from rehearse.eval.sandbox_connection import SandboxConnection
from rehearse.eval.sandboxes import (
    CustomerAgentSandbox,
    SandboxHandle,
    SandboxLifecycleError,
    VoiceAgentSandbox,
)
from rehearse.eval.transports import RuntimeDuplexEndpoint, TransportClosedError


class RecordingCustomerAgent:
    name = "recording-customer"
    version = "test"

    def __init__(self) -> None:
        self.start_calls: list[tuple[SandboxHandle, BenchmarkExample]] = []
        self.close_calls = 0
        self.transport: RuntimeDuplexEndpoint | None = None

    async def start(
        self,
        handle: SandboxHandle,
        example: BenchmarkExample,
        transport: RuntimeDuplexEndpoint | None = None,
    ) -> None:
        self.start_calls.append((handle, example))
        self.transport = transport

    async def close(self) -> None:
        self.close_calls += 1


class RecordingVoiceRuntime:
    name = "recording-voice-agent"
    version = "test"

    def __init__(self) -> None:
        self.start_calls: list[tuple[SandboxHandle, BenchmarkExample]] = []
        self.close_calls = 0
        self.transport: RuntimeDuplexEndpoint | None = None

    async def start(
        self,
        handle: SandboxHandle,
        example: BenchmarkExample,
        transport: RuntimeDuplexEndpoint | None = None,
    ) -> None:
        self.start_calls.append((handle, example))
        self.transport = transport

    async def close(self) -> None:
        self.close_calls += 1


def _example() -> BenchmarkExample:
    return BenchmarkExample(
        id="ex1",
        benchmark="sandbox",
        payload={"customer_script": ["hello", "I am unsure"]},
        expected={},
    )


async def test_voice_agent_sandbox_can_start_and_close(tmp_path: Path):
    runtime = RecordingVoiceRuntime()
    sandbox = VoiceAgentSandbox(runtime=runtime, model_slots={"coach": "test-model"})

    handle = await sandbox.start(example=_example(), run_dir=tmp_path / "voice", rng_seed=7)

    assert sandbox.status == "running"
    assert handle.kind == "voice-agent"
    assert handle.example_id == "ex1"
    assert handle.model_slots == {"coach": "test-model"}
    assert handle.run_dir.exists()
    assert runtime.start_calls[0][0] == handle

    await sandbox.close()
    assert sandbox.status == "closed"
    assert runtime.close_calls == 1


async def test_customer_agent_sandbox_accepts_injected_customer_agent(tmp_path: Path):
    customer = RecordingCustomerAgent()
    sandbox = CustomerAgentSandbox(customer_agent=customer)

    handle = await sandbox.start(example=_example(), run_dir=tmp_path / "customer", rng_seed=11)

    assert sandbox.status == "running"
    assert handle.kind == "customer-agent"
    assert handle.name == "recording-customer"
    assert customer.start_calls[0][1].payload["customer_script"] == ["hello", "I am unsure"]

    await sandbox.close()
    await sandbox.close()
    assert sandbox.status == "closed"
    assert customer.close_calls == 1


async def test_sandbox_lifecycle_context_manager_closes_on_exception(tmp_path: Path):
    runtime = RecordingVoiceRuntime()
    sandbox = VoiceAgentSandbox(runtime=runtime)

    with pytest.raises(RuntimeError, match="boom"):
        async with sandbox.lifecycle(example=_example(), run_dir=tmp_path / "voice", rng_seed=0):
            assert sandbox.status == "running"
            raise RuntimeError("boom")

    assert sandbox.status == "closed"
    assert runtime.close_calls == 1


async def test_running_sandbox_cannot_be_started_twice(tmp_path: Path):
    sandbox = CustomerAgentSandbox()
    await sandbox.start(example=_example(), run_dir=tmp_path / "customer", rng_seed=0)

    with pytest.raises(SandboxLifecycleError, match="already running"):
        await sandbox.start(example=_example(), run_dir=tmp_path / "customer2", rng_seed=1)

    await sandbox.close()


async def test_closed_sandbox_cannot_be_restarted(tmp_path: Path):
    sandbox = VoiceAgentSandbox()
    await sandbox.start(example=_example(), run_dir=tmp_path / "voice", rng_seed=0)
    await sandbox.close()

    with pytest.raises(SandboxLifecycleError, match="cannot be restarted"):
        await sandbox.start(example=_example(), run_dir=tmp_path / "voice2", rng_seed=1)


async def test_sandbox_connection_wires_customer_and_runtime_transport(tmp_path: Path):
    customer_agent = RecordingCustomerAgent()
    voice_runtime = RecordingVoiceRuntime()
    connection = SandboxConnection(
        customer=CustomerAgentSandbox(customer_agent=customer_agent),
        runtime=VoiceAgentSandbox(runtime=voice_runtime),
    )

    handles = await connection.start(example=_example(), run_dir=tmp_path / "rollout", rng_seed=3)

    assert handles.customer.kind == "customer-agent"
    assert handles.runtime.kind == "voice-agent"
    assert customer_agent.transport is not None
    assert voice_runtime.transport is not None

    sent = await customer_agent.transport.send("text", payload={"text": "hello"})
    received = await voice_runtime.transport.receive(timeout_s=0.1)
    assert received == sent
    assert received.source == "customer"
    assert received.payload == {"text": "hello"}

    reply = await voice_runtime.transport.send("text", payload={"text": "hi"})
    observed = await customer_agent.transport.receive(timeout_s=0.1)
    assert observed == reply
    assert observed.source == "runtime"

    await connection.close()
    assert customer_agent.close_calls == 1
    assert voice_runtime.close_calls == 1

    with pytest.raises(TransportClosedError):
        await customer_agent.transport.send("text", payload={"text": "after close"})


async def test_sandbox_connection_context_closes_everything_on_exception(tmp_path: Path):
    customer_agent = RecordingCustomerAgent()
    voice_runtime = RecordingVoiceRuntime()
    connection = SandboxConnection(
        customer=CustomerAgentSandbox(customer_agent=customer_agent),
        runtime=VoiceAgentSandbox(runtime=voice_runtime),
    )

    with pytest.raises(RuntimeError, match="boom"):
        async with connection.lifecycle(
            example=_example(),
            run_dir=tmp_path / "rollout",
            rng_seed=0,
        ):
            raise RuntimeError("boom")

    assert customer_agent.close_calls == 1
    assert voice_runtime.close_calls == 1
    assert connection.transport.status == "closed"
