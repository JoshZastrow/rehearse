"""Lifecycle-managed eval sandboxes.

The eval client eventually coordinates two isolated actors per example:

- a voice-agent runtime sandbox, which owns the system under test
- a customer-agent sandbox, which owns simulated customer behavior

This module defines the lifecycle boundary before the full conversation loop
exists. The default implementations are intentionally inert, but they make the
start/close contract testable and give future subprocess/container sandboxes a
small interface to preserve.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable
from uuid import uuid4

from rehearse.eval.protocols import BenchmarkExample
from rehearse.eval.transports import RuntimeDuplexEndpoint

SandboxKind = Literal["voice-agent", "customer-agent"]
SandboxStatus = Literal["not_started", "running", "closed"]


class SandboxLifecycleError(RuntimeError):
    """Raised when a sandbox lifecycle method is called out of order."""


@dataclass(frozen=True)
class SandboxHandle:
    """Runtime metadata for one started sandbox."""

    id: str
    kind: SandboxKind
    name: str
    version: str
    example_id: str
    run_dir: Path
    rng_seed: int
    model_slots: dict[str, str] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@runtime_checkable
class VoiceAgentRuntime(Protocol):
    """Injected implementation used by VoiceAgentSandbox."""

    name: str
    version: str

    async def start(
        self,
        handle: SandboxHandle,
        example: BenchmarkExample,
        transport: RuntimeDuplexEndpoint | None = None,
    ) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class CustomerAgentRuntime(Protocol):
    """Injected implementation used by CustomerAgentSandbox."""

    name: str
    version: str

    async def start(
        self,
        handle: SandboxHandle,
        example: BenchmarkExample,
        transport: RuntimeDuplexEndpoint | None = None,
    ) -> None: ...

    async def close(self) -> None: ...


class NoopVoiceAgentRuntime:
    """Placeholder runtime until the real voice-agent boundary is available."""

    name = "noop-voice-agent"
    version = "v0"

    def __init__(self) -> None:
        self.started = False
        self.closed = False

    async def start(
        self,
        handle: SandboxHandle,
        example: BenchmarkExample,
        transport: RuntimeDuplexEndpoint | None = None,
    ) -> None:
        self.started = True
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class ScriptedCustomerAgentRuntime:
    """Simple customer runtime driven by the benchmark payload.

    If the example contains ``payload["customer_script"]``, future rollout code
    can consume those utterances from this object. For now, lifecycle is the only
    behavior under test.
    """

    name = "scripted-customer"
    version = "v0"

    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.script: list[str] = []

    async def start(
        self,
        handle: SandboxHandle,
        example: BenchmarkExample,
        transport: RuntimeDuplexEndpoint | None = None,
    ) -> None:
        raw_script = example.payload.get("customer_script", [])
        self.script = [str(turn) for turn in raw_script] if isinstance(raw_script, list) else []
        self.started = True
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _ManagedSandbox:
    kind: SandboxKind

    def __init__(
        self,
        *,
        name: str,
        version: str,
        model_slots: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.version = version
        self.model_slots = dict(model_slots or {})
        self.status: SandboxStatus = "not_started"
        self.handle: SandboxHandle | None = None

    def _new_handle(
        self,
        *,
        example: BenchmarkExample,
        run_dir: Path,
        rng_seed: int,
    ) -> SandboxHandle:
        return SandboxHandle(
            id=f"{self.kind}-{uuid4().hex[:8]}",
            kind=self.kind,
            name=self.name,
            version=self.version,
            example_id=example.id,
            run_dir=run_dir,
            rng_seed=rng_seed,
            model_slots=self.model_slots,
        )

    def _ensure_can_start(self) -> None:
        if self.status == "running":
            raise SandboxLifecycleError(f"{self.kind} sandbox is already running")
        if self.status == "closed":
            raise SandboxLifecycleError(f"{self.kind} sandbox cannot be restarted after close")

    async def close(self) -> None:
        raise NotImplementedError


class VoiceAgentSandbox(_ManagedSandbox):
    """Lifecycle wrapper around the voice-agent runtime under evaluation."""

    kind: SandboxKind = "voice-agent"

    def __init__(
        self,
        *,
        runtime: VoiceAgentRuntime | None = None,
        model_slots: dict[str, str] | None = None,
    ) -> None:
        self.runtime = runtime or NoopVoiceAgentRuntime()
        super().__init__(
            name=self.runtime.name,
            version=self.runtime.version,
            model_slots=model_slots,
        )

    async def start(
        self,
        *,
        example: BenchmarkExample,
        run_dir: Path,
        rng_seed: int,
        transport: RuntimeDuplexEndpoint | None = None,
    ) -> SandboxHandle:
        self._ensure_can_start()
        run_dir.mkdir(parents=True, exist_ok=True)
        handle = self._new_handle(example=example, run_dir=run_dir, rng_seed=rng_seed)
        await self.runtime.start(handle, example, transport)
        self.handle = handle
        self.status = "running"
        return handle

    async def close(self) -> None:
        if self.status == "closed":
            return
        if self.status == "running":
            await self.runtime.close()
        self.status = "closed"

    @asynccontextmanager
    async def lifecycle(
        self,
        *,
        example: BenchmarkExample,
        run_dir: Path,
        rng_seed: int,
        transport: RuntimeDuplexEndpoint | None = None,
    ) -> AsyncIterator[SandboxHandle]:
        handle = await self.start(
            example=example,
            run_dir=run_dir,
            rng_seed=rng_seed,
            transport=transport,
        )
        try:
            yield handle
        finally:
            await self.close()


class CustomerAgentSandbox(_ManagedSandbox):
    """Lifecycle wrapper around a simulated customer agent."""

    kind: SandboxKind = "customer-agent"

    def __init__(
        self,
        *,
        customer_agent: CustomerAgentRuntime | None = None,
        model_slots: dict[str, str] | None = None,
    ) -> None:
        self.customer_agent = customer_agent or ScriptedCustomerAgentRuntime()
        super().__init__(
            name=self.customer_agent.name,
            version=self.customer_agent.version,
            model_slots=model_slots,
        )

    async def start(
        self,
        *,
        example: BenchmarkExample,
        run_dir: Path,
        rng_seed: int,
        transport: RuntimeDuplexEndpoint | None = None,
    ) -> SandboxHandle:
        self._ensure_can_start()
        run_dir.mkdir(parents=True, exist_ok=True)
        handle = self._new_handle(example=example, run_dir=run_dir, rng_seed=rng_seed)
        await self.customer_agent.start(handle, example, transport)
        self.handle = handle
        self.status = "running"
        return handle

    async def close(self) -> None:
        if self.status == "closed":
            return
        if self.status == "running":
            await self.customer_agent.close()
        self.status = "closed"

    @asynccontextmanager
    async def lifecycle(
        self,
        *,
        example: BenchmarkExample,
        run_dir: Path,
        rng_seed: int,
        transport: RuntimeDuplexEndpoint | None = None,
    ) -> AsyncIterator[SandboxHandle]:
        handle = await self.start(
            example=example,
            run_dir=run_dir,
            rng_seed=rng_seed,
            transport=transport,
        )
        try:
            yield handle
        finally:
            await self.close()
