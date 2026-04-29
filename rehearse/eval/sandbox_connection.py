"""Connect customer and voice-agent sandboxes through a non-Twilio transport."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from rehearse.eval.protocols import BenchmarkExample
from rehearse.eval.sandboxes import CustomerAgentSandbox, SandboxHandle, VoiceAgentSandbox
from rehearse.eval.transports import InMemoryDuplexTransport


@dataclass(frozen=True)
class SandboxConnectionHandles:
    customer: SandboxHandle
    runtime: SandboxHandle


class SandboxConnection:
    """Lifecycle owner for one customer-to-runtime sandbox connection.

    The connection starts both sandboxes with opposite ends of an in-memory
    duplex transport. It does not drive turns or choose what anyone says.
    """

    def __init__(
        self,
        *,
        customer: CustomerAgentSandbox,
        runtime: VoiceAgentSandbox,
        transport: InMemoryDuplexTransport | None = None,
    ) -> None:
        self.customer = customer
        self.runtime = runtime
        self.transport = transport or InMemoryDuplexTransport()
        self.handles: SandboxConnectionHandles | None = None

    async def start(
        self,
        *,
        example: BenchmarkExample,
        run_dir: Path,
        rng_seed: int,
    ) -> SandboxConnectionHandles:
        run_dir.mkdir(parents=True, exist_ok=True)
        customer_handle = await self.customer.start(
            example=example,
            run_dir=run_dir / "customer",
            rng_seed=rng_seed,
            transport=self.transport.customer,
        )
        runtime_handle = await self.runtime.start(
            example=example,
            run_dir=run_dir / "runtime",
            rng_seed=rng_seed,
            transport=self.transport.runtime,
        )
        self.handles = SandboxConnectionHandles(
            customer=customer_handle,
            runtime=runtime_handle,
        )
        return self.handles

    async def close(self) -> None:
        await self.customer.close()
        await self.runtime.close()
        await self.transport.close()

    @asynccontextmanager
    async def lifecycle(
        self,
        *,
        example: BenchmarkExample,
        run_dir: Path,
        rng_seed: int,
    ) -> AsyncIterator[SandboxConnectionHandles]:
        handles = await self.start(example=example, run_dir=run_dir, rng_seed=rng_seed)
        try:
            yield handles
        finally:
            await self.close()
