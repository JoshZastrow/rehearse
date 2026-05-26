"""Duplex transport boundary for runtime eval and serving.

Production uses Twilio as the phone gateway. Evals need the same conceptual
duplex boundary without the phone network: a customer side can send audio/text
events to the runtime side, and the runtime side can send assistant events back.

This module intentionally does not model conversation policy. It only moves
events between two endpoints.

Canonical location: `rehearse/transport.py`. The old path
`rehearse/eval/transports` re-exports from here for backwards compatibility.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import uuid4

TransportSide = Literal["customer", "runtime"]
TransportStatus = Literal["open", "closed"]
TransportEventKind = Literal["audio", "text", "prosody", "control"]


class TransportClosedError(RuntimeError):
    """Raised when sending on a closed transport."""


@dataclass(frozen=True)
class TransportEvent:
    """One event crossing the sandbox transport boundary."""

    id: str
    source: TransportSide
    kind: TransportEventKind
    payload: dict[str, Any] = field(default_factory=dict)
    data: bytes | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@runtime_checkable
class TwoWayChannel(Protocol):
    """One side of a duplex runtime transport."""

    side: TransportSide

    async def send(
        self,
        kind: TransportEventKind,
        *,
        payload: dict[str, Any] | None = None,
        data: bytes | None = None,
    ) -> TransportEvent: ...

    async def receive(self, timeout_s: float | None = None) -> TransportEvent: ...

    async def close(self) -> None: ...


class InMemoryTransportEndpoint:
    """Queue-backed endpoint used by eval and unit tests."""

    def __init__(
        self,
        *,
        side: TransportSide,
        inbox: asyncio.Queue[TransportEvent],
        peer_inbox: asyncio.Queue[TransportEvent],
        owner: InMemoryDuplexTransport,
    ) -> None:
        self.side = side
        self._inbox = inbox
        self._peer_inbox = peer_inbox
        self._owner = owner

    async def send(
        self,
        kind: TransportEventKind,
        *,
        payload: dict[str, Any] | None = None,
        data: bytes | None = None,
    ) -> TransportEvent:
        if self._owner.status == "closed":
            raise TransportClosedError("transport is closed")
        event = TransportEvent(
            id=uuid4().hex,
            source=self.side,
            kind=kind,
            payload=dict(payload or {}),
            data=data,
        )
        self._owner.record(event)
        await self._peer_inbox.put(event)
        return event

    async def receive(self, timeout_s: float | None = None) -> TransportEvent:
        if timeout_s is None:
            return await self._inbox.get()
        return await asyncio.wait_for(self._inbox.get(), timeout=timeout_s)

    async def close(self) -> None:
        await self._owner.close()


class InMemoryTwoWayChannel:
    """In-memory two-way channel pair connecting a synthetic caller to a runtime.

    Optionally invokes ``on_event`` for every event that crosses the boundary.
    Used by --verbose to stream turns to stdout during a rollout. The hook
    swallows its own exceptions so a logging error cannot break a rollout.
    """

    def __init__(
        self,
        *,
        on_event: Callable[[TransportEvent], None] | None = None,
    ) -> None:
        self.status: TransportStatus = "open"
        self.events: list[TransportEvent] = []
        self.on_event = on_event
        customer_inbox: asyncio.Queue[TransportEvent] = asyncio.Queue()
        runtime_inbox: asyncio.Queue[TransportEvent] = asyncio.Queue()
        self.customer = InMemoryTransportEndpoint(
            side="customer",
            inbox=customer_inbox,
            peer_inbox=runtime_inbox,
            owner=self,
        )
        self.runtime = InMemoryTransportEndpoint(
            side="runtime",
            inbox=runtime_inbox,
            peer_inbox=customer_inbox,
            owner=self,
        )

    async def close(self) -> None:
        self.status = "closed"

    def record(self, event: TransportEvent) -> None:
        self.events.append(event)
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:
                pass


# Public alias used by RuntimeHost and spec §5.2.
Channel = TwoWayChannel
RuntimeTransport = TwoWayChannel  # legacy alias; prefer Channel.


# ---------------------------------------------------------------------------
# Deprecation shims: telecom-jargon names re-exported with DeprecationWarning.
# Old names will be removed in the eval-system PR after this one (Phase 5).
# ---------------------------------------------------------------------------

_DEPRECATED_NAMES: dict[str, str] = {
    "InMemoryDuplexTransport": "InMemoryTwoWayChannel",
    "RuntimeDuplexEndpoint": "TwoWayChannel",
}


def __getattr__(name: str):
    """Module-level __getattr__ for soft-deprecated names (PEP 562)."""
    if name in _DEPRECATED_NAMES:
        import warnings

        new = _DEPRECATED_NAMES[name]
        warnings.warn(
            f"{name} is deprecated; use {new}. "
            "The old name will be removed in the next eval-system release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
