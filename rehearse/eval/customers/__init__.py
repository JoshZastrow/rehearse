"""Protocol and result type for eval customer drivers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from rehearse.transport import RuntimeDuplexEndpoint
from rehearse.types import Phase


class CustomerDriver(Protocol):
    """Drive the customer side of a runtime-sandbox eval rollout."""

    name: str
    version: str

    async def run(
        self,
        *,
        transport: RuntimeDuplexEndpoint,
        runtime_phase: Callable[[], Phase],
    ) -> "CustomerDriverResult": ...


@dataclass
class CustomerDriverResult:
    """Summary produced when a CustomerDriver finishes."""

    turns_sent: int
    turns_per_phase: dict[str, int]
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
