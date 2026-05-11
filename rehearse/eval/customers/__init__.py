"""Protocol and result type for synthetic callers used in eval rollouts.

A `Caller` simulates the human counterparty side of a runtime-sandbox session.
Concrete implementations (e.g. `SyntheticCaller`) live alongside this module.

Naming note: `CustomerDriver` and `CustomerDriverResult` are deprecated aliases
kept for backwards compatibility for one release cycle. New code should use
`Caller` and `CallerResult`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from rehearse.transport import TwoWayChannel
from rehearse.types import Phase


class Caller(Protocol):
    """Simulate the human counterparty side of a runtime-sandbox rollout."""

    name: str
    version: str

    async def run(
        self,
        *,
        transport: TwoWayChannel,
        runtime_phase: Callable[[], Phase],
    ) -> "CallerResult": ...


@dataclass
class CallerResult:
    """Summary produced when a Caller finishes."""

    turns_sent: int
    turns_per_phase: dict[str, int]
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    token_usage: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Deprecation shims: test-automation-jargon names re-exported.
# ---------------------------------------------------------------------------

_DEPRECATED_NAMES: dict[str, str] = {
    "CustomerDriver": "Caller",
    "CustomerDriverResult": "CallerResult",
}


def __getattr__(name: str):
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
