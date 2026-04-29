"""Agent-shaped primitives for sandbox rollouts.

This mirrors the OpenAI Agents SDK convention closely enough for eval code:
call ``SandboxAgentRunner.run(starting_agent, input, context=..., max_turns=...)``
and receive a result with a final output plus metadata. The concrete agent can
own its own loop and tools; the eval environment only manages lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class SandboxAgentRunResult:
    """Result returned by one sandbox agent run."""

    final_output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SandboxAgent(Protocol):
    """Agent contract used by the sandbox runner."""

    name: str
    version: str

    async def run(
        self,
        input: Any,
        *,
        context: Any | None = None,
        max_turns: int = 10,
    ) -> SandboxAgentRunResult: ...


class SandboxAgentRunner:
    """Small runner with an Agents SDK-like API."""

    @classmethod
    async def run(
        cls,
        starting_agent: SandboxAgent,
        input: Any,
        *,
        context: Any | None = None,
        max_turns: int = 10,
    ) -> SandboxAgentRunResult:
        return await starting_agent.run(input, context=context, max_turns=max_turns)
