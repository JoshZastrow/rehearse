"""AgentRegistry — maps role names to RehearseAgent instances.

Built once at app startup and injected into CLMResponder. Adding a new agent
requires only: create the class, call registry.register(), and add routing
logic in the router. No changes to CLMResponder or any other agent.
"""

from __future__ import annotations

from rehearse.agents.roles.base import RehearseAgent


class AgentRegistry:
    """Name → agent instance map. Thread-safe for reads after startup."""

    def __init__(self) -> None:
        self._agents: dict[str, RehearseAgent] = {}

    def register(self, agent: RehearseAgent) -> None:
        """Register an agent. Overwrites any existing agent with the same name."""
        self._agents[agent.name] = agent

    def get(self, name: str) -> RehearseAgent:
        """Return the named agent, or intake_coach as a safe default."""
        return self._agents.get(name, self._agents.get("intake_coach", next(iter(self._agents.values()))))

    def names(self) -> list[str]:
        return list(self._agents.keys())
