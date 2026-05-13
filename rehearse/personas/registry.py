"""PersonaRecord, PersonaRegistry, and the seed PERSONA_REGISTRY.

A Persona is a practice partner character: a combination of a Hume voice,
a CLM (fine-tuned model or system prompt), and metadata that the routing
agent uses to match the right character to the caller's situation.

Phase 1: prompt-based personas with Hume builtin voices.
Phase 2: clm_endpoint points to a fine-tuned model trained on real data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class PersonaRecord:
    """One practice partner persona available for routing."""

    id: str
    name: str
    description: str
    gender: Literal["male", "female"]
    voice_name: str
    voice_id: str | None = None
    system_prompt_template: str = ""
    clm_endpoint: str | None = None
    tags: list[str] = field(default_factory=list)

    def to_tool_dict(self) -> dict[str, Any]:
        """Serialize for the routing agent's list_personas tool response.

        Exposes only the fields the agent needs to make a routing decision.
        Does not expose voice_name, voice_id, or system_prompt_template.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "gender": self.gender,
            "tags": self.tags,
        }


class PersonaRegistry:
    """Lookup and filter personas by gender and tags."""

    def __init__(self, records: list[PersonaRecord]) -> None:
        self._records = records

    def get(self, persona_id: str) -> PersonaRecord | None:
        for r in self._records:
            if r.id == persona_id:
                return r
        return None

    def list(
        self,
        *,
        gender: str | None = None,
        tags: list[str] | None = None,
    ) -> list[PersonaRecord]:
        results = self._records
        if gender:
            results = [r for r in results if r.gender == gender]
        if tags:
            results = [r for r in results if all(t in r.tags for t in tags)]
        return results

    def default(self, gender: Literal["male", "female"]) -> PersonaRecord:
        """Return the default persona for a given gender."""
        candidates = self.list(gender=gender, tags=["default"])
        if candidates:
            return candidates[0]
        fallback = self.list(gender=gender)
        if fallback:
            return fallback[0]
        return self._records[0]

    def to_tool_response(self, *, gender: str | None = None) -> list[dict]:
        """Return all (or gender-filtered) personas as tool response dicts."""
        return [r.to_tool_dict() for r in self.list(gender=gender)]


# ---------------------------------------------------------------------------
# Seed registry — add new personas here; no code change needed elsewhere
# ---------------------------------------------------------------------------

PERSONA_REGISTRY: list[PersonaRecord] = [
    PersonaRecord(
        id="female_coach_default",
        name="Female Practice Partner",
        description=(
            "Warm but direct female counterparty for general rehearsal. "
            "Pushes back when the caller circles the point."
        ),
        gender="female",
        voice_name="Inspiring Woman",
        system_prompt_template=(
            "You are playing a female character named Sarah. "
            "Your role is {relationship}. Stay in character."
        ),
        tags=["default", "female"],
    ),
    PersonaRecord(
        id="male_coach_default",
        name="Male Practice Partner",
        description=(
            "Calm but direct male counterparty for general rehearsal. "
            "Listens, then challenges the caller's reasoning."
        ),
        gender="male",
        voice_name="Wise Man",
        system_prompt_template=(
            "You are playing a male character named Alex. "
            "Your role is {relationship}. Stay in character."
        ),
        tags=["default", "male"],
    ),
]
