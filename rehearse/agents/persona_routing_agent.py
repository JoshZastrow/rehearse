"""PersonaRoutingAgent — lightweight agent that picks a persona via tool call.

Runs once at IntakeComplete. Given the intake transcript and an optional gender
hint, it calls the list_personas tool and returns the best-matching PersonaRecord.
Falls back to the gender-appropriate default on any error.

Model: claude-haiku-4-5-20251001 (fast, cheap, sufficient for one-shot selection).
Max tokens: 30. Temperature: 0. Runs within the bridge utterance window (~2s).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import structlog

from rehearse.personas.registry import PersonaRecord, PersonaRegistry

log = structlog.get_logger(__name__)

_TOOL_SCHEMA: dict[str, Any] = {
    "name": "list_personas",
    "description": (
        "Return available practice partner personas. "
        "Optionally filter by gender. Returns a list of persona objects."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "gender": {
                "type": "string",
                "enum": ["male", "female"],
                "description": "Filter personas by gender.",
            }
        },
    },
}

_SYSTEM = (
    "You are a routing agent. Given a caller's intake transcript, call the "
    "list_personas tool (optionally filtered by gender), then respond with JSON: "
    '{"persona_id": "<id>"}. Choose the persona whose description best fits '
    "the caller's situation. Return ONLY the JSON object."
)


class PersonaRoutingAgent:
    """Selects a PersonaRecord from the registry via a single LLM tool call."""

    def __init__(
        self,
        registry: PersonaRegistry,
        *,
        client: Any,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self._registry = registry
        self._client = client
        self._model = model

    async def select(
        self,
        transcript: str,
        gender_hint: str | None,
    ) -> PersonaRecord:
        """Return the best-matching PersonaRecord for this intake transcript.

        Falls back to the gender-appropriate default persona on any error.
        """
        try:
            return await self._run(transcript, gender_hint)
        except Exception as exc:
            log.warning("persona_routing_agent.failed", error=str(exc))
            return self._fallback(gender_hint)

    async def _run(self, transcript: str, gender_hint: str | None) -> PersonaRecord:
        messages: list[dict] = [
            {"role": "user", "content": f"Intake transcript:\n{transcript}"}
        ]

        # Turn 1: model may call list_personas
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=200,
            temperature=0,
            system=_SYSTEM,
            tools=[_TOOL_SCHEMA],
            messages=messages,
        )

        # Handle tool use
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "list_personas":
                    gender_filter = block.input.get("gender") or gender_hint
                    personas = self._registry.to_tool_response(gender=gender_filter)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(personas),
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            # Turn 2: model picks from tool results
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=30,
                temperature=0,
                system=_SYSTEM,
                tools=[_TOOL_SCHEMA],
                messages=messages,
            )

        # Parse final response
        for block in response.content:
            if block.type == "text":
                text = block.text.strip()
                try:
                    data = json.loads(text)
                    persona_id = data.get("persona_id", "")
                    record = self._registry.get(persona_id)
                    if record:
                        return record
                except (json.JSONDecodeError, AttributeError):
                    pass

        return self._fallback(gender_hint)

    def _fallback(self, gender_hint: str | None) -> PersonaRecord:
        gender = gender_hint if gender_hint in ("male", "female") else "female"
        return self._registry.default(gender=gender)
