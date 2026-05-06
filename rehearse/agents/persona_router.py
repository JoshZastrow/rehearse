"""Classify an inbound SMS body into one of the registered persona keys.

Called once per call at SMS-trigger time, before the outbound call connects.
The chosen key is stored on the session manifest and read by `HumeEVIClient`
at connect time to pick the right Hume EVI config.

All failure modes (no client, empty body, API error, unknown key) fall back
to `"default"` — the call still happens, just with the default persona.
"""

from __future__ import annotations

from typing import Any

import structlog

from rehearse.services.hume_configs import HumePersonaConfig

log = structlog.get_logger(__name__)

_DEFAULT_KEY = "default"
_SKIP_BODIES: frozenset[str] = frozenset({"", "<inbound-call>"})
_STRIP_CHARS = " \t\n\r.,!?\"'"


async def infer_persona_key(
    sms_body: str,
    personas: dict[str, HumePersonaConfig],
    *,
    anthropic_client: Any | None,
    model: str,
    fallback: str = _DEFAULT_KEY,
) -> str:
    """Return the persona key the classifier picked for this SMS body."""
    body = sms_body.strip()
    if anthropic_client is None:
        log.info("persona_router.skip", reason="no_client")
        return fallback
    if body in _SKIP_BODIES:
        log.info("persona_router.skip", reason="empty_or_inbound_marker")
        return fallback

    prompt = _build_prompt(body, personas)
    try:
        message = await anthropic_client.messages.create(
            model=model,
            max_tokens=20,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        log.warning("persona_router.error", body=body, error=str(exc))
        return fallback

    raw = _extract_text(message)
    candidate = raw.strip(_STRIP_CHARS).lower()
    if candidate not in personas:
        log.warning(
            "persona_router.unknown_key", body=body, raw=raw, candidate=candidate
        )
        return fallback
    log.info("persona_router.picked", body=body, key=candidate)
    return candidate


def _build_prompt(sms_body: str, personas: dict[str, HumePersonaConfig]) -> str:
    """Render the classifier prompt from the persona registry."""
    options = "\n".join(
        f"- {persona.persona_key}: {persona.routing_description}"
        for persona in personas.values()
    )
    return (
        "Pick the best persona for this rehearsal request.\n\n"
        "Options:\n"
        f"{options}\n\n"
        f'Request: "{sms_body}"\n\n'
        "Reply with ONLY the persona key (one word from the list above), "
        "nothing else."
    )


def _extract_text(message: Any) -> str:
    """Pull the first text block out of an Anthropic Messages API response."""
    content = getattr(message, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            return text
    return ""
