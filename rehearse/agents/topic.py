"""Topic classification for intake situations.

Maps a free-text situation to one of three categories used for per-topic
gender preference routing.
"""

from __future__ import annotations

from typing import Literal

import structlog

log = structlog.get_logger(__name__)

TopicCategory = Literal["work", "relationship", "other"]

_PROMPT = """\
Classify the following situation into exactly one category.

Categories:
- work: conversations about jobs, managers, colleagues, salary, performance, career, hiring, business
- relationship: conversations about romantic partners, family, friends, personal bonds, moving in, breakups
- other: anything that does not clearly fit work or relationship

Reply with exactly one word: work, relationship, or other.

Situation: {situation}"""


async def classify_topic(
    situation: str,
    *,
    client,
    model: str = "claude-haiku-4-5-20251001",
) -> TopicCategory:
    """Classify a situation string into a topic category.

    Returns "other" on any error or unrecognized response.
    max_tokens=10 and temperature=0 keep this fast and deterministic.
    """
    if not situation or not situation.strip():
        return "other"

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=10,
            temperature=0,
            messages=[
                {"role": "user", "content": _PROMPT.format(situation=situation.strip())}
            ],
        )
        raw = ""
        if response.content:
            raw = response.content[0].text.strip().lower()

        if raw in ("work", "relationship", "other"):
            return raw  # type: ignore[return-value]

        log.warning("classify_topic.unexpected_response", raw=raw)
        return "other"

    except Exception as exc:
        log.warning("classify_topic.failed", error=str(exc))
        return "other"
