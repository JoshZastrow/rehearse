"""Build runtime prompts and lightweight structured intake records.

This file keeps the coach prompt, character prompt, and simple deterministic
helpers that turn intake transcript snippets into a stored intake record and a
compiled counterparty persona.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime

from rehearse.types import CounterpartyPersona, IntakeRecord

COACH_PROMPT = """You are Rehearse, a calm and practical conversation coach.

You are on a live phone call with a user who wants to practice a difficult
conversation. Keep your replies short, natural to say aloud, and easy to
follow over audio.

Your goals are:
- quickly understand what the user wants to practice
- ask one focused question at a time
- reflect back the real constraint or tension you hear
- coach toward a concrete next line the user could actually say

Rules:
- sound like a warm human coach, not a chatbot
- avoid bullet points, stage directions, or markdown
- avoid long monologues
- if context is missing, ask for it instead of guessing
- if the user seems ready, move from intake into realistic rehearsal
"""

CHARACTER_PROMPT_TEMPLATE = """You are roleplaying the other person in the user's conversation.

Stay in character. Speak in short, natural spoken sentences. React to what the
user says instead of explaining your role. Keep the interaction realistic and
slightly resistant, not theatrical.

Counterparty:
- relationship: {relationship}
- description: {description}
- stakes: {stakes}
- user goal: {user_goal}

Hot buttons:
{hot_buttons}

Likely reactions:
{likely_reactions}
"""


def coach_system_prompt() -> str:
    """Return the runtime's default coach prompt."""
    return COACH_PROMPT


def build_intake_record(
    *,
    session_id: str,
    user_turns: Sequence[str],
    captured_at: datetime,
) -> IntakeRecord:
    """Build a simple deterministic intake record from user transcript turns."""
    joined = " ".join(turn.strip() for turn in user_turns if turn.strip())
    relationship = _infer_relationship(joined)
    description = _infer_counterparty_description(joined, relationship)
    return IntakeRecord(
        session_id=session_id,
        situation=_summarize_situation(joined),
        counterparty_name=_infer_counterparty_name(joined),
        counterparty_relationship=relationship,
        counterparty_description=description,
        stakes=_infer_stakes(joined, relationship),
        user_goal=_infer_user_goal(joined),
        desired_tone=_infer_tone(joined),
        captured_at=captured_at,
    )


def compile_character(intake: IntakeRecord, *, compiled_at: datetime) -> CounterpartyPersona:
    """Compile one intake record into a deterministic practice persona."""
    hot_buttons = _infer_hot_buttons(intake)
    likely_reactions = _infer_likely_reactions(intake)
    prompt = CHARACTER_PROMPT_TEMPLATE.format(
        relationship=intake.counterparty_relationship,
        description=intake.counterparty_description,
        stakes=intake.stakes,
        user_goal=intake.user_goal,
        hot_buttons="\n".join(f"- {item}" for item in hot_buttons),
        likely_reactions="\n".join(f"- {item}" for item in likely_reactions),
    )
    return CounterpartyPersona(
        session_id=intake.session_id,
        name=intake.counterparty_name,
        relationship=intake.counterparty_relationship,
        personality_prompt=prompt,
        hot_buttons=hot_buttons,
        likely_reactions=likely_reactions,
        compiled_at=compiled_at,
    )


def character_system_prompt(persona: CounterpartyPersona | str) -> str:
    """Return the stored persona prompt or render a plain string as character context."""
    if isinstance(persona, CounterpartyPersona):
        return persona.personality_prompt
    fallback = persona.strip() or "Be the other person in the conversation."
    return CHARACTER_PROMPT_TEMPLATE.format(
        relationship="counterparty",
        description=fallback,
        stakes="Respond naturally to the user's ask.",
        user_goal="Get through the conversation realistically.",
        hot_buttons="- unclear requests",
        likely_reactions="- asks follow-up questions",
    )


def _summarize_situation(text: str) -> str:
    """Return a short situation summary from the user's intake text."""
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return "The user wants help rehearsing a difficult conversation."
    sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0]
    return sentence[:200]


def _infer_relationship(text: str) -> str:
    """Infer the counterparty relationship from plain transcript cues."""
    lowered = text.lower()
    patterns = [
        ("recruiter", "recruiter"),
        ("hiring manager", "hiring manager"),
        ("manager", "manager"),
        ("boss", "manager"),
        ("founder", "cofounder"),
        ("cofounder", "cofounder"),
        ("coworker", "coworker"),
        ("client", "client"),
        ("customer", "customer"),
        ("partner", "partner"),
        ("spouse", "spouse"),
        ("mom", "parent"),
        ("dad", "parent"),
        ("parent", "parent"),
        ("friend", "friend"),
    ]
    for needle, label in patterns:
        if needle in lowered:
            return label
    return "counterparty"


def _infer_counterparty_name(text: str) -> str | None:
    """Infer a counterparty name when the caller explicitly names someone."""
    match = re.search(r"\b(?:with|to|about)\s+([A-Z][a-z]+)\b", text)
    if match:
        return match.group(1)
    return None


def _infer_counterparty_description(text: str, relationship: str) -> str:
    """Return a plain-English description of the counterparty."""
    lowered = text.lower()
    if "job offer" in lowered or "compensation" in lowered or "salary" in lowered:
        return (
            f"A {relationship} discussing a job offer who cares about budget, fairness, "
            "and whether the user sounds well prepared."
        )
    return (
        f"A {relationship} with their own constraints who will react to the user's tone, "
        "clarity, and confidence."
    )


def _infer_stakes(text: str, relationship: str) -> str:
    """Infer the main stakes described by the caller."""
    lowered = text.lower()
    if "equity" in lowered or "salary" in lowered or "compensation" in lowered:
        return "The user is trying to improve the financial terms of a job offer."
    if "offer" in lowered:
        return "The user wants to protect the quality of an offer without losing it."
    return f"The user wants a better outcome in a difficult conversation with a {relationship}."


def _infer_user_goal(text: str) -> str:
    """Infer the user's desired outcome from the transcript."""
    lowered = text.lower()
    patterns = [
        r"(?:i want to|i need to|i'm trying to)\s+([^.!?]+)",
        r"(?:help me)\s+([^.!?]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, lowered)
        if matches:
            return matches[-1].strip().capitalize()
    if lowered:
        return "Handle the conversation clearly and confidently."
    return "Clarify what they should say."


def _infer_tone(text: str) -> str | None:
    """Infer a desired tone when the caller says one explicitly."""
    lowered = text.lower()
    for tone in ["calm", "confident", "warm", "firm", "direct", "diplomatic"]:
        if tone in lowered:
            return tone
    return None


def _infer_hot_buttons(intake: IntakeRecord) -> list[str]:
    """Return likely triggers for the counterparty based on the intake."""
    buttons = ["vague asks", "rambling explanations"]
    lowered = f"{intake.situation} {intake.stakes}".lower()
    if "salary" in lowered or "compensation" in lowered or "equity" in lowered:
        buttons.append("demands without a clear business case")
        buttons.append("surprise changes late in the offer process")
    return buttons


def _infer_likely_reactions(intake: IntakeRecord) -> list[str]:
    """Return likely reactions for the compiled counterparty persona."""
    reactions = [
        "asks the user to be more specific",
        "pushes back on unclear or oversized requests",
    ]
    lowered = f"{intake.situation} {intake.stakes}".lower()
    if "job offer" in lowered or "compensation" in lowered or "salary" in lowered:
        reactions.append("anchors on internal compensation bands")
        reactions.append("tests whether the user will stay collaborative under pressure")
    return reactions
