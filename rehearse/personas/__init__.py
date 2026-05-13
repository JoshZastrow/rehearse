"""Persona registry and prompts for Rehearse.

This package contains:
- Runtime prompts (coach, character, feedback) and intake helpers (legacy personas.py content)
- PersonaRegistry and PersonaRecord for the practice partner routing system
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from rehearse.types import CounterpartyPersona, IntakeRecord

ConsentClassification = Literal["granted", "declined", "unclear"]

CONSENT_PROMPT = (
    "Quick heads up before we start: this call is recorded so I can give you "
    "feedback afterward. If that's okay, just say yes to begin. If not, say "
    "no and I'll end the call now."
)

CONSENT_REPROMPT = (
    "Sorry — I didn't catch that. Are you okay with the call being recorded? "
    "Yes or no."
)

CONSENT_DECLINE_ACK = (
    "No problem — I won't record this call. Take care."
)

CONSENT_REMINDER = "Quick note: this call is being recorded."  # kept for test assertions

CONSENT_REMINDERS: tuple[str, ...] = (
    "Quick note: this call is being recorded.",
    "Just so you know, I record calls so I can give you better feedback.",
    "One quick thing — I'll be recording this for coaching purposes.",
    "Heads up: this call is recorded so I can follow up with feedback.",
    "Just a reminder that this call is being recorded for coaching.",
    "By the way, I do record sessions — helps me give you sharper feedback.",
)

_AFFIRMATIVE: frozenset[str] = frozenset(
    {
        "yes", "yeah", "yep", "yup", "sure", "okay", "ok", "go ahead",
        "that's fine", "thats fine", "sounds good", "alright", "absolutely", "fine",
    }
)

_NEGATIVE: frozenset[str] = frozenset(
    {
        "no", "nope", "nah", "not okay", "don't", "do not", "decline",
        "stop", "i'd rather not", "id rather not", "no thanks",
    }
)


def classify_consent(text: str) -> ConsentClassification:
    norm = text.strip().lower().rstrip(".!?,")
    if not norm:
        return "unclear"
    for phrase in _NEGATIVE:
        if norm == phrase or norm.startswith(phrase + " ") or norm.startswith(phrase + ","):
            return "declined"
    for phrase in _AFFIRMATIVE:
        if norm == phrase or norm.startswith(phrase + " ") or norm.startswith(phrase + ","):
            return "granted"
    return "unclear"


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

Eliciting practice partner preference:
Once you understand what the caller wants to practice, ask naturally who they'd
prefer to work through it with. Make it feel like genuine curiosity, not a menu.
Vary the phrasing — some options:
- "Do you tend to find it easier talking through this kind of thing with men or women?"
- "Is there a gender you feel more comfortable sparring with on something like this?"
- "Some people have a strong sense of who they'd rather practice with — do you?"
- "Would you rather run this with a man or a woman?"
Ask once, accept whatever they say, confirm briefly, and move on. If they say
they don't care or have no preference, that's a fine answer — don't push.
"""

FEEDBACK_PROMPT = """You are Rehearse, the coach again, debriefing the user
right after a short rehearsal. The roleplay is over. Drop any character voice.

Your one job for the next ~60 seconds:
- Reflect ONE specific thing the user did well, citing their actual words.
- Offer ONE concrete next line they could try if they have to do this for real.
- Be warm and direct. Plain spoken sentences. No bullet lists, no markdown.
  Two short paragraphs maximum.
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

_FEEDBACK_READY_CUES: frozenset[str] = frozenset({
    "i think i'm ready", "i'm ready", "let's go", "let's do it", "start",
    "begin", "go ahead", "let's practice", "start the practice",
    "great, thanks", "great thanks", "okay, thanks", "that was helpful",
    "that's helpful", "that was great", "i think i'm done", "let's stop there",
    "we can stop", "end scene", "cut",
})


def coach_system_prompt() -> str:
    return COACH_PROMPT


def feedback_coach_system_prompt() -> str:
    return FEEDBACK_PROMPT


def build_intake_record(
    *,
    session_id: str,
    user_turns: Sequence[str],
    captured_at: datetime,
    gender_preference: str | None = None,
) -> IntakeRecord:
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
        gender_preference=gender_preference,
        captured_at=captured_at,
    )


def compile_character(intake: IntakeRecord, *, compiled_at: datetime) -> CounterpartyPersona:
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
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return "The user wants help rehearsing a difficult conversation."
    sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0]
    return sentence[:200]


def _infer_relationship(text: str) -> str:
    lowered = text.lower()
    patterns = [
        ("recruiter", "recruiter"), ("hiring manager", "hiring manager"),
        ("manager", "manager"), ("boss", "manager"), ("founder", "cofounder"),
        ("cofounder", "cofounder"), ("coworker", "coworker"), ("client", "client"),
        ("customer", "customer"), ("partner", "partner"), ("spouse", "spouse"),
        ("mom", "parent"), ("dad", "parent"), ("parent", "parent"), ("friend", "friend"),
    ]
    for needle, label in patterns:
        if needle in lowered:
            return label
    return "counterparty"


def _infer_counterparty_name(text: str) -> str | None:
    match = re.search(r"\b(?:with|to|about)\s+([A-Z][a-z]+)\b", text)
    return match.group(1) if match else None


def _infer_counterparty_description(text: str, relationship: str) -> str:
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
    lowered = text.lower()
    if "equity" in lowered or "salary" in lowered or "compensation" in lowered:
        return "The user is trying to improve the financial terms of a job offer."
    if "offer" in lowered:
        return "The user wants to protect the quality of an offer without losing it."
    return f"The user wants a better outcome in a difficult conversation with a {relationship}."


def _infer_user_goal(text: str) -> str:
    lowered = text.lower()
    patterns = [
        r"(?:i want to|i need to|i'm trying to)\s+([^.!?]+)",
        r"(?:help me)\s+([^.!?]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, lowered)
        if matches:
            return matches[-1].strip().capitalize()
    return "Handle the conversation clearly and confidently." if lowered else "Clarify what they should say."


def _infer_tone(text: str) -> str | None:
    lowered = text.lower()
    for tone in ["calm", "confident", "warm", "firm", "direct", "diplomatic"]:
        if tone in lowered:
            return tone
    return None


def _infer_hot_buttons(intake: IntakeRecord) -> list[str]:
    buttons = ["vague asks", "rambling explanations"]
    lowered = f"{intake.situation} {intake.stakes}".lower()
    if "salary" in lowered or "compensation" in lowered or "equity" in lowered:
        buttons += ["demands without a clear business case", "surprise changes late in the offer process"]
    return buttons


def _infer_likely_reactions(intake: IntakeRecord) -> list[str]:
    reactions = ["asks the user to be more specific", "pushes back on unclear or oversized requests"]
    lowered = f"{intake.situation} {intake.stakes}".lower()
    if "job offer" in lowered or "compensation" in lowered or "salary" in lowered:
        reactions += ["anchors on internal compensation bands", "tests whether the user will stay collaborative under pressure"]
    return reactions
