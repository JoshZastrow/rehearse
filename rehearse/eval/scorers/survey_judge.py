"""Survey response quality scorer.

Reads a `SurveyRecord` and scores the quality of the user's survey response.
Default mode is deterministic (captured + verbatim richness). When an
Anthropic client is injected, the scorer uses Claude to judge the response.

Spec: docs/specs/v2026-05-13-survey-agent.md §6.3
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from rehearse.types import RubricDimension, RubricScore, SurveyRecord

_PROMPT_VERSION = "survey-judge-v1"

_LLM_SYSTEM = (
    "You are evaluating the quality of a user's survey response at the end of a "
    "coaching call. Score the response 0.0–1.0 on whether it is informative and "
    "actionable. 0 = no response or bare yes/no. 0.5 = vague but captured. "
    "1.0 = specific, grounded in something from the call.\n"
    "Respond with only a JSON object: {\"score\": <float>, \"rationale\": <str>}"
)


def score_survey_response(
    record: SurveyRecord,
    *,
    session_id: str,
    run_id: str,
    anthropic_client: Any | None = None,
) -> RubricScore:
    """Score survey response quality; use LLM judge when client is provided."""
    if anthropic_client is not None:
        return _llm_score(record, session_id=session_id, run_id=run_id, client=anthropic_client)
    return _deterministic_score(record, session_id=session_id, run_id=run_id)


def _deterministic_score(
    record: SurveyRecord,
    *,
    session_id: str,
    run_id: str,
) -> RubricScore:
    """Score based on capture status and verbatim richness (no LLM)."""
    captured_responses = [r for r in record.responses if r.captured]
    if not captured_responses:
        value = 0.0
        rationale = "No survey responses were captured."
    else:
        resp = captured_responses[0]
        verbatim = resp.verbatim or ""
        word_count = len(verbatim.split())
        if word_count >= 5:
            value = 1.0
            rationale = f"Response captured with {word_count} words of verbatim feedback."
        elif word_count >= 1:
            value = 0.5
            rationale = "Response captured but minimal (bare yes/no)."
        else:
            value = 0.3
            rationale = "Response captured without verbatim text."

    return RubricScore(
        run_id=run_id,
        example_id=session_id,
        session_id=session_id,
        dimension=RubricDimension.SURVEY_RESPONSE_QUALITY,
        value=value,
        scorer="deterministic",
        rationale=rationale,
        modality="text",
        judge_prompt_version=_PROMPT_VERSION,
    )


def _llm_score(
    record: SurveyRecord,
    *,
    session_id: str,
    run_id: str,
    client: Any,
) -> RubricScore:
    """Score using Claude; falls back to deterministic on any error."""
    import json as _json

    captured = [r for r in record.responses if r.captured and r.verbatim]
    if not captured:
        return _deterministic_score(record, session_id=session_id, run_id=run_id)

    verbatim = captured[0].verbatim or ""
    user_prompt = (
        f"Survey question: {captured[0].question_text}\n"
        f"User response: {verbatim}"
    )

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            system=_LLM_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )
        payload = _json.loads(message.content[0].text)
        value = float(payload["score"])
        rationale = str(payload.get("rationale", ""))
    except Exception:
        return _deterministic_score(record, session_id=session_id, run_id=run_id)

    return RubricScore(
        run_id=run_id,
        example_id=session_id,
        session_id=session_id,
        dimension=RubricDimension.SURVEY_RESPONSE_QUALITY,
        value=max(0.0, min(1.0, value)),
        scorer="llm_judge",
        rationale=rationale,
        modality="text",
        judge_prompt_version=_PROMPT_VERSION,
    )
