"""Convert (BenchmarkExample, RolloutResult) into DeepEval test cases.

DeepEval's metric API consumes either an `LLMTestCase` (single-turn) or a
`ConversationalTestCase` (multi-turn). Coaching trajectories are inherently
multi-turn, so the conversational shape is canonical; the single-turn shape
is provided for the rare metric that only judges the final exchange.

Source order for the transcript:
  1. `rollout.artifacts_dir / "transcript.jsonl"` if `artifacts_dir` is set
     and the file exists.
  2. `rollout.payload["transcript"]` (a list of frame dicts) otherwise.

Each frame is `{"speaker": "user" | "coach" | ..., "text": str, ...}`. The
`speaker` field is mapped to a DeepEval `Turn.role`: `coach` → `assistant`,
`user` → `user`, anything else → `system`. Lines that are not valid JSON
are dropped (matching the existing judge's behavior).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deepeval.test_case import ConversationalTestCase, LLMTestCase, Turn

from rehearse.eval.protocols import BenchmarkExample, RolloutResult


def _frames_from_rollout(rollout: RolloutResult) -> list[dict[str, Any]]:
    """Resolve transcript frames from disk artifacts or inline payload."""

    if rollout.artifacts_dir is not None:
        path = Path(rollout.artifacts_dir) / "transcript.jsonl"
        if path.exists():
            return _parse_jsonl(path.read_text().splitlines())

    if rollout.payload and isinstance(rollout.payload.get("transcript"), list):
        return [f for f in rollout.payload["transcript"] if isinstance(f, dict)]

    raise ValueError(
        f"no transcript available for rollout {rollout.example_id} "
        f"(artifacts_dir={rollout.artifacts_dir}, payload keys="
        f"{list(rollout.payload.keys()) if rollout.payload else None})"
    )


def _parse_jsonl(lines: list[str]) -> list[dict[str, Any]]:
    """Best-effort JSONL parse; silently skips malformed lines."""

    out: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


_ROLE_MAP = {"coach": "assistant", "user": "user"}


def _frame_to_turn(frame: dict[str, Any]) -> Turn:
    speaker = str(frame.get("speaker", "")).lower()
    role = _ROLE_MAP.get(speaker, "system")
    text = str(frame.get("text", "")).strip()
    return Turn(role=role, content=text)


def _scenario_blurb(example: BenchmarkExample) -> str | None:
    """Render the scenario fields as a short string for the metric prompt."""

    scenario = example.payload.get("scenario") or {}
    if not isinstance(scenario, dict) or not scenario:
        return None

    parts: list[str] = []
    if v := scenario.get("situation"):
        parts.append(f"Situation: {v}")
    if v := scenario.get("goal"):
        parts.append(f"User goal: {v}")
    cp_role = scenario.get("counterparty_role")
    cp_style = scenario.get("counterparty_style")
    if cp_role or cp_style:
        parts.append(f"Counterparty: {cp_role or '?'} ({cp_style or 'unspecified'})")
    if v := scenario.get("stakes"):
        parts.append(f"Stakes: {v}")

    opening_emotion = (
        example.expected.get("opening_emotion")
        or example.payload.get("opening_emotion")
        or scenario.get("emotional_state")
    )
    if opening_emotion:
        parts.append(f"User's opening emotional state: {opening_emotion}")

    return " | ".join(parts) if parts else None


_DEFAULT_CHATBOT_ROLE = (
    "An emotionally attuned conversation coach helping the user rehearse a "
    "real conversation they need to have. Reads affect, validates, paces, and "
    "moves the user toward clearer phrasing."
)


def to_conversational_test_case(
    example: BenchmarkExample, rollout: RolloutResult
) -> ConversationalTestCase:
    """Build a DeepEval `ConversationalTestCase` from a coaching rollout.

    Maps transcript frames to `Turn`s; surfaces scenario metadata on the
    test case for metric grounding.
    """

    frames = _frames_from_rollout(rollout)
    if not frames:
        raise ValueError(f"transcript is empty for rollout {rollout.example_id}")

    turns = [_frame_to_turn(f) for f in frames]

    return ConversationalTestCase(
        turns=turns,
        scenario=_scenario_blurb(example),
        chatbot_role=_DEFAULT_CHATBOT_ROLE,
    )


def to_llm_test_case(example: BenchmarkExample, rollout: RolloutResult) -> LLMTestCase:
    """Collapse a coaching trajectory into a single-turn `LLMTestCase`.

    Useful for DeepEval metrics that only operate on (input, actual_output)
    pairs — e.g. faithfulness, answer relevance. Maps to the *last* user
    message and the *last* coach response.
    """

    frames = _frames_from_rollout(rollout)
    last_user = next(
        (f for f in reversed(frames) if str(f.get("speaker", "")).lower() == "user"),
        None,
    )
    last_coach = next(
        (f for f in reversed(frames) if str(f.get("speaker", "")).lower() == "coach"),
        None,
    )

    return LLMTestCase(
        input=str(last_user.get("text", "")) if last_user else "",
        actual_output=str(last_coach.get("text", "")) if last_coach else "",
    )
