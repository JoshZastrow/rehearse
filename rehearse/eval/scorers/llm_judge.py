"""LLM-judge scorers.

Houses a reusable `LLMJudge` primitive (Anthropic-backed, structured-JSON
output) and concrete scorers that depend on it.

Today's scorer:
  - `TrajectoryJudgeScorer` — scores a multi-turn coaching transcript on
    `emotion_responsiveness` and `coaching_trajectory_quality` and emits a
    `weighted_reward` per `docs/specs/v2026-04-29-mme-seeded-rl-sandbox-eval.md`.

The primitive can be reused by future LLM-judge scorers (persona fidelity,
holistic usefulness) without re-implementing the API plumbing.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.types import RubricScore

_DEFAULT_JUDGE_MODEL = "claude-opus-4-7"
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class LLMJudgeError(RuntimeError):
    """Raised when a judge call fails or its output cannot be parsed."""


class LLMJudge:
    """Anthropic-backed judge with structured-JSON output.

    Reusable across scorers. Each `judge()` call sends a system+user prompt
    and parses the response as a JSON object. Caller defines the schema.
    """

    def __init__(
        self,
        *,
        model: str = _DEFAULT_JUDGE_MODEL,
        max_tokens: int = 2048,
        temperature: float | None = None,
        client: Any = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        # `temperature` is deprecated on Claude Opus 4.7 and other extended-
        # thinking-class models. Default to None and only pass it if the
        # caller explicitly opts in.
        self.temperature = temperature
        self._client = client

    def _client_lazy(self) -> Any:
        if self._client is None:
            from anthropic import AsyncAnthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise LLMJudgeError("ANTHROPIC_API_KEY not set")
            self._client = AsyncAnthropic(api_key=api_key)
        return self._client

    async def judge(self, *, system: str, user: str) -> dict[str, Any]:
        client = self._client_lazy()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        try:
            resp = await client.messages.create(**kwargs)
        except Exception as exc:
            raise LLMJudgeError(f"judge call failed: {type(exc).__name__}: {exc}") from exc

        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        return _parse_json(text)


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    match = _JSON_BLOCK.search(text)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    raise LLMJudgeError(f"could not parse JSON from judge output: {text[:300]!r}")


_TRAJECTORY_JUDGE_SYSTEM = (
    "You are an evaluator scoring a multi-turn coaching conversation on two "
    "dimensions. The user came in with a specific emotional state and a real "
    "conversation they need to prepare for. Your job is to score how well the "
    "coach (assistant in the transcript) handled this conversation.\n\n"
    "Score on a 0.0–1.0 scale (continuous, not just 0/0.5/1.0):\n\n"
    "1. emotion_responsiveness — Did the coach perceive and adapt to the user's "
    "emotional state?\n"
    "  • 1.0 — Reads the affect, validates it, and adapts as the user shifts.\n"
    "  • 0.5 — Generic empathy; not harmful, not clearly grounded in the state.\n"
    "  • 0.0 — Misreads, escalates, dismisses, or ignores the affect.\n\n"
    "2. coaching_trajectory_quality — Did the conversation move the user "
    "toward a better real-world conversation?\n"
    "  • 1.0 — User ends with clearer, calmer, actionable phrasing they could "
    "actually say.\n"
    "  • 0.5 — Some useful coaching, but generic or uneven across turns.\n"
    "  • 0.0 — Rambling, unsafe, invalidating, or not useful for the scenario.\n\n"
    "Pick 1–3 turn indices (0-based, looking at the [N] labels in the "
    "transcript) that drove each score.\n\n"
    "Respond with ONLY this JSON object, nothing else:\n"
    '{"emotion_responsiveness": {"score": <0.0-1.0>, "rationale": "<one '
    'sentence>", "key_moments": [<turn_idx>, ...]}, '
    '"coaching_trajectory_quality": {"score": <0.0-1.0>, "rationale": "<one '
    'sentence>", "key_moments": [<turn_idx>, ...]}}'
)


class TrajectoryJudgeScorer:
    """Score a coaching-dialogue transcript on the two RLE3 dimensions.

    Reads `transcript.jsonl` from `rollout.artifacts_dir`, calls an LLM judge,
    and emits three RubricScore rows:
      - emotion_responsiveness (0.0–1.0)
      - coaching_trajectory_quality (0.0–1.0)
      - weighted_reward (0.0–1.0; weighted aggregate)

    Default weights are 0.45 / 0.55 per spec §7.3 but can be overridden via
    `weights={"emotion_responsiveness": w1, "coaching_trajectory_quality": w2}`.
    Weights from the example payload's `rubric_weights` win when present.

    Persists the full judge output (with key_moments + rationales) at
    `rollout.artifacts_dir / "judge.json"`.
    """

    name = "trajectory_judge"
    dimension = "weighted_reward"

    DEFAULT_WEIGHTS = {
        "emotion_responsiveness": 0.45,
        "coaching_trajectory_quality": 0.55,
    }

    def __init__(
        self,
        *,
        judge: LLMJudge | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.judge = judge or LLMJudge()
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)

    async def score(
        self,
        example: BenchmarkExample,
        rollout: RolloutResult,
        run_id: str,
    ) -> list[RubricScore]:
        weights = dict(example.payload.get("rubric_weights") or self.weights)

        if rollout.status != "ok" or not rollout.artifacts_dir:
            return self._zeros(
                example, run_id, weights, rationale=f"rollout {rollout.status}"
            )

        transcript_path = Path(rollout.artifacts_dir) / "transcript.jsonl"
        if not transcript_path.exists():
            return self._zeros(
                example,
                run_id,
                weights,
                rationale=f"transcript not found at {transcript_path}",
            )

        transcript_lines = transcript_path.read_text().splitlines()
        if not transcript_lines:
            return self._zeros(
                example, run_id, weights, rationale="transcript is empty"
            )

        rendered_transcript = _render_transcript_for_judge(transcript_lines)
        scenario = example.payload.get("scenario") or {}
        opening_emotion = (
            example.expected.get("opening_emotion")
            or example.payload.get("opening_emotion")
            or scenario.get("emotional_state")
            or "unknown"
        )

        user_prompt = (
            f"Scenario:\n"
            f"  Situation: {scenario.get('situation', '<unspecified>')}\n"
            f"  User goal: {scenario.get('goal', '<unspecified>')}\n"
            f"  Counterparty: {scenario.get('counterparty_role', '<unspecified>')} — "
            f"{scenario.get('counterparty_style', '<unspecified>')}\n"
            f"  Stakes: {scenario.get('stakes', '<unspecified>')}\n"
            f"  User's opening emotional state: {opening_emotion}\n\n"
            f"Transcript:\n{rendered_transcript}\n"
        )

        try:
            judge_output = await self.judge.judge(
                system=_TRAJECTORY_JUDGE_SYSTEM, user=user_prompt
            )
        except LLMJudgeError as exc:
            return self._zeros(example, run_id, weights, rationale=str(exc))

        try:
            er_score, er_rationale = _extract_dim(judge_output, "emotion_responsiveness")
            ct_score, ct_rationale = _extract_dim(
                judge_output, "coaching_trajectory_quality"
            )
        except KeyError as exc:
            return self._zeros(
                example,
                run_id,
                weights,
                rationale=f"judge output missing key: {exc}",
            )

        weighted = (
            weights.get("emotion_responsiveness", 0.0) * er_score
            + weights.get("coaching_trajectory_quality", 0.0) * ct_score
        )

        judge_artifact = {
            "judge_output": judge_output,
            "weights": weights,
            "weighted_reward": weighted,
        }
        (Path(rollout.artifacts_dir) / "judge.json").write_text(
            json.dumps(judge_artifact, indent=2)
        )

        return [
            RubricScore(
                run_id=run_id,
                example_id=example.id,
                dimension="emotion_responsiveness",
                value=er_score,
                scorer="llm_judge",
                rationale=er_rationale,
            ),
            RubricScore(
                run_id=run_id,
                example_id=example.id,
                dimension="coaching_trajectory_quality",
                value=ct_score,
                scorer="llm_judge",
                rationale=ct_rationale,
            ),
            RubricScore(
                run_id=run_id,
                example_id=example.id,
                dimension="weighted_reward",
                value=weighted,
                scorer="llm_judge",
                rationale=(
                    f"{weights.get('emotion_responsiveness', 0.0):.2f}*"
                    f"{er_score:.2f} + "
                    f"{weights.get('coaching_trajectory_quality', 0.0):.2f}*"
                    f"{ct_score:.2f} = {weighted:.3f}"
                ),
            ),
        ]

    @staticmethod
    def _zeros(
        example: BenchmarkExample,
        run_id: str,
        weights: dict[str, float],
        rationale: str,
    ) -> list[RubricScore]:
        return [
            RubricScore(
                run_id=run_id,
                example_id=example.id,
                dimension=dim,
                value=0.0,
                scorer="llm_judge",
                rationale=rationale,
            )
            for dim in (
                "emotion_responsiveness",
                "coaching_trajectory_quality",
                "weighted_reward",
            )
        ]


def _render_transcript_for_judge(jsonl_lines: list[str]) -> str:
    """Render JSONL TranscriptFrames as `[idx] SPEAKER: text`."""
    out: list[str] = []
    for idx, line in enumerate(jsonl_lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        speaker = row.get("speaker", "unknown")
        if speaker == "user":
            label = "USER"
        elif speaker == "coach":
            label = "COACH"
        else:
            label = speaker.upper()
        out.append(f"[{idx}] {label}: {row.get('text', '').strip()}")
    return "\n".join(out)


def _extract_dim(judge_output: dict[str, Any], dim: str) -> tuple[float, str | None]:
    block = judge_output.get(dim)
    if not isinstance(block, dict):
        raise KeyError(dim)
    score_raw = block.get("score")
    if score_raw is None:
        raise KeyError(f"{dim}.score")
    try:
        score = float(score_raw)
    except (TypeError, ValueError) as exc:
        raise KeyError(f"{dim}.score (invalid: {score_raw!r})") from exc
    score = max(0.0, min(1.0, score))
    rationale_parts = []
    if block.get("rationale"):
        rationale_parts.append(str(block["rationale"]))
    if block.get("key_moments"):
        rationale_parts.append(f"key_moments={block['key_moments']}")
    rationale = " | ".join(rationale_parts) if rationale_parts else None
    return score, rationale
