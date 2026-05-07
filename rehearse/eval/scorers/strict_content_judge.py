"""`StrictContentJudgeScorer` — Anthropic-backed content judge with deductions.

Replaces the DeepEval G-Eval path for the `voice-rollout-judges` eval. Two
reasons to fork:

  1. DeepEval's `GEval` with `MultiTurnParams` has been crashing inside
     `MetricToScorer` ("Unable to evaluate test cases that are not of type
     'LLMTestCase'"). Avoidable by collapsing to a single-turn case, but
     then we lose the multi-turn rubric we want.
  2. The audio judges run hot — they default to 1.0 on competent coaching.
     For aggregate calibration near 0.5 we need at least one dimension that
     is critical by construction. This scorer's rubric uses an explicit
     deduction list against named coaching failure modes, so a "decent but
     generic" rollout lands in 0.45–0.65 instead of 0.95.
"""

from __future__ import annotations

import json
from pathlib import Path

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.scorers.llm_judge import LLMJudge, LLMJudgeError
from rehearse.types import RubricScore


_PROMPT_VERSION = "strict-content-v1"

_SYSTEM = """You are scoring a coaching call transcript on whether what the coach said
moved the user toward a usable, speakable plan for their actual real-world
conversation. Be a strict, demanding judge — most coaches deserve 0.4–0.7.
Reserve 0.9+ for transcripts where every named failure mode is absent and
the coach produced at least one specific, attributable, speakable line.

Score `content_quality` ∈ [0.0, 1.0] (continuous). Anchors:
  • 0.95–1.00 — Excellent. User leaves with concrete phrasing they could
    actually say in real life. Coach named the tradeoff, helped them find
    the words, didn't lecture.
  • 0.70–0.85 — Useful and on-track but generic in places, or missed a
    chance to land specific phrasing.
  • 0.45–0.65 — Decent presence but didn't move the needle. Validating
    without progressing, or progressing without validating.
  • 0.20–0.40 — Coach occupied the time but the user didn't get traction.
    Generic empathy, vague advice, or solutioning before grounding.
  • 0.00–0.15 — Off-rails: unsafe, dismissive, lectures, or fundamentally
    wrong target.

Apply deductions explicitly (start from a base of 0.85 and deduct):
  -0.10 each: lectured / monologued / explained at length without check-in.
  -0.10 each: jumped to advice before naming what the user is feeling.
  -0.10 each: offered generic empathy that did not name anything specific.
  -0.10 each: never produced a concrete, speakable sentence the user
              could use in the real conversation.
  -0.10 each: dismissed, redirected, or invalidated the user's stated state.
  -0.05 each: filler/preamble/throat-clearing in a coach turn.
  -0.05 each: coach turn longer than ~4 sentences when the moment did not
              call for it.
  +0.05 once: coach explicitly named a tradeoff the user was avoiding.
  +0.05 once: produced a quotable, attributable line the user said landed.

Respond with ONLY this JSON object, nothing else:
{"content_quality": {"score": <0.0-1.0>, "deductions": ["<each applied
deduction or credit, briefly>"], "rationale": "<one to two sentences>"}}
"""


class StrictContentJudgeScorer:
    name = "strict_content_judge"
    dimension = "content_quality"

    def __init__(
        self,
        *,
        judge: LLMJudge | None = None,
        prompt_version: str = _PROMPT_VERSION,
    ) -> None:
        # Default to Sonnet — Opus 4.7 is the upstream default and is
        # ~5× the cost for content-quality judging. Override via `judge=`.
        self.judge = judge or LLMJudge(model="claude-sonnet-4-6")
        self.prompt_version = prompt_version

    async def score(
        self,
        example: BenchmarkExample,
        rollout: RolloutResult,
        run_id: str,
    ) -> list[RubricScore]:
        if rollout.status != "ok" or rollout.artifacts_dir is None:
            return [self._zero(example, run_id, f"rollout {rollout.status}")]
        transcript_path = Path(rollout.artifacts_dir) / "transcript.jsonl"
        if not transcript_path.exists():
            return [self._zero(example, run_id, "no transcript.jsonl")]
        transcript = _render(transcript_path.read_text().splitlines())
        scenario = example.payload.get("scenario") or {}
        user_prompt = (
            f"Scenario:\n  Situation: {scenario.get('situation', '<unspecified>')}\n"
            f"  User's goal: {scenario.get('goal', '<unspecified>')}\n"
            f"  Counterparty: {scenario.get('counterparty_role', '<unspecified>')} — "
            f"{scenario.get('counterparty_style', '')}\n"
            f"  Stakes: {scenario.get('stakes', '<unspecified>')}\n"
            f"  Opening emotional state: {scenario.get('emotional_state', '<unspecified>')}\n\n"
            f"Transcript:\n{transcript}\n"
        )
        try:
            output = await self.judge.judge(system=_SYSTEM, user=user_prompt)
        except LLMJudgeError as exc:
            return [self._zero(example, run_id, str(exc))]
        dim = output.get("content_quality")
        if not isinstance(dim, dict) or "score" not in dim:
            return [self._zero(example, run_id, "judge missing content_quality")]
        deductions = dim.get("deductions") or []
        rationale = dim.get("rationale") or ""
        if isinstance(deductions, list) and deductions:
            rationale = f"{rationale} | deductions: {'; '.join(map(str, deductions))}"
        return [
            RubricScore(
                run_id=run_id,
                example_id=example.id,
                dimension=self.dimension,
                value=float(dim["score"]),
                scorer="llm_judge",
                rationale=rationale or None,
                modality="text",
                judge_prompt_version=self.prompt_version,
            )
        ]

    def _zero(
        self, example: BenchmarkExample, run_id: str, rationale: str
    ) -> RubricScore:
        return RubricScore(
            run_id=run_id,
            example_id=example.id,
            dimension=self.dimension,
            value=0.0,
            scorer="llm_judge",
            rationale=rationale,
            modality="text",
            judge_prompt_version=self.prompt_version,
        )


def _render(lines: list[str]) -> str:
    out: list[str] = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            f = json.loads(line)
        except json.JSONDecodeError:
            continue
        spk = f.get("speaker", "?")
        label = "USER" if spk == "user" else "COACH" if spk == "coach" else spk.upper()
        out.append(f"[{i}] {label}: {(f.get('text') or '').strip()}")
    return "\n".join(out)
