"""`SurveyPresenceJudge` — did the survey happen and was the response coherent?

Reads `survey.json` + SURVEY-phase rows from `transcript.jsonl` in the rollout
artifacts directory. Sends both to Claude Haiku and asks it to confirm:
  - Whether a survey question was asked
  - Whether the caller responded meaningfully
  - How specific/grounded the response was

Uses `LLMJudge` (same pattern as `StrictContentJudgeScorer`) which lazily
loads the ANTHROPIC_API_KEY — no explicit client injection needed for normal
use. Pass `judge=` to override the model or inject a mock client in tests.

Spec: docs/specs/v2026-05-13-survey-agent.md §11
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.scorers.llm_judge import LLMJudge, LLMJudgeError
from rehearse.types import RubricDimension, RubricScore, SurveyRecord

_PROMPT_VERSION = "survey-presence-v1"

_SYSTEM = """You are scoring whether a post-call survey exchange was completed and coherent.

You will receive the survey question the coach asked, the caller's captured
response (if any), and the SURVEY-phase transcript excerpt.

Score `survey_response_quality` ∈ [0.0, 1.0] using these anchors:

  1.0 — Response is specific and grounded: references something concrete from
        the call ("the part about slowing down my delivery helped").
  0.7 — Genuine but vague: caller confirms it was useful without specifics.
  0.4 — Bare yes/no with no elaboration.
  0.1 — Survey question was asked but caller gave no usable response.
  0.0 — Survey did not run or survey.json is absent.

Respond with ONLY this JSON object, nothing else:
{"survey_response_quality": {"score": <0.0-1.0>, "rationale": "<one sentence>"}}
"""


class SurveyPresenceJudge:
    """Score whether the survey phase ran and produced a coherent caller response.

    Default: uses LLMJudge (Haiku) which reads ANTHROPIC_API_KEY from env.
    Override via `judge=LLMJudge(model=..., client=...)` for tests or cost control.
    """

    name = "survey_presence_judge"
    dimension = RubricDimension.SURVEY_RESPONSE_QUALITY

    def __init__(self, *, judge: LLMJudge | None = None, anthropic_client: Any | None = None) -> None:
        # `anthropic_client` accepted for test-injection compatibility; it is
        # wrapped in an LLMJudge so the real judging path is always the same.
        if anthropic_client is not None:
            self._judge = LLMJudge(model="claude-haiku-4-5-20251001", client=anthropic_client)
        else:
            self._judge = judge or LLMJudge(model="claude-haiku-4-5-20251001")

    async def score(
        self,
        example: BenchmarkExample,
        rollout: RolloutResult,
        run_id: str,
    ) -> list[RubricScore]:
        if rollout.status != "ok" or rollout.artifacts_dir is None:
            return [self._zero(example, run_id, f"rollout {rollout.status}")]

        survey_path = Path(rollout.artifacts_dir) / "survey.json"
        if not survey_path.exists():
            return [self._zero(example, run_id, "no survey.json in rollout artifacts")]

        try:
            record = SurveyRecord.model_validate_json(survey_path.read_text())
        except Exception as exc:
            return [self._zero(example, run_id, f"survey.json parse error: {exc}")]

        transcript_excerpt = _load_survey_transcript(rollout.artifacts_dir)
        return [await self._llm_score(example, run_id, record, transcript_excerpt)]

    async def _llm_score(
        self,
        example: BenchmarkExample,
        run_id: str,
        record: SurveyRecord,
        transcript_excerpt: str,
    ) -> RubricScore:
        captured = [r for r in record.responses if r.captured]
        question = record.questions[0].text if record.questions else "(no question recorded)"
        verbatim = captured[0].verbatim if captured else "(no response captured)"

        user_prompt = (
            f"Survey question asked by coach: {question}\n"
            f"Caller's captured response: {verbatim}\n"
        )
        if transcript_excerpt:
            user_prompt += f"\nSurvey phase transcript:\n{transcript_excerpt}"

        try:
            output = await self._judge.judge(system=_SYSTEM, user=user_prompt)
        except LLMJudgeError as exc:
            return self._zero(example, run_id, f"judge error: {exc}")

        dim = output.get("survey_response_quality")
        if not isinstance(dim, dict) or "score" not in dim:
            return self._zero(example, run_id, "judge returned unexpected shape")

        value = max(0.0, min(1.0, float(dim["score"])))
        rationale = str(dim.get("rationale", ""))
        return RubricScore(
            run_id=run_id,
            example_id=example.id,
            dimension=self.dimension,
            value=value,
            scorer="llm_judge",
            rationale=rationale or None,
            modality="text",
            judge_prompt_version=_PROMPT_VERSION,
        )

    def _zero(self, example: BenchmarkExample, run_id: str, rationale: str) -> RubricScore:
        return RubricScore(
            run_id=run_id,
            example_id=example.id,
            dimension=self.dimension,
            value=0.0,
            scorer="llm_judge",
            rationale=rationale,
            modality="text",
            judge_prompt_version=_PROMPT_VERSION,
        )


def _load_survey_transcript(artifacts_dir: Path | str) -> str:
    path = Path(artifacts_dir) / "transcript.jsonl"
    if not path.exists():
        return ""
    lines = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            f = json.loads(line)
        except json.JSONDecodeError:
            continue
        if f.get("phase") != "survey":
            continue
        spk = f.get("speaker", "?")
        label = "COACH" if spk == "coach" else "USER"
        lines.append(f"{label}: {(f.get('text') or '').strip()}")
    return "\n".join(lines)
