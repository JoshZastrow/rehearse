"""`SurveyPresenceJudge` — did the survey actually happen, and was it coherent?

Reads `survey.json` from a rollout's artifacts directory and optionally the
SURVEY-phase rows from `transcript.jsonl`. Scores whether:
  - The survey ran at all (survey.json present, status complete/partial)
  - The caller's response was captured
  - The response was substantive (verbatim richness)
  - The exchange was coherent (LLM path only)

Default mode is deterministic (no API key required). Inject `anthropic_client`
to upgrade to Claude Haiku as judge, which also reads the transcript excerpt.

Spec: docs/specs/v2026-05-13-survey-agent.md §11
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.types import RubricDimension, RubricScore, SurveyRecord

_PROMPT_VERSION = "survey-presence-v1"

_SYSTEM = """You are scoring whether a post-call survey exchange was natural and coherent.

Read the survey question the coach asked and the caller's response. Score
`survey_response_quality` ∈ [0.0, 1.0]:

  1.0 — Response is specific, grounded, and references something real from the call.
  0.7 — Positive and genuine but vague ("yes it was helpful").
  0.4 — Bare yes/no with no elaboration.
  0.1 — No response captured or completely off-topic.
  0.0 — Survey did not run.

Respond with ONLY this JSON, nothing else:
{"survey_response_quality": {"score": <0.0-1.0>, "rationale": "<one sentence>"}}
"""


class SurveyPresenceJudge:
    """Score whether the survey phase ran and produced a coherent caller response."""

    name = "survey_presence_judge"
    dimension = RubricDimension.SURVEY_RESPONSE_QUALITY

    def __init__(self, *, anthropic_client: Any | None = None) -> None:
        self._client = anthropic_client

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

        if self._client is not None:
            transcript_excerpt = _load_survey_transcript(rollout.artifacts_dir)
            return [await self._llm_score(example, run_id, record, transcript_excerpt)]

        return [self._deterministic_score(example, run_id, record)]

    def _deterministic_score(
        self,
        example: BenchmarkExample,
        run_id: str,
        record: SurveyRecord,
    ) -> RubricScore:
        captured = [r for r in record.responses if r.captured]
        if not captured:
            return RubricScore(
                run_id=run_id,
                example_id=example.id,
                dimension=self.dimension,
                value=0.2,
                scorer="deterministic",
                rationale="Survey ran but no response was captured.",
                modality="text",
                judge_prompt_version=_PROMPT_VERSION,
            )

        verbatim = captured[0].verbatim or ""
        word_count = len(verbatim.split())
        if word_count >= 8:
            value, rationale = 0.9, f"Rich verbatim response ({word_count} words)."
        elif word_count >= 3:
            value, rationale = 0.7, f"Genuine but brief response ({word_count} words)."
        elif word_count >= 1:
            value, rationale = 0.5, "Bare yes/no captured without elaboration."
        else:
            value, rationale = 0.3, "Response captured but no verbatim text."

        return RubricScore(
            run_id=run_id,
            example_id=example.id,
            dimension=self.dimension,
            value=value,
            scorer="deterministic",
            rationale=rationale,
            modality="text",
            judge_prompt_version=_PROMPT_VERSION,
        )

    async def _llm_score(
        self,
        example: BenchmarkExample,
        run_id: str,
        record: SurveyRecord,
        transcript_excerpt: str,
    ) -> RubricScore:
        captured = [r for r in record.responses if r.captured]
        question = record.questions[0].text if record.questions else ""
        verbatim = captured[0].verbatim if captured else "(no response)"

        user_prompt = (
            f"Survey question asked: {question}\n"
            f"Caller's response: {verbatim}\n"
        )
        if transcript_excerpt:
            user_prompt += f"\nSurvey phase transcript:\n{transcript_excerpt}"

        try:
            resp = await self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=128,
                system=_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
            )
            payload = json.loads(resp.content[0].text)
            dim = payload["survey_response_quality"]
            value = max(0.0, min(1.0, float(dim["score"])))
            rationale = str(dim.get("rationale", ""))
        except Exception:
            return self._deterministic_score(example, run_id, record)

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
            scorer="deterministic",
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
