"""`ContentJudgeScorer` — text-side content quality via DeepEval G-Eval.

Spec 1 of the v2026-05-06 roadmap. The scorer is a thin orchestrator:

  1. Build a G-Eval metric (DeepEval) with criteria narrowed to *what
     was said* — content, safety, trajectory direction. Audio-aware
     dimensions (`affect_perception`, `delivery_quality`) live in their
     own scorers added by Spec 2.
  2. Wrap that metric via `MetricToScorer` so it satisfies the
     `Scorer` protocol and runs through the existing eval runner.
  3. Stamp `modality="text"` and `judge_prompt_version` on every
     emitted `RubricScore` so the data card can trace which prompt
     produced which scores.

The default G-Eval criteria are the v1 anchors from Appendix A of the
v2026-05-05 spec, narrowed for content (no delivery/prosody language).
A version bump in `prompt_version` should accompany any criteria edit
so old and new scores cannot mix in training data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rehearse.eval.deepeval_adapter import MetricToScorer
from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.types import RubricScore

if TYPE_CHECKING:
    from deepeval.metrics import BaseMetric


_DEFAULT_PROMPT_VERSION = "content-quality-v1"


_DEFAULT_CRITERIA = (
    "Evaluate whether the coach's responses moved the user toward clearer, "
    "calmer, actionable phrasing for the real conversation they were "
    "rehearsing for. Consider only what was said — content, safety, and "
    "trajectory direction. Do not judge tone, prosody, or delivery; that "
    "is scored elsewhere.\n\n"
    "Score on a 0.0 to 1.0 continuous scale:\n"
    "  • 1.0 — User ends with clearer, calmer, actionable phrasing they "
    "could actually say.\n"
    "  • 0.5 — Some useful coaching, but generic or uneven across turns.\n"
    "  • 0.0 — Rambling, unsafe, invalidating, or not useful for the scenario."
)


def _build_default_geval(prompt_version: str) -> BaseMetric:
    """Construct the default G-Eval metric for content quality.

    Uses DeepEval's `AnthropicModel` so the default judge is Claude (Sonnet)
    rather than OpenAI's GPT. The codebase standardizes on Anthropic for text
    judges; passing OpenAI here would force every CI/dev environment to set
    `OPENAI_API_KEY` for what is otherwise an Anthropic-only stack.

    Imported lazily so the module loads even when DeepEval isn't fully
    initialized (e.g. during test collection without API keys).
    """
    from deepeval.metrics import GEval
    from deepeval.models import AnthropicModel
    from deepeval.test_case import MultiTurnParams

    return GEval(
        name=f"ContentQuality.{prompt_version}",
        criteria=_DEFAULT_CRITERIA,
        evaluation_params=[MultiTurnParams.CONTENT, MultiTurnParams.ROLE],
        threshold=0.5,
        model=AnthropicModel(model="claude-sonnet-4-5"),
    )


class ContentJudgeScorer:
    """Score `content_quality` ∈ [0, 1] via a DeepEval G-Eval metric.

    Parameters
    ----------
    metric:
        Optional pre-built DeepEval metric. If omitted, a default G-Eval
        metric is constructed using `_DEFAULT_CRITERIA`. Pass a stub for
        unit tests; pass a custom GEval for prompt-iteration experiments.
    prompt_version:
        Stamped on every emitted RubricScore. Bump it whenever the
        criteria text or the underlying metric changes.
    """

    name = "content_judge"
    dimension = "content_quality"

    def __init__(
        self,
        *,
        metric: BaseMetric | None = None,
        prompt_version: str = _DEFAULT_PROMPT_VERSION,
    ) -> None:
        self.metric: Any = metric or _build_default_geval(prompt_version)
        self.prompt_version = prompt_version
        self._adapter = MetricToScorer(
            self.metric,
            dimension=self.dimension,
            name=self.name,
            test_case_kind="conversational",
        )

    async def score(
        self,
        example: BenchmarkExample,
        rollout: RolloutResult,
        run_id: str,
    ) -> list[RubricScore]:
        rows = await self._adapter.score(example, rollout, run_id)
        # Stamp modality and prompt version on every emitted row.
        return [
            row.model_copy(
                update={
                    "modality": "text",
                    "judge_prompt_version": self.prompt_version,
                }
            )
            for row in rows
        ]
