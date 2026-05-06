"""DeepEval adapter layer.

Bridges our voice-shaped eval harness (transcript artifacts, `Scorer`
protocol, `RolloutResult`) to the DeepEval framework. See
`docs/specs/v2026-05-06-eval-system-roadmap.md` §4 (Mini-spec 0).

What goes through this adapter:

  - `to_conversational_test_case(example, rollout)` and
    `to_llm_test_case(example, rollout)` — convert coaching trajectories
    into DeepEval test cases.
  - `MetricToScorer(metric, dimension)` — wraps a DeepEval `BaseMetric`
    so it satisfies our `Scorer` protocol and can run inside the
    existing `runner.py` execution path.
  - `ScorerToMetric(scorer, ...)` — wraps a custom `Scorer` so it
    satisfies the DeepEval `BaseMetric` interface and can run inside
    `evaluate()` / `assert_test()`.

Audio, timing, and meta scorers stay on the rehearse `Scorer` /
`MetaScorer` protocol; this adapter does *not* try to push them through
DeepEval's text-shaped metric API.
"""

from rehearse.eval.deepeval_adapter.metric_to_scorer import MetricToScorer
from rehearse.eval.deepeval_adapter.scorer_to_metric import ScorerToMetric
from rehearse.eval.deepeval_adapter.test_case import (
    to_conversational_test_case,
    to_llm_test_case,
)

__all__ = [
    "MetricToScorer",
    "ScorerToMetric",
    "to_conversational_test_case",
    "to_llm_test_case",
]
