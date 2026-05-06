"""`AggregateScorer` — pure aggregator producing `weighted_reward`.

Composes the per-dimension `RubricScore` rows that other scorers in the
plan have already produced, computes a weighted average against
configurable per-dimension weights, and writes a `judge.json` provenance
artifact alongside the rollout (when `artifacts_dir` is set).

Spec reference: v2026-05-06 roadmap §5 (Mini-spec 1) and v2026-05-05
spec §5.7. Stability scores (Spec 8) are *not* aggregated here — they
describe the policy across rollouts, not a single trajectory.

Note on shape: the standard `Scorer` protocol consumes one rollout. An
aggregator is structurally different — it takes the *outputs* of other
scorers as input. So this class exposes `aggregate(...)` rather than
`score(...)`. The runner is the place that wires it in: after collecting
per-rollout scores, call `aggregate.aggregate(example, rollout, scores)`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.types import RubricScore


class AggregateScorer:
    """Pure-function aggregator over per-dimension `RubricScore` rows."""

    name = "aggregate"
    dimension = "weighted_reward"

    def __init__(
        self,
        *,
        weights: dict[str, float],
        version: str = "aggregate-v1",
    ) -> None:
        if not weights:
            raise ValueError("AggregateScorer requires at least one weighted dimension")
        if any(w < 0 for w in weights.values()):
            raise ValueError(f"weights must be non-negative; got {weights!r}")
        self._weights = dict(weights)
        self.version = version

    async def aggregate(
        self,
        example: BenchmarkExample,
        rollout: RolloutResult,
        per_dim_scores: list[RubricScore],
        run_id: str,
    ) -> list[RubricScore]:
        """Emit one `weighted_reward` row.

        `per_dim_scores` is the list of per-dimension scores produced by
        other scorers in the plan. Per-example `rubric_weights` on the
        example payload beat the constructor defaults; missing dimensions
        renormalize the remaining weights and tag the result with a
        `partial_modality` flag.
        """

        weights = dict(
            example.payload.get("rubric_weights")
            if isinstance(example.payload.get("rubric_weights"), dict)
            else self._weights
        )

        flags: list[str] = []
        if not per_dim_scores:
            flags.append("no_inputs")
            return [self._row(run_id, example, value=0.0, flags=flags)]

        # Group by dimension; if multiple rows for the same dimension, take last.
        by_dim: dict[str, RubricScore] = {}
        for s in per_dim_scores:
            by_dim[str(s.dimension)] = s

        present = {d: w for d, w in weights.items() if d in by_dim}
        if not present:
            flags.append("no_inputs")
            return [self._row(run_id, example, value=0.0, flags=flags)]

        if len(present) < len(weights):
            flags.append("partial_modality")

        total = sum(present.values())
        renorm = {d: w / total for d, w in present.items()} if total > 0 else {}
        weighted = sum(renorm[d] * float(by_dim[d].value) for d in renorm)

        # Provenance: write judge.json next to the rollout artifacts.
        if rollout.artifacts_dir is not None:
            self._write_judge_json(
                Path(rollout.artifacts_dir),
                weighted_reward=weighted,
                weights=renorm,
                per_dim={d: float(by_dim[d].value) for d in renorm},
                judge_prompt_versions={
                    d: by_dim[d].judge_prompt_version
                    for d in renorm
                    if by_dim[d].judge_prompt_version
                },
                aggregator_version=self.version,
                flags=flags,
            )

        return [self._row(run_id, example, value=weighted, flags=flags)]

    def _row(
        self,
        run_id: str,
        example: BenchmarkExample,
        *,
        value: float,
        flags: list[str],
    ) -> RubricScore:
        return RubricScore(
            run_id=run_id,
            example_id=example.id,
            dimension="weighted_reward",
            value=float(value),
            scorer="deterministic",
            modality="aggregate",
            judge_prompt_version=self.version,
            flags=list(flags),
        )

    @staticmethod
    def _write_judge_json(out_dir: Path, **payload: Any) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "judge.json").write_text(json.dumps(payload, indent=2))
