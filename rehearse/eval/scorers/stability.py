"""`StabilityScorer` — meta-scorer over `repetitions > 1` rollouts.

Spec 8 of the v2026-05-06 eval-system roadmap. Diagnostic only — never
PR-gating. Groups per-rollout `RubricScore` rows by dimension and emits
one `stability.<dim>` row per dimension:

    stability = 1 - normalized_stddev(values_for_dimension)

where `normalized_stddev = stddev / |mean|` when the mean is non-zero,
otherwise the raw stddev (so a list of identical zeros still scores 1.0).
The result is clipped to `[0, 1]`.

Flags:
  - `stability_unmeasurable` — fewer than 2 rollouts present; value forced
    to 0.0.
  - `mixed_temperature` — sampling temperature varies across rollouts (read
    from `rollout.payload["temperature"]`). The variance may stem from the
    config rather than model nondeterminism, so the row should be excluded
    from training-data filters.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Sequence

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.types import RubricScore

_JUDGE_PROMPT_VERSION = "stability-v1"


class StabilityScorer:
    name = "stability"

    async def score_meta(
        self,
        example: BenchmarkExample,
        rollouts: Sequence[RolloutResult],
        per_rollout_scores: Sequence[Sequence[RubricScore]],
        run_id: str,
    ) -> list[RubricScore]:
        flags: list[str] = []
        if _temperatures_differ(rollouts):
            flags.append("mixed_temperature")

        if len(rollouts) < 2:
            dims = sorted({_dim(s) for scores in per_rollout_scores for s in scores})
            return [
                RubricScore(
                    run_id=run_id,
                    example_id=example.id,
                    dimension=f"stability.{dim}",
                    value=0.0,
                    scorer="deterministic",
                    modality="meta",
                    judge_prompt_version=_JUDGE_PROMPT_VERSION,
                    flags=[*flags, "stability_unmeasurable"],
                    rationale="fewer than 2 rollouts; stability requires repetitions > 1",
                )
                for dim in dims
            ]

        grouped: dict[str, list[float]] = defaultdict(list)
        for scores in per_rollout_scores:
            for s in scores:
                grouped[_dim(s)].append(s.value)

        rows: list[RubricScore] = []
        for dim in sorted(grouped):
            values = grouped[dim]
            if len(values) < 2:
                rows.append(
                    RubricScore(
                        run_id=run_id,
                        example_id=example.id,
                        dimension=f"stability.{dim}",
                        value=0.0,
                        scorer="deterministic",
                        modality="meta",
                        judge_prompt_version=_JUDGE_PROMPT_VERSION,
                        flags=[*flags, "stability_unmeasurable"],
                        rationale=f"only one observation of {dim}",
                    )
                )
                continue
            stability = _stability(values)
            rows.append(
                RubricScore(
                    run_id=run_id,
                    example_id=example.id,
                    dimension=f"stability.{dim}",
                    value=stability,
                    scorer="deterministic",
                    modality="meta",
                    judge_prompt_version=_JUDGE_PROMPT_VERSION,
                    flags=list(flags),
                    rationale=(
                        f"n={len(values)} mean={statistics.fmean(values):.3f} "
                        f"stddev={statistics.pstdev(values):.3f}"
                    ),
                )
            )
        return rows


def _dim(score: RubricScore) -> str:
    return score.dimension if isinstance(score.dimension, str) else score.dimension.value


def _stability(values: Sequence[float]) -> float:
    mean = statistics.fmean(values)
    stddev = statistics.pstdev(values)
    normalized = stddev / abs(mean) if mean != 0 else stddev
    return max(0.0, min(1.0, 1.0 - normalized))


def _temperatures_differ(rollouts: Sequence[RolloutResult]) -> bool:
    temps = []
    for r in rollouts:
        if r.payload and "temperature" in r.payload:
            temps.append(r.payload["temperature"])
    if len(temps) < 2:
        return False
    return len(set(temps)) > 1
