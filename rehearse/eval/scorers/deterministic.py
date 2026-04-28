"""Deterministic scorers — no LLM judge.

Phase 2 ships the EQ-Bench correlation scorer. Phase 3 adds pacing,
groundedness, fault recall/precision against rehearse-seed sessions.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.types import RubricScore


def _pearson(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    den_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    den_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))
    if den_a == 0 or den_b == 0:
        return 0.0
    return num / (den_a * den_b)


_JSON_BLOCK = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_eq_ratings(text: str, expected_keys: list[str]) -> dict[str, float] | None:
    """Pull a JSON object of emotion ratings out of a model response.

    Tolerates leading/trailing prose. Returns None if no parseable object
    contains the expected keys.
    """
    candidates = _JSON_BLOCK.findall(text)
    candidates.insert(0, text)
    for chunk in candidates:
        try:
            obj = json.loads(chunk)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if all(k in obj for k in expected_keys):
            try:
                return {k: float(obj[k]) for k in expected_keys}
            except (TypeError, ValueError):
                continue
    return None


class EQBenchCorrelationScorer:
    """Pearson correlation between predicted and reference emotion intensities,
    rescaled to 0–100 (EQ-Bench convention: (corr+1)*50)."""

    name = "eq_bench_correlation"
    dimension = "eq_bench_score"

    async def score(
        self,
        example: BenchmarkExample,
        rollout: RolloutResult,
        run_id: str,
    ) -> list[RubricScore]:
        if rollout.status != "ok" or not rollout.payload:
            return [self._zero(example, run_id, rationale=f"rollout {rollout.status}")]

        expected: dict[str, Any] = example.expected.get("ratings", {})
        if not expected:
            return [self._zero(example, run_id, rationale="example has no expected ratings")]

        keys = list(expected.keys())
        output = rollout.payload.get("output") or ""
        predicted = parse_eq_ratings(output, keys)
        if predicted is None:
            return [
                self._zero(
                    example,
                    run_id,
                    rationale=f"could not parse ratings from output: {output[:200]!r}",
                )
            ]

        ref = [float(expected[k]) for k in keys]
        pred = [predicted[k] for k in keys]
        corr = _pearson(pred, ref)
        score_value = (corr + 1.0) * 50.0
        return [
            RubricScore(
                run_id=run_id,
                example_id=example.id,
                dimension=self.dimension,
                value=score_value,
                scorer="deterministic",
                rationale=f"pearson={corr:.3f} predicted={predicted} expected={expected}",
            )
        ]

    @staticmethod
    def _zero(example: BenchmarkExample, run_id: str, rationale: str) -> RubricScore:
        return RubricScore(
            run_id=run_id,
            example_id=example.id,
            dimension=EQBenchCorrelationScorer.dimension,
            value=0.0,
            scorer="deterministic",
            rationale=rationale,
        )
