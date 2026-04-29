"""Simple rule-based scorers for eval runs.

This file holds the cheap, deterministic scoring helpers that do not need an
LLM judge. Right now it focuses on parsing model output and checking whether an
MME-Emotion prediction exactly matches the expected label.
"""

from __future__ import annotations

import json
import re
from typing import Any

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.types import RubricScore

_JSON_BLOCK = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_json_object_with_keys(text: str, expected_keys: list[str]) -> dict[str, Any] | None:
    """Find and return the first JSON object that contains the required keys."""
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
            return obj
    return None


class MMERecognitionScorer:
    """Score whether a predicted emotion label exactly matches the expected one."""

    name = "mme_recognition"
    dimension = "mme_recognition_accuracy"

    async def score(
        self,
        example: BenchmarkExample,
        rollout: RolloutResult,
        run_id: str,
    ) -> list[RubricScore]:
        """Return one score row for a single MME-Emotion example."""
        if rollout.status != "ok" or not rollout.payload:
            error = f": {rollout.error}" if rollout.error else ""
            return [self._zero(example, run_id, rationale=f"rollout {rollout.status}{error}")]

        expected_label = str(example.expected.get("label", "")).strip()
        if not expected_label:
            return [self._zero(example, run_id, rationale="example has no expected label")]

        output = rollout.payload.get("output") or ""
        parsed = parse_json_object_with_keys(str(output), ["label"])
        if parsed is None:
            return [
                self._zero(
                    example,
                    run_id,
                    rationale=f"could not parse label from output: {str(output)[:200]!r}",
                )
            ]

        predicted_label = str(parsed.get("label", "")).strip()
        value = 1.0 if predicted_label.lower() == expected_label.lower() else 0.0
        return [
            RubricScore(
                run_id=run_id,
                example_id=example.id,
                dimension=self.dimension,
                value=value,
                scorer="deterministic",
                rationale=f"predicted={predicted_label!r} expected={expected_label!r}",
            )
        ]

    @staticmethod
    def _zero(example: BenchmarkExample, run_id: str, rationale: str) -> RubricScore:
        """Return a zero-valued score row when scoring cannot proceed."""
        return RubricScore(
            run_id=run_id,
            example_id=example.id,
            dimension=MMERecognitionScorer.dimension,
            value=0.0,
            scorer="deterministic",
            rationale=rationale,
        )
