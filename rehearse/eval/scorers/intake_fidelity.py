"""`IntakeFidelityScorer` — compare intake.json fields against expected values.

Reads `intake.json` from `rollout.artifacts_dir`. Compares against optional
`expected.intake_relationship`, `expected.intake_stakes`, and
`expected.intake_user_goal` fields on the benchmark example.

Returns a score in [0, 1] equal to the fraction of present `expected.intake_*`
fields that match the actual `intake.json` value (case-insensitive substring).
Returns an empty list if no `expected.intake_*` fields are present on the row.
Returns a zero score with `flags=["intake_missing"]` if `intake.json` is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.types import RubricScore

_SCORER_VERSION = "intake-fidelity-v1"

# Map expected key suffix → intake.json field name
_FIELD_MAP: dict[str, str] = {
    "intake_relationship": "counterparty_relationship",
    "intake_stakes": "stakes",
    "intake_user_goal": "user_goal",
}


class IntakeFidelityScorer:
    name = "intake_fidelity"
    dimension = "intake_fidelity"

    async def score(
        self,
        example: BenchmarkExample,
        rollout: RolloutResult,
        run_id: str,
    ) -> list[RubricScore]:
        # Collect expected.intake_* fields from the example.
        expected_checks: dict[str, str] = {}
        for suffix, field in _FIELD_MAP.items():
            value = example.expected.get(suffix)
            if value is not None:
                expected_checks[field] = str(value)

        # No expected fields → no signal.
        if not expected_checks:
            return []

        # Missing intake.json → degraded score with flag.
        if rollout.artifacts_dir is None:
            return [self._flagged(example, run_id, "artifacts_dir is None")]

        intake_path = Path(rollout.artifacts_dir) / "intake.json"
        if not intake_path.exists():
            return [
                RubricScore(
                    run_id=run_id,
                    example_id=example.id,
                    dimension=self.dimension,
                    value=0.0,
                    scorer="deterministic",
                    rationale=(
                        "intake.json not found — RuntimeHost did not complete INTAKE phase. "
                        f"Check failures/{example.id}/."
                    ),
                    modality="text",
                    judge_prompt_version=_SCORER_VERSION,
                    flags=["intake_missing"],
                )
            ]

        try:
            intake = json.loads(intake_path.read_text())
        except Exception as exc:
            return [self._flagged(example, run_id, f"intake.json unreadable: {exc}")]

        matches = 0
        mismatches: list[str] = []
        for field, expected_val in expected_checks.items():
            actual_val = str(intake.get(field, "")).lower()
            if expected_val.lower() in actual_val:
                matches += 1
            else:
                mismatches.append(f"{field}: expected {expected_val!r}, got {intake.get(field)!r}")

        total = len(expected_checks)
        score = matches / total if total > 0 else 1.0
        rationale = f"{matches}/{total} intake fields match"
        if mismatches:
            rationale += "; mismatches: " + "; ".join(mismatches)

        return [
            RubricScore(
                run_id=run_id,
                example_id=example.id,
                dimension=self.dimension,
                value=score,
                scorer="deterministic",
                rationale=rationale,
                modality="text",
                judge_prompt_version=_SCORER_VERSION,
            )
        ]

    def _flagged(
        self, example: BenchmarkExample, run_id: str, rationale: str
    ) -> RubricScore:
        return RubricScore(
            run_id=run_id,
            example_id=example.id,
            dimension=self.dimension,
            value=0.0,
            scorer="deterministic",
            rationale=rationale,
            modality="text",
            judge_prompt_version=_SCORER_VERSION,
            flags=["intake_missing"],
        )
