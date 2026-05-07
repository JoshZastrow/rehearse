"""Dataset for `voice-rollout-judges` eval.

Loads synthetic coaching scenarios from a JSONL manifest. Each line is one
example with: `id`, `difficulty`, `scenario` (situation/goal/counterparty/
stakes/emotional_state), optional `expected`, optional `rubric_weights`.

The manifest path defaults to
`evals/datasets/voice-rollout-judges/v1/scenarios.jsonl`. Override with
`VOICE_ROLLOUT_SCENARIOS_PATH` for tuning runs against alternate sets.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

from rehearse.eval.protocols import BenchmarkExample


_DEFAULT_PATH = Path("evals/datasets/voice-rollout-judges/v1/scenarios.jsonl")


class VoiceRolloutJudgesDataset:
    name = "voice-rollout-judges"
    version = "v1"

    def __init__(self, path: Path | None = None) -> None:
        env_override = os.environ.get("VOICE_ROLLOUT_SCENARIOS_PATH")
        self._path = Path(env_override) if env_override else (path or _DEFAULT_PATH)

    def load(self) -> Iterable[BenchmarkExample]:
        if not self._path.exists():
            raise FileNotFoundError(f"scenarios manifest not found: {self._path}")
        examples: list[BenchmarkExample] = []
        for raw in self._path.read_text().splitlines():
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            row = json.loads(raw)
            payload: dict = {
                "scenario": row["scenario"],
                "max_turns": int(row.get("max_turns", 4)),
                "timeout_s": int(row.get("timeout_s", 90)),
                "silence_between_turns_s": float(row.get("silence_between_turns_s", 1.5)),
            }
            if "rubric_weights" in row:
                payload["rubric_weights"] = row["rubric_weights"]
            if "coach_description" in row:
                payload["coach_description"] = row["coach_description"]
            examples.append(
                BenchmarkExample(
                    id=row["id"],
                    benchmark=self.name,
                    payload=payload,
                    expected=row.get("expected") or {},
                    metadata={"difficulty": row.get("difficulty", "unknown")},
                )
            )
        return examples
