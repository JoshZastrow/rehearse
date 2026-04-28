"""EQ-Bench adapter.

Source: https://github.com/EQ-bench/EQ-Bench

The vendored sample at evals/benchmarks/eq-bench/sample/questions.json is a
stand-in with the same shape (dialogue + four reference emotion intensities).
To run against the real set, drop the upstream questions JSON at
evals/benchmarks/eq-bench/{commit_sha}/questions.json and pass
EQ_BENCH_DATA_PATH or set the EQBenchBenchmark.data_path attribute.

Why only `raw-llm`: EQ-Bench is single-turn text rating. Forcing it through
target=full would be type-confusion. Its job here is a model-level baseline
on the synthesis slot, not a product-quality measure.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

from rehearse.eval.protocols import BenchmarkExample, Scorer
from rehearse.eval.scorers.deterministic import EQBenchCorrelationScorer

_DEFAULT_PATH = Path("evals/benchmarks/eq-bench/sample/questions.json")


def _build_prompt(dialogue: str, character: str, emotions: list[str]) -> str:
    emotion_lines = "\n".join(f"- {e}" for e in emotions)
    keys_json = ", ".join(f'"{e}": <0-10>' for e in emotions)
    return (
        "Read this dialogue carefully, then rate the intensity of each "
        f"emotion experienced by {character}.\n\n"
        f"Dialogue:\n{dialogue}\n\n"
        "For each emotion, give an integer rating from 0 (not at all) to 10 "
        "(extremely intense):\n"
        f"{emotion_lines}\n\n"
        "Respond with ONLY a JSON object on a single line, no prose, no "
        "code fences:\n"
        f"{{{keys_json}}}\n"
    )


class EQBenchBenchmark:
    name = "eq-bench"
    version = "sample-v0"
    supported_targets = frozenset({"raw-llm"})
    preferred_target = "raw-llm"

    def __init__(self, data_path: Path | None = None) -> None:
        env = os.environ.get("EQ_BENCH_DATA_PATH")
        self.data_path = Path(data_path or env or _DEFAULT_PATH)

    def load(self) -> Iterable[BenchmarkExample]:
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"EQ-Bench data not found at {self.data_path}. "
                "Vendor the upstream questions.json or keep the sample fixture."
            )
        data = json.loads(self.data_path.read_text())
        meta_version = data.get("_meta", {}).get("vendored_commit", "unknown")
        for q in data["questions"]:
            yield BenchmarkExample(
                id=q["id"],
                benchmark=self.name,
                payload={
                    "prompt": _build_prompt(q["dialogue"], q["character"], q["emotions"]),
                    "max_tokens": 200,
                    "temperature": 0.0,
                },
                expected={"ratings": q["reference_ratings"]},
                metadata={
                    "character": q["character"],
                    "emotions": q["emotions"],
                    "vendored_commit": meta_version,
                },
            )

    def scoring_plan(self) -> list[Scorer]:
        return [EQBenchCorrelationScorer()]

    def rollout_timeout_s(self) -> int:
        return 60
