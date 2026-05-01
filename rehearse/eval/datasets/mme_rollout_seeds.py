"""MME-emotion rollout-seed dataset.

Reads the v0-rollout-seeds manifest. Each example pairs an MME-Emotion clip
(with its upstream emotion label) with a hand-authored rehearse-product
scenario. The dataset emits BenchmarkExamples that drive an LLM-vs-LLM
sandbox dialogue conditioned on the opening emotion.

Audio observation step is deferred per RLE3 spec; the opening emotion is
taken directly from the upstream MME label and folded into the customer
agent's emotional_state.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

from rehearse.eval.protocols import BenchmarkExample

_DEFAULT_PATH = Path("evals/datasets/mme-emotion/v0-rollout-seeds/manifest.json")


class MMERolloutSeedDataset:
    name = "mme-emotion-rollout-seeds"
    version = "v0"

    def __init__(self, data_path: Path | None = None) -> None:
        env = os.environ.get("MME_ROLLOUT_SEEDS_MANIFEST_PATH")
        self.data_path = Path(data_path or env or _DEFAULT_PATH)

    def load(self) -> Iterable[BenchmarkExample]:
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"MME rollout-seeds manifest not found at {self.data_path}. "
                "Set MME_ROLLOUT_SEEDS_MANIFEST_PATH or vendor the manifest."
            )

        data = json.loads(self.data_path.read_text())
        rubric_weights = dict(data.get("rubric_weights") or {})
        manifest_dir = self.data_path.parent

        for row in data["examples"]:
            scenario = dict(row.get("scenario") or {})
            clip = dict(row.get("clip") or {})
            rollout = dict(row.get("rollout") or {})
            opening_emotion = clip.get("opening_emotion") or "Neutral"

            scenario.setdefault(
                "emotional_state",
                f"{opening_emotion.lower()} — entering the conversation in this state",
            )

            clip_path = clip.get("path")
            resolved_clip_path: str | None = None
            if clip_path:
                p = Path(clip_path)
                resolved_clip_path = str(p if p.is_absolute() else manifest_dir / p)

            yield BenchmarkExample(
                id=row["id"],
                benchmark=self.name,
                payload={
                    "scenario": scenario,
                    "max_turns": int(rollout.get("max_turns", 6)),
                    "timeout_s": int(rollout.get("timeout_s", 120)),
                    "customer_agent": "llm",
                    "coach_agent": "llm",
                    "opening_emotion": opening_emotion,
                    "rubric_weights": rubric_weights,
                    **({"clip_path": resolved_clip_path} if resolved_clip_path else {}),
                },
                expected={
                    "opening_emotion": opening_emotion,
                    "min_runtime_turns": 2,
                },
                metadata={
                    "source_id": clip.get("source_id"),
                    "source_eval": "mme-emotion",
                },
            )
