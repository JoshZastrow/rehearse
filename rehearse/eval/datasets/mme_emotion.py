"""MME-Emotion 10-clip dataset adapter.

The checked-in manifest is intentionally small. It lets the eval path, prompt
shape, and scorers be tested without vendoring the full MME-Emotion corpus.
Actual media files are expected at the manifest paths before running a real
audio-native environment.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rehearse.eval.protocols import BenchmarkExample

_DEFAULT_PATH = Path("evals/datasets/mme-emotion/v0-10clip/manifest.json")

_LABEL_SET = [
    "Anger",
    "Sadness",
    "Surprise",
    "Happiness",
    "Excited",
    "Fear",
    "Frustration",
    "Neutral",
    "Other",
]


def _build_prompt(label_set: list[str]) -> str:
    labels = ", ".join(label_set)
    return (
        "You are listening to a short clip of a person speaking. "
        "Classify the emotion they are expressing. Respond in JSON: "
        '{"label": <one of: '
        f"{labels}"
        '>, "reasoning": <2-4 sentences citing tone, pacing, and word choice>}'
    )


class MMEEmotionDataset:
    name = "mme-emotion"
    version = "v0-10clip"

    def __init__(self, data_path: Path | None = None) -> None:
        env = os.environ.get("MME_EMOTION_MANIFEST_PATH")
        self.data_path = Path(data_path or env or _DEFAULT_PATH)

    def load(self) -> Iterable[BenchmarkExample]:
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"MME-Emotion manifest not found at {self.data_path}. "
                "Set MME_EMOTION_MANIFEST_PATH or vendor the v0-10clip manifest."
            )

        data = json.loads(self.data_path.read_text())
        label_set = list(data.get("label_set") or _LABEL_SET)
        prompt = _build_prompt(label_set)
        root = self.data_path.parent

        for row in data["clips"]:
            media_path = _resolve_media_path(root, row)
            yield BenchmarkExample(
                id=row["id"],
                benchmark=self.name,
                payload={
                    "video_path": media_path,
                    "audio_max_s": int(row.get("audio_max_s", data.get("audio_max_s", 30))),
                    "prompt": prompt,
                    "label_set": label_set,
                },
                expected={"label": row["label"]},
                metadata={
                    "subset": row.get("subset", data.get("subset")),
                    "duration_s": row.get("duration_s"),
                    "speaker_id": row.get("speaker_id"),
                    "source_id": row.get("source_id"),
                },
            )


def _resolve_media_path(root: Path, row: dict[str, Any]) -> str:
    raw = row.get("video_path") or row.get("audio_path") or row["file"]
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    return str(root / path)
