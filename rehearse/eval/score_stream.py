"""Append-only `scores.jsonl` writer for live score streaming.

Each `RubricScore` produced during a run is appended as a single JSON line
the moment a scorer returns it. A separate `rehearse-eval watch <run_dir>`
process tails the file and re-renders an aggregate. Decoupled — runner
writes, watcher reads, JSONL is the protocol.

Schema (one line per score):

    {
      "run_id": "...", "example_id": "...", "dimension": "content_quality",
      "value": 0.73, "scorer": "llm_judge", "rationale": "...",
      "modality": "text", "judge_prompt_version": "strict-content-v1",
      "flags": [], "emitted_at": "2026-05-08T12:34:56Z"
    }

When the run finishes, an empty `scores.jsonl.done` sentinel is written.
The watcher exits cleanly on its appearance.

Writes are line-buffered: each `publish(...)` appends one line and flushes,
so a watcher tailing the file sees scores within ~ms of them being emitted.
"""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import IO

from rehearse.types import RubricScore


class ScoreStreamWriter(AbstractContextManager["ScoreStreamWriter"]):
    """Thread-safe line-buffered append writer for `scores.jsonl`."""

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir
        self._path = run_dir / "scores.jsonl"
        self._sentinel = run_dir / "scores.jsonl.done"
        self._lock = Lock()
        self._fh: IO[str] | None = None

    def open(self) -> "ScoreStreamWriter":
        self._run_dir.mkdir(parents=True, exist_ok=True)
        # Line-buffered ('1') so each append is flushed to disk on newline.
        self._fh = open(self._path, "a", buffering=1, encoding="utf-8")
        return self

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        # Always write the sentinel, even if zero scores were emitted.
        self._sentinel.write_text("")

    def publish(self, score: RubricScore) -> None:
        if self._fh is None:
            raise RuntimeError("ScoreStreamWriter not opened")
        line = score.model_dump(mode="json")
        line["emitted_at"] = datetime.now(tz=UTC).isoformat(timespec="seconds")
        with self._lock:
            self._fh.write(json.dumps(line) + "\n")

    @property
    def path(self) -> Path:
        return self._path

    @property
    def sentinel_path(self) -> Path:
        return self._sentinel

    def __enter__(self) -> "ScoreStreamWriter":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
