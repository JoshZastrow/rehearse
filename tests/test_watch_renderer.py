"""Tests for `rehearse-eval watch` — the live aggregate renderer.

The renderer's job is simple: read JSONL lines as they arrive, accumulate
per-dimension means, and stop when the sentinel appears. We test the
follow-and-aggregate behavior end-to-end against a synthetic stream.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from rehearse.eval.harness.watch import AggregateState, watch


def _row(example_id: str, dim: str, value: float) -> str:
    return json.dumps(
        {
            "run_id": "run-1",
            "example_id": example_id,
            "dimension": dim,
            "value": value,
            "scorer": "deterministic",
            "rationale": None,
            "modality": "text",
            "judge_prompt_version": None,
            "flags": [],
            "emitted_at": "2026-05-08T12:00:00Z",
        }
    )


def test_aggregate_state_accumulates_per_dimension() -> None:
    state = AggregateState()
    state.add(json.loads(_row("a", "content_quality", 0.8)))
    state.add(json.loads(_row("b", "content_quality", 0.6)))
    state.add(json.loads(_row("a", "affect_perception", 0.9)))

    assert state._values["content_quality"] == [0.8, 0.6]
    assert state._values["affect_perception"] == [0.9]


def test_aggregate_state_handles_weighted_reward_separately() -> None:
    state = AggregateState()
    state.add(json.loads(_row("a", "content_quality", 0.8)))
    state.add(json.loads(_row("a", "weighted_reward", 0.75)))

    table = state.render(Path("/tmp/run-x"))
    # Just confirm rendering doesn't crash and includes both dims.
    rendered = "\n".join(str(c) for col in table.columns for c in col._cells)
    assert "content_quality" in rendered
    assert "weighted_reward" in rendered


def test_watch_returns_1_for_missing_run_dir(tmp_path: Path) -> None:
    """Watching a nonexistent dir errors cleanly with exit code 1."""
    code = watch(tmp_path / "does-not-exist", poll_interval_s=0.01)
    assert code == 1


def test_watch_reads_existing_lines_and_exits_on_sentinel(tmp_path: Path) -> None:
    """Pre-populate scores.jsonl, write sentinel, watch reads everything and exits."""
    scores = tmp_path / "scores.jsonl"
    sentinel = tmp_path / "scores.jsonl.done"
    scores.write_text(
        _row("a", "content_quality", 0.8) + "\n"
        + _row("b", "content_quality", 0.6) + "\n"
    )
    sentinel.write_text("")

    code = watch(tmp_path, poll_interval_s=0.01)
    assert code == 0


def test_watch_picks_up_lines_appended_after_start(tmp_path: Path) -> None:
    """Start watch in background; append a line; confirm it terminates on sentinel."""
    scores = tmp_path / "scores.jsonl"
    sentinel = tmp_path / "scores.jsonl.done"
    # Start with an empty file so the watcher is in tail mode.
    scores.write_text("")

    result: dict[str, int | None] = {"code": None}

    def _run_watch() -> None:
        result["code"] = watch(tmp_path, poll_interval_s=0.01)

    t = threading.Thread(target=_run_watch, daemon=True)
    t.start()

    # Append a few lines, then signal done.
    time.sleep(0.05)
    with open(scores, "a") as fh:
        fh.write(_row("a", "content_quality", 0.8) + "\n")
        fh.write(_row("a", "weighted_reward", 0.75) + "\n")
    time.sleep(0.05)
    sentinel.write_text("")

    t.join(timeout=2.0)
    assert not t.is_alive(), "watch did not exit on sentinel"
    assert result["code"] == 0
