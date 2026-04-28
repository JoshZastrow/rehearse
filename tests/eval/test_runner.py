"""End-to-end runner test on the noop eval + echo environment.

Exercises: example loading, subprocess executor, scoring, artifact writes
(run.json, results.jsonl, summary.md).
"""

from __future__ import annotations

import json
from pathlib import Path

from rehearse.eval.runner import RunConfig, execute_run


async def test_noop_run_produces_full_artifact_bundle(tmp_path: Path):
    config = RunConfig(
        eval_name="noop",
        environment="echo",
        concurrency=2,
        runs_root=tmp_path,
    )
    outcome = await execute_run(config)

    assert outcome.n_examples == 2
    assert outcome.n_ok == 2
    assert outcome.n_error == 0
    assert outcome.aggregate_scores["noop_score"] == 1.0

    run_json = json.loads((outcome.run_dir / "run.json").read_text())
    assert run_json["id"] == outcome.run_id
    assert run_json["pipeline_version"].startswith("noop@")
    assert len(run_json["example_ids"]) == 2

    lines = (outcome.run_dir / "results.jsonl").read_text().splitlines()
    assert len(lines) == 2
    for line in lines:
        row = json.loads(line)
        assert row["dimension"] == "noop_score"
        assert row["value"] == 1.0

    summary = (outcome.run_dir / "summary.md").read_text()
    assert "noop" in summary
    assert "ok=2" in summary
    assert "noop_score" in summary


async def test_unsupported_environment_rejected(tmp_path: Path):
    import pytest

    config = RunConfig(eval_name="noop", environment="raw-llm", runs_root=tmp_path)
    with pytest.raises(ValueError, match="does not support environment"):
        await execute_run(config)
