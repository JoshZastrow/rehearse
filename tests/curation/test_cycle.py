"""End-to-end training cycle: sweep -> train -> investigate -> tombstone -> SIA."""

from __future__ import annotations

import json

from rehearse.train.curation.cli import run_training_cycle
from rehearse.train.curation.criteria import load_criteria
from rehearse.train.curation.ledger import RunLedger
from rehearse.train.curation.reviewer import SessionReviewer
from tests.curation.conftest import (
    FakeJudgeBackend,
    FakeTrainer,
    FakeVolumeClient,
    approve_response,
    make_session_dir,
)


async def test_cycle_convicts_tombstones_and_updates_criteria(tmp_path):
    root = tmp_path / "sessions"
    for i in range(5):
        make_session_dir(root, f"s{i:02d}")
    reviewer = SessionReviewer(backend=FakeJudgeBackend(lambda s, u: approve_response()))
    feedback_backend = FakeJudgeBackend(
        lambda s, u: json.dumps({"criteria": "# v2", "improvement": "ablation evidence"})
    )
    ledger = RunLedger(tmp_path / "ledger.jsonl")
    volume = FakeVolumeClient()
    harm = {"s99": 0.5}
    kwargs = dict(
        sessions_root=root,
        criteria_dir=tmp_path / "criteria",
        out=tmp_path / "curated.jsonl",
        ledger=ledger,
        trainer_factory=lambda ids: FakeTrainer(session_ids=ids, harm=harm),
        volume_client=volume,
        reviewer=reviewer,
        feedback_backend=feedback_backend,
        max_steps=120,
    )

    first = await run_training_cycle(**kwargs)  # clean corpus: nothing to regress against
    assert first.investigation.outcome == "no_regression"
    assert load_criteria(tmp_path / "criteria")[0] == 1  # SIA not fired

    make_session_dir(root, "s99")  # harmful newcomer arrives
    second = await run_training_cycle(**kwargs)

    assert second.investigation.outcome == "convicted"
    assert [c.session_id for c in second.investigation.convicted] == ["s99"]
    assert "s99" not in (tmp_path / "curated.jsonl").read_text()
    assert json.loads((root / "s99" / "review.json").read_text())["decision"] == "rejected"
    assert "s99" not in volume.pushed["data/curated_sessions.jsonl"].decode()
    assert load_criteria(tmp_path / "criteria")[0] == 2  # SIA fired on eviction
