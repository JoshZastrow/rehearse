"""RunLedger: JSONL run records, noise band, best accepted run, resume lookup."""

from __future__ import annotations

from rehearse.train.curation.ledger import RunLedger, RunRecord


def _rec(
    run_id,
    *,
    kind="training",
    loss=3.0,
    seed=0,
    mh="m1",
    excluded=(),
    sessions=("a", "b"),
    status="completed",
    investigation_id=None,
    flagged=(),
):
    return RunRecord(
        run_id=run_id,
        kind=kind,
        manifest_hash=mh,
        session_ids=list(sessions),
        excluded_session_ids=list(excluded),
        seed=seed,
        max_steps=120,
        heldout_loss=loss,
        eval_set_version=1,
        status=status,
        investigation_id=investigation_id,
        flagged_cohort=list(flagged),
    )


def test_append_and_read_round_trip(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.jsonl")
    ledger.append(_rec("r1"))
    ledger.append(_rec("r2", loss=2.9, seed=1))
    assert [r.run_id for r in ledger.records()] == ["r1", "r2"]
    assert RunLedger(tmp_path / "ledger.jsonl").records()[1].heldout_loss == 2.9


def test_noise_band_needs_two_unexcluded_runs_same_manifest(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.jsonl")
    ledger.append(_rec("r1", kind="baseline", loss=3.00))
    assert ledger.noise_band("m1", eval_set_version=1) is None
    ledger.append(_rec("r2", kind="baseline", loss=3.04, seed=1))
    ledger.append(_rec("r3", kind="ablation", loss=2.0, excluded=("a",)))  # excluded: no count
    assert abs(ledger.noise_band("m1", eval_set_version=1) - 0.04) < 1e-9


def test_best_accepted_ignores_failed_ablation_and_excluded_run(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.jsonl")
    ledger.append(_rec("r1", loss=3.2))
    ledger.append(_rec("r2", loss=2.8, mh="m2", sessions=("a",)))
    ledger.append(_rec("r3", loss=1.0, status="failed"))
    ledger.append(_rec("r4", loss=1.1, kind="ablation", excluded=("a",)))
    best = ledger.best_accepted(eval_set_version=1, exclude_run_id="r1")
    assert best is not None and best.run_id == "r2"


def test_find_run_for_resume(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.jsonl")
    ledger.append(_rec("r1", kind="ablation", excluded=("a", "b"), loss=2.5))
    hit = ledger.find_run(manifest_hash="m1", excluded_ids=["b", "a"], seed=0, eval_set_version=1)
    assert hit is not None and hit.run_id == "r1"
    miss = ledger.find_run(manifest_hash="m1", excluded_ids=["c"], seed=0, eval_set_version=1)
    assert miss is None


def test_flagged_cohort_marks_cooldown(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.jsonl")
    assert not ledger.is_flagged("m1")
    ledger.append(_rec("r1", kind="ablation", investigation_id="i1", flagged=("a", "b"), loss=None))
    assert ledger.is_flagged("m1")
