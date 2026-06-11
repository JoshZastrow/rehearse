"""AblationOrchestrator: regression detection, bisection, guardrails, resume."""

from __future__ import annotations

from rehearse.train.curation.ablation import AblationOrchestrator
from rehearse.train.curation.ledger import RunLedger, RunRecord
from tests.curation.conftest import FakeTrainer

SESSIONS = ["s00", "s01", "s02", "s03", "s04", "s05"]


def _ledger_with_history(tmp_path, *, prior_sessions, band_losses=(3.0, 3.02)):
    """Ledger holding baseline runs of the prior (good) manifest."""
    ledger = RunLedger(tmp_path / "ledger.jsonl")
    mh_prior = AblationOrchestrator.manifest_hash(prior_sessions)
    for i, loss in enumerate(band_losses):
        ledger.append(
            RunRecord(
                run_id=f"b{i}",
                kind="baseline",
                manifest_hash=mh_prior,
                session_ids=list(prior_sessions),
                seed=i,
                max_steps=120,
                heldout_loss=loss,
                eval_set_version=1,
            )
        )
    return ledger


def _current(sessions, loss, seed=0):
    return RunRecord(
        run_id="cur",
        kind="training",
        manifest_hash=AblationOrchestrator.manifest_hash(sessions),
        session_ids=list(sessions),
        seed=seed,
        max_steps=120,
        heldout_loss=loss,
        eval_set_version=1,
    )


async def test_no_regression_within_band(tmp_path):
    ledger = _ledger_with_history(tmp_path, prior_sessions=SESSIONS[:4])
    trainer = FakeTrainer(session_ids=SESSIONS[:5])
    orch = AblationOrchestrator(ledger=ledger, trainer=trainer)

    result = await orch.investigate(_current(SESSIONS[:5], loss=3.015))

    assert result.outcome == "no_regression"
    assert trainer.calls == []


async def test_single_new_session_convicted(tmp_path):
    ledger = _ledger_with_history(tmp_path, prior_sessions=SESSIONS[:4])
    trainer = FakeTrainer(session_ids=SESSIONS[:5], harm={"s04": 0.5})
    orch = AblationOrchestrator(ledger=ledger, trainer=trainer)

    result = await orch.investigate(_current(SESSIONS[:5], loss=3.5))

    assert result.outcome == "convicted"
    [conviction] = result.convicted
    assert conviction.session_id == "s04"
    assert conviction.delta > 0.4


async def test_bisection_isolates_culprit_among_four_new(tmp_path):
    ledger = _ledger_with_history(tmp_path, prior_sessions=SESSIONS[:2])
    trainer = FakeTrainer(session_ids=SESSIONS, harm={"s03": 0.5})
    orch = AblationOrchestrator(ledger=ledger, trainer=trainer, max_runs=6)

    result = await orch.investigate(_current(SESSIONS, loss=3.5))

    assert result.outcome == "convicted"
    assert [c.session_id for c in result.convicted] == ["s03"]
    assert len(trainer.calls) <= 6


async def test_budget_exhaustion_flags_without_conviction(tmp_path):
    ledger = _ledger_with_history(tmp_path, prior_sessions=SESSIONS[:2])
    # Harm split across both bisection halves: v1 cannot isolate it.
    trainer = FakeTrainer(session_ids=SESSIONS, harm={"s02": 0.3, "s05": 0.3})
    orch = AblationOrchestrator(ledger=ledger, trainer=trainer, max_runs=2)

    result = await orch.investigate(_current(SESSIONS, loss=3.6))

    assert result.outcome == "inconclusive"
    assert result.convicted == []
    assert ledger.is_flagged(AblationOrchestrator.manifest_hash(SESSIONS))
    assert len(trainer.calls) == 2


async def test_cooldown_skips_previously_flagged_manifest(tmp_path):
    ledger = _ledger_with_history(tmp_path, prior_sessions=SESSIONS[:2])
    trainer = FakeTrainer(session_ids=SESSIONS, harm={"s02": 0.3, "s05": 0.3})
    orch = AblationOrchestrator(ledger=ledger, trainer=trainer, max_runs=2)
    await orch.investigate(_current(SESSIONS, loss=3.6))

    again = await orch.investigate(_current(SESSIONS, loss=3.6))

    assert again.outcome == "skipped_cooldown"


async def test_resume_reuses_ledgered_runs(tmp_path):
    ledger = _ledger_with_history(tmp_path, prior_sessions=SESSIONS[:4])
    trainer = FakeTrainer(session_ids=SESSIONS[:5], harm={"s04": 0.5})
    orch = AblationOrchestrator(ledger=ledger, trainer=trainer)
    first = await orch.investigate(_current(SESSIONS[:5], loss=3.5))
    assert first.outcome == "convicted"
    calls_after_first = len(trainer.calls)

    second = await orch.investigate(_current(SESSIONS[:5], loss=3.5))

    assert second.outcome == "convicted"
    assert len(trainer.calls) == calls_after_first  # all runs served from the ledger
