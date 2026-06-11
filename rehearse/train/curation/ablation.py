"""Ablation investigations: regression -> cohort bisection -> conviction.

Triggered after a training run. Compares the run's held-out loss against the
best prior accepted run in the ledger; on regression past the seed-noise band,
bisects the suspect cohort (sessions added since that run) with budget-matched
ablation runs until one session is isolated, or the budget is exhausted.

Guardrails: at most max_runs trainer dispatches per investigation (ledger
resume hits are free); an inconclusive investigation flags the corpus state so
the same manifest is never re-investigated (cooldown) until it changes.
Convictions are returned to the caller — tombstoning lives in curate.py.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel

from rehearse.train.curation.ledger import RunLedger, RunRecord
from rehearse.train.curation.trainer import Trainer

logger = logging.getLogger(__name__)


class Conviction(BaseModel):
    session_id: str
    delta: float
    noise_band: float
    run_ids: list[str]


class InvestigationResult(BaseModel):
    investigation_id: str
    outcome: Literal["no_regression", "convicted", "inconclusive", "skipped_cooldown"]
    convicted: list[Conviction] = []
    runs_used: int = 0


class AblationOrchestrator:
    def __init__(self, *, ledger: RunLedger, trainer: Trainer, max_runs: int = 6):
        self._ledger = ledger
        self._trainer = trainer
        self._max_runs = max_runs

    @staticmethod
    def manifest_hash(session_ids: Iterable[str]) -> str:
        return hashlib.sha1("\n".join(sorted(session_ids)).encode()).hexdigest()[:12]

    def _run_or_resume(
        self, current: RunRecord, excluded: list[str], seed: int, kind: str, investigation_id: str
    ) -> tuple[RunRecord, bool]:
        hit = self._ledger.find_run(
            manifest_hash=current.manifest_hash,
            excluded_ids=excluded,
            seed=seed,
            eval_set_version=current.eval_set_version,
        )
        if hit is not None:
            logger.info("Resume hit for excluded=%s seed=%d: %s", excluded, seed, hit.run_id)
            return hit, True
        loss = self._trainer.run(
            excluded_ids=list(excluded), seed=seed, max_steps=current.max_steps, kind=kind
        )
        record = RunRecord(
            run_id=f"{investigation_id}-{uuid.uuid4().hex[:6]}",
            kind=kind,  # type: ignore[arg-type]
            manifest_hash=current.manifest_hash,
            session_ids=current.session_ids,
            excluded_session_ids=list(excluded),
            seed=seed,
            max_steps=current.max_steps,
            heldout_loss=loss,
            eval_set_version=current.eval_set_version,
            investigation_id=investigation_id,
        )
        self._ledger.append(record)
        return record, False

    def _flag(self, current: RunRecord, cohort: list[str], investigation_id: str) -> None:
        self._ledger.append(
            RunRecord(
                run_id=f"{investigation_id}-flag",
                kind="ablation",
                manifest_hash=current.manifest_hash,
                session_ids=current.session_ids,
                seed=current.seed,
                max_steps=current.max_steps,
                heldout_loss=None,
                eval_set_version=current.eval_set_version,
                investigation_id=investigation_id,
                flagged_cohort=cohort or list(current.session_ids),
            )
        )

    def _noise_band(self, current: RunRecord, prior: RunRecord, budget: int) -> tuple[float, int]:
        """Band for the current manifest, falling back to the prior manifest's
        baselines (seed noise transfers across nearby corpora), then to one
        extra baseline seed on the current manifest, then to 0."""
        version = current.eval_set_version
        band = self._ledger.noise_band(current.manifest_hash, eval_set_version=version)
        if band is None:
            band = self._ledger.noise_band(prior.manifest_hash, eval_set_version=version)
        if band is None and budget > 0:
            _, cached = self._run_or_resume(
                current, [], current.seed + 1, "baseline", f"inv-{current.run_id}"
            )
            if not cached:
                budget -= 1
            band = self._ledger.noise_band(current.manifest_hash, eval_set_version=version)
        return band if band is not None else 0.0, budget

    async def investigate(self, current: RunRecord) -> InvestigationResult:
        investigation_id = f"inv-{current.run_id}"
        result = InvestigationResult(investigation_id=investigation_id, outcome="no_regression")

        if self._ledger.is_flagged(current.manifest_hash):
            result.outcome = "skipped_cooldown"
            return result
        prior = self._ledger.best_accepted(
            eval_set_version=current.eval_set_version, exclude_run_id=current.run_id
        )
        if prior is None or current.heldout_loss is None:
            return result

        budget = self._max_runs
        band, budget = self._noise_band(current, prior, budget)
        target = prior.heldout_loss
        assert target is not None

        if current.heldout_loss <= target + band:
            result.runs_used = self._max_runs - budget
            return result
        logger.info(
            "Regression: %.4f vs best accepted %.4f (band %.4f) — investigating",
            current.heldout_loss,
            target,
            band,
        )

        candidates = sorted(set(current.session_ids) - set(prior.session_ids))
        run_ids: list[str] = []

        while candidates and budget > 0:
            if len(candidates) == 1:
                record, cached = self._run_or_resume(
                    current, candidates, current.seed, "ablation", investigation_id
                )
                if not cached:
                    budget -= 1
                run_ids.append(record.run_id)
                if record.heldout_loss is not None and record.heldout_loss <= target + band:
                    result.convicted = [
                        Conviction(
                            session_id=candidates[0],
                            delta=current.heldout_loss - record.heldout_loss,
                            noise_band=band,
                            run_ids=run_ids,
                        )
                    ]
                    result.outcome = "convicted"
                    result.runs_used = self._max_runs - budget
                    return result
                candidates = []  # sole suspect doesn't explain the regression
                break

            half_a = candidates[: len(candidates) // 2]
            half_b = candidates[len(candidates) // 2 :]
            rec_a, cached = self._run_or_resume(
                current, half_a, current.seed, "ablation", investigation_id
            )
            if not cached:
                budget -= 1
            run_ids.append(rec_a.run_id)
            if rec_a.heldout_loss is not None and rec_a.heldout_loss <= target + band:
                candidates = half_a
                continue
            if budget <= 0:
                break
            rec_b, cached = self._run_or_resume(
                current, half_b, current.seed, "ablation", investigation_id
            )
            if not cached:
                budget -= 1
            run_ids.append(rec_b.run_id)
            if rec_b.heldout_loss is not None and rec_b.heldout_loss <= target + band:
                candidates = half_b
                continue
            break  # harm spans both halves — v1 gives up rather than guess

        self._flag(current, candidates, investigation_id)
        result.outcome = "inconclusive"
        result.runs_used = self._max_runs - budget
        return result
