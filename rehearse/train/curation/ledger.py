"""Run ledger: the durable record every ablation decision is computed from.

JSONL, one RunRecord per line. Lives locally (and mirrored to the training
volume at data/run_ledger.jsonl by the cycle composition). Comparisons are
only valid between records with the same eval_set_version — changing the
held-out split bumps the version and resets the baseline requirement.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class RunRecord(BaseModel):
    run_id: str
    kind: Literal["baseline", "training", "ablation"]
    manifest_hash: str
    session_ids: list[str]
    excluded_session_ids: list[str] = []
    seed: int
    max_steps: int
    heldout_loss: float | None = None
    eval_set_version: int = 1
    status: Literal["completed", "failed"] = "completed"
    investigation_id: str | None = None
    flagged_cohort: list[str] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RunLedger:
    def __init__(self, path: Path):
        self._path = path

    def append(self, record: RunRecord) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a") as fh:
            fh.write(record.model_dump_json() + "\n")

    def records(self) -> list[RunRecord]:
        if not self._path.exists():
            return []
        return [
            RunRecord.model_validate_json(line)
            for line in self._path.read_text().splitlines()
            if line.strip()
        ]

    def _comparable(self, eval_set_version: int) -> list[RunRecord]:
        return [
            r
            for r in self.records()
            if r.status == "completed"
            and r.heldout_loss is not None
            and r.eval_set_version == eval_set_version
        ]

    def noise_band(self, manifest_hash: str, *, eval_set_version: int) -> float | None:
        """Seed-to-seed spread over full-corpus runs of one manifest; None if <2 samples."""
        losses = [
            r.heldout_loss
            for r in self._comparable(eval_set_version)
            if r.manifest_hash == manifest_hash and not r.excluded_session_ids
        ]
        if len(losses) < 2:
            return None
        return max(losses) - min(losses)

    def best_accepted(
        self, *, eval_set_version: int, exclude_run_id: str | None = None
    ) -> RunRecord | None:
        """Lowest-loss completed full-corpus run; the reference ablations must recover to."""
        candidates = [
            r
            for r in self._comparable(eval_set_version)
            if r.kind in ("baseline", "training")
            and not r.excluded_session_ids
            and r.run_id != exclude_run_id
        ]
        return min(candidates, key=lambda r: r.heldout_loss, default=None)

    def find_run(
        self, *, manifest_hash: str, excluded_ids: list[str], seed: int, eval_set_version: int
    ) -> RunRecord | None:
        """Resume lookup: a completed run with identical configuration, if any."""
        for r in self._comparable(eval_set_version):
            if (
                r.manifest_hash == manifest_hash
                and r.seed == seed
                and sorted(r.excluded_session_ids) == sorted(excluded_ids)
            ):
                return r
        return None

    def is_flagged(self, manifest_hash: str) -> bool:
        """Cooldown: an inconclusive investigation already flagged this corpus state."""
        return any(r.manifest_hash == manifest_hash and r.flagged_cohort for r in self.records())
