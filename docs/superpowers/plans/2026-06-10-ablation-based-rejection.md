# Ablation-Based Rejection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After each training run, detect held-out-loss regressions, automatically isolate harmful sessions via budget-matched ablation bisection on Modal, tombstone convictions from the curated manifest, and feed verdicts to the SIA criteria loop.

**Architecture:** Client-side orchestrator in `rehearse/train/curation/` (spec: `docs/superpowers/specs/2026-06-10-ablation-based-rejection-design.md`). A `RunLedger` (JSONL) records every run; an `AblationOrchestrator` compares runs against a seed-noise band and bisects suspect cohorts through a `Trainer` protocol (fake in tests, Modal-backed live). Tombstoning rewrites the curated manifest and flips `review.json`. All new behavior is hermetically testable.

**Tech Stack:** Python 3.13, pydantic v2, pytest (asyncio_mode=auto), Modal SDK, existing curation package (reviewer/criteria/curate/feedback/volume).

**Conventions:** venv at `.venv/`; run tests as `.venv/bin/python -m pytest tests/curation/ -q`. Worktree: `~/Github/rehearse/.claude/worktrees/session-curation`, branch `worktree-session-curation`. Never edit `train/**` (user's active workspace). Commit after each task.

---

## File structure

- Create: `rehearse/train/curation/ledger.py` — `RunRecord`, `RunLedger` (append/read, noise band, best accepted run, resume lookup, cooldown flag)
- Create: `rehearse/train/curation/trainer.py` — `Trainer` protocol + `ModalTrainer` live adapter
- Create: `rehearse/train/curation/ablation.py` — `Conviction`, `InvestigationResult`, `AblationOrchestrator`
- Modify: `rehearse/train/curation/types.py` — add `split`, `ablation` fields to `SessionReview`
- Modify: `rehearse/train/curation/curate.py` — held-out split assignment, heldout manifest, `tombstone_sessions`
- Create: `rehearse/train/curation/cli.py` — `sweep` / `train` / `ablate` commands composing the cycle
- Test: `tests/curation/test_ledger.py`, `tests/curation/test_split.py`, `tests/curation/test_ablation.py`, `tests/curation/test_tombstone.py`, `tests/curation/test_cycle.py`; `FakeTrainer` added to `tests/curation/conftest.py`

---

### Task 1: RunLedger

**Files:** Create `rehearse/train/curation/ledger.py`; Test `tests/curation/test_ledger.py`

- [ ] **Step 1: Write failing tests**

```python
"""RunLedger: JSONL run records, noise band, best accepted run, resume lookup."""
from pathlib import Path
from rehearse.train.curation.ledger import RunLedger, RunRecord


def _rec(run_id, *, kind="training", loss=3.0, seed=0, mh="m1", excluded=(), sessions=("a", "b"),
         status="completed", investigation_id=None, flagged=()):
    return RunRecord(
        run_id=run_id, kind=kind, manifest_hash=mh, session_ids=list(sessions),
        excluded_session_ids=list(excluded), seed=seed, max_steps=120, heldout_loss=loss,
        eval_set_version=1, status=status, investigation_id=investigation_id,
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
    ledger.append(_rec("r3", kind="ablation", loss=2.0, excluded=("a",)))  # excluded runs don't count
    assert abs(ledger.noise_band("m1", eval_set_version=1) - 0.04) < 1e-9


def test_best_accepted_ignores_failed_ablation_and_other_eval_versions(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.jsonl")
    ledger.append(_rec("r1", loss=3.2))
    ledger.append(_rec("r2", loss=2.8, mh="m2", sessions=("a",)))
    ledger.append(_rec("r3", loss=1.0, status="failed"))
    ledger.append(_rec("r4", loss=1.1, kind="ablation", excluded=("a",)))
    best = ledger.best_accepted(eval_set_version=1, exclude_run_id="r1")
    assert best.run_id == "r2"


def test_find_run_for_resume(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.jsonl")
    ledger.append(_rec("r1", kind="ablation", excluded=("a", "b"), loss=2.5))
    hit = ledger.find_run(manifest_hash="m1", excluded_ids=["b", "a"], seed=0, eval_set_version=1)
    assert hit is not None and hit.run_id == "r1"
    assert ledger.find_run(manifest_hash="m1", excluded_ids=["c"], seed=0, eval_set_version=1) is None


def test_flagged_cohort_marks_cooldown(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.jsonl")
    assert not ledger.is_flagged("m1")
    ledger.append(_rec("r1", kind="ablation", investigation_id="i1", flagged=("a", "b")))
    assert ledger.is_flagged("m1")
```

- [ ] **Step 2:** Run `.venv/bin/python -m pytest tests/curation/test_ledger.py -q` — expect ModuleNotFoundError.
- [ ] **Step 3: Implement `ledger.py`**

```python
"""Run ledger: the durable record every ablation decision is computed from."""
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
        return [RunRecord.model_validate_json(line)
                for line in self._path.read_text().splitlines() if line.strip()]

    def _comparable(self, eval_set_version: int):
        return [r for r in self.records()
                if r.status == "completed" and r.heldout_loss is not None
                and r.eval_set_version == eval_set_version]

    def noise_band(self, manifest_hash: str, *, eval_set_version: int) -> float | None:
        losses = [r.heldout_loss for r in self._comparable(eval_set_version)
                  if r.manifest_hash == manifest_hash and not r.excluded_session_ids]
        if len(losses) < 2:
            return None
        return max(losses) - min(losses)

    def best_accepted(self, *, eval_set_version: int, exclude_run_id: str | None = None) -> RunRecord | None:
        candidates = [r for r in self._comparable(eval_set_version)
                      if r.kind in ("baseline", "training") and not r.excluded_session_ids
                      and r.run_id != exclude_run_id]
        return min(candidates, key=lambda r: r.heldout_loss, default=None)

    def find_run(self, *, manifest_hash: str, excluded_ids: list[str], seed: int,
                 eval_set_version: int) -> RunRecord | None:
        for r in self._comparable(eval_set_version):
            if (r.manifest_hash == manifest_hash and r.seed == seed
                    and sorted(r.excluded_session_ids) == sorted(excluded_ids)):
                return r
        return None

    def is_flagged(self, manifest_hash: str) -> bool:
        return any(r.manifest_hash == manifest_hash and r.flagged_cohort for r in self.records())
```

- [ ] **Step 4:** Run tests — expect 5 passed.
- [ ] **Step 5:** Commit: `feat: run ledger for ablation investigations`

### Task 2: Held-out split assignment

**Files:** Modify `rehearse/train/curation/types.py` (add `split`, `ablation` fields), `rehearse/train/curation/curate.py`; Test `tests/curation/test_split.py`

- [ ] **Step 1: Write failing tests**

```python
"""Held-out split: sticky, deterministic, ~20% min 2 once corpus >= 4 approved."""
import json
from tests.curation.conftest import FakeJudgeBackend, FakeVolumeClient, approve_response, make_session_dir
from rehearse.train.curation.curate import curate_sessions
from rehearse.train.curation.reviewer import SessionReviewer


async def _curate(tmp_path, n, volume=None):
    root = tmp_path / "sessions"
    for i in range(n):
        make_session_dir(root, f"s{i:02d}")
    return await curate_sessions(
        root, criteria_dir=tmp_path / "criteria", out=tmp_path / "curated.jsonl",
        reviewer=SessionReviewer(backend=FakeJudgeBackend(lambda s, u: approve_response())),
        volume_client=volume,
    )


async def test_small_corpus_has_no_heldout(tmp_path):
    result = await _curate(tmp_path, 3)
    assert all(r.split == "train" for r in result.approved)


async def test_split_is_two_heldout_at_six_and_sticky(tmp_path):
    result = await _curate(tmp_path, 6)
    heldout = sorted(r.session_id for r in result.approved if r.split == "heldout")
    assert len(heldout) == 2
    # Re-running curation (cached reviews) keeps the same assignment.
    again = await _curate(tmp_path, 6)
    assert sorted(r.session_id for r in again.approved if r.split == "heldout") == heldout
    # Stored in review.json so it survives manifest rebuilds.
    saved = json.loads((tmp_path / "sessions" / heldout[0] / "review.json").read_text())
    assert saved["split"] == "heldout"


async def test_manifests_separate_train_and_heldout(tmp_path):
    volume = FakeVolumeClient()
    result = await _curate(tmp_path, 6, volume=volume)
    train_ids = {json.loads(line)["path"].split("/")[-2]
                 for line in (tmp_path / "curated.jsonl").read_text().splitlines()}
    heldout_ids = {r.session_id for r in result.approved if r.split == "heldout"}
    assert train_ids.isdisjoint(heldout_ids) and len(train_ids) == 4
    remote_heldout = volume.pushed["data/heldout_sessions.jsonl"].decode()
    assert all(sid in remote_heldout for sid in heldout_ids)
```

- [ ] **Step 2:** Run — expect AttributeError (`SessionReview` has no `split`) / assertion failures.
- [ ] **Step 3: Implement.** In `types.py` add to `SessionReview`:

```python
    split: Literal["train", "heldout"] | None = None
    ablation: dict | None = None
```

In `curate.py`: after the review loop, assign splits over approved reviews, persist into each `review.json`, build the train manifest from `split == "train"` only, and emit `data/heldout_sessions.jsonl`:

```python
import hashlib

def _heldout_target(n_approved: int) -> int:
    return max(2, round(0.2 * n_approved)) if n_approved >= 4 else 0

def _assign_splits(approved: list[SessionReview], sessions_root: Path) -> None:
    target = _heldout_target(len(approved))
    held = [r for r in approved if r.split == "heldout"]
    unassigned = [r for r in approved if r.split is None]
    unassigned.sort(key=lambda r: hashlib.sha1(r.session_id.encode()).hexdigest())
    for review in unassigned:
        review.split = "heldout" if len(held) < target else "train"
        if review.split == "heldout":
            held.append(review)
    for review in approved:
        if review.split is None:
            review.split = "train"
        path = _session_dir(sessions_root, review.session_id) / "review.json"
        path.write_text(review.model_dump_json(indent=2))
```

(`_session_dir` resolves the reviewed dir; in `curate_sessions` keep a `{session_id: Path}` map from the scan loop.) Existing reviews loaded from `review.json` carry their stored `split` (stickiness). Heldout entries go to a second manifest list pushed as `data/heldout_sessions.jsonl` with the same `/data/data/sessions/<id>/audio_stereo.wav` path rewriting; heldout session files are pushed like train ones. Update the existing `test_curate.py` expectations only if they break (2-session corpora stay all-train, so they shouldn't).

- [ ] **Step 4:** Run full curation suite — all pass.
- [ ] **Step 5:** Commit: `feat: sticky held-out split in curation`

### Task 3: FakeTrainer + Trainer protocol

**Files:** Create `rehearse/train/curation/trainer.py` (protocol only this task); Modify `tests/curation/conftest.py`

- [ ] **Step 1:** Add to conftest (no test yet — FakeTrainer is test infrastructure used by Task 4 tests):

```python
class FakeTrainer:
    """Planted loss model: loss = base + sum(harm of included sessions) + seed jitter."""

    def __init__(self, *, session_ids, harm=None, base=3.0, seed_jitter=None):
        self.session_ids = list(session_ids)
        self._harm = harm or {}
        self._base = base
        self._jitter = seed_jitter or {}
        self.calls: list[dict] = []

    def run(self, *, excluded_ids, seed, max_steps, kind):
        included = [s for s in self.session_ids if s not in excluded_ids]
        loss = self._base + sum(self._harm.get(s, 0.0) for s in included) + self._jitter.get(seed, 0.0)
        self.calls.append({"excluded": sorted(excluded_ids), "seed": seed, "kind": kind})
        return loss
```

`trainer.py` holds the protocol:

```python
from typing import Protocol

class Trainer(Protocol):
    def run(self, *, excluded_ids: list[str], seed: int, max_steps: int, kind: str) -> float:
        """Dispatch one budget-matched run over the curated corpus minus excluded_ids; return held-out loss."""
```

- [ ] **Step 2:** Commit with Task 4 (no standalone test).

### Task 4: AblationOrchestrator

**Files:** Create `rehearse/train/curation/ablation.py`; Test `tests/curation/test_ablation.py`

- [ ] **Step 1: Write failing tests**

```python
"""AblationOrchestrator: regression detection, bisection, guardrails, resume."""
from rehearse.train.curation.ablation import AblationOrchestrator
from rehearse.train.curation.ledger import RunLedger, RunRecord
from tests.curation.conftest import FakeTrainer

SESSIONS = ["s00", "s01", "s02", "s03", "s04", "s05"]


def _ledger_with_history(tmp_path, *, prior_sessions, prior_loss=3.0, band_losses=(3.0, 3.02)):
    ledger = RunLedger(tmp_path / "ledger.jsonl")
    mh_prior = AblationOrchestrator.manifest_hash(prior_sessions)
    for i, loss in enumerate(band_losses):
        ledger.append(RunRecord(run_id=f"b{i}", kind="baseline", manifest_hash=mh_prior,
                                session_ids=prior_sessions, seed=i, max_steps=120,
                                heldout_loss=loss, eval_set_version=1))
    return ledger, prior_loss


def _current(sessions, loss, seed=0):
    return RunRecord(run_id="cur", kind="training",
                     manifest_hash=AblationOrchestrator.manifest_hash(sessions),
                     session_ids=sessions, seed=seed, max_steps=120,
                     heldout_loss=loss, eval_set_version=1)


async def test_no_regression_within_band(tmp_path):
    ledger, _ = _ledger_with_history(tmp_path, prior_sessions=SESSIONS[:4])
    trainer = FakeTrainer(session_ids=SESSIONS[:5])
    orch = AblationOrchestrator(ledger=ledger, trainer=trainer)
    result = await orch.investigate(_current(SESSIONS[:5], loss=3.015))
    assert result.outcome == "no_regression"
    assert trainer.calls == []


async def test_single_new_session_convicted(tmp_path):
    ledger, _ = _ledger_with_history(tmp_path, prior_sessions=SESSIONS[:4])
    trainer = FakeTrainer(session_ids=SESSIONS[:5], harm={"s04": 0.5})
    orch = AblationOrchestrator(ledger=ledger, trainer=trainer)
    result = await orch.investigate(_current(SESSIONS[:5], loss=3.5))
    assert result.outcome == "convicted"
    [conviction] = result.convicted
    assert conviction.session_id == "s04"
    assert conviction.delta > 0.4

async def test_bisection_isolates_culprit_among_four_new(tmp_path):
    ledger, _ = _ledger_with_history(tmp_path, prior_sessions=SESSIONS[:2])
    trainer = FakeTrainer(session_ids=SESSIONS, harm={"s03": 0.5})
    orch = AblationOrchestrator(ledger=ledger, trainer=trainer, max_runs=6)
    result = await orch.investigate(_current(SESSIONS, loss=3.5))
    assert result.outcome == "convicted"
    assert [c.session_id for c in result.convicted] == ["s03"]
    assert len(trainer.calls) <= 6


async def test_budget_exhaustion_flags_without_conviction(tmp_path):
    ledger, _ = _ledger_with_history(tmp_path, prior_sessions=SESSIONS[:2])
    trainer = FakeTrainer(session_ids=SESSIONS, harm={"s02": 0.3, "s05": 0.3})  # split culprits
    orch = AblationOrchestrator(ledger=ledger, trainer=trainer, max_runs=2)
    result = await orch.investigate(_current(SESSIONS, loss=3.6))
    assert result.outcome == "inconclusive"
    assert result.convicted == []
    assert ledger.is_flagged(AblationOrchestrator.manifest_hash(SESSIONS))


async def test_cooldown_skips_previously_flagged_manifest(tmp_path):
    ledger, _ = _ledger_with_history(tmp_path, prior_sessions=SESSIONS[:2])
    trainer = FakeTrainer(session_ids=SESSIONS, harm={"s02": 0.3, "s05": 0.3})
    orch = AblationOrchestrator(ledger=ledger, trainer=trainer, max_runs=2)
    await orch.investigate(_current(SESSIONS, loss=3.6))
    again = await orch.investigate(_current(SESSIONS, loss=3.6))
    assert again.outcome == "skipped_cooldown"


async def test_resume_reuses_ledgered_runs(tmp_path):
    ledger, _ = _ledger_with_history(tmp_path, prior_sessions=SESSIONS[:4])
    trainer = FakeTrainer(session_ids=SESSIONS[:5], harm={"s04": 0.5})
    orch = AblationOrchestrator(ledger=ledger, trainer=trainer)
    first = await orch.investigate(_current(SESSIONS[:5], loss=3.5))
    calls_after_first = len(trainer.calls)
    second = await orch.investigate(_current(SESSIONS[:5], loss=3.5))
    assert second.outcome == "convicted"
    assert len(trainer.calls) == calls_after_first  # all runs served from the ledger
```

- [ ] **Step 2:** Run — expect import error.
- [ ] **Step 3: Implement `ablation.py`**

```python
"""Ablation investigations: regression -> cohort bisection -> conviction."""
from __future__ import annotations
import hashlib
import logging
from pydantic import BaseModel
from rehearse.train.curation.ledger import RunLedger, RunRecord

logger = logging.getLogger(__name__)


class Conviction(BaseModel):
    session_id: str
    delta: float
    noise_band: float
    run_ids: list[str]


class InvestigationResult(BaseModel):
    investigation_id: str
    outcome: str  # no_regression | convicted | inconclusive | skipped_cooldown
    convicted: list[Conviction] = []
    runs_used: int = 0


class AblationOrchestrator:
    def __init__(self, *, ledger: RunLedger, trainer, max_runs: int = 6):
        self._ledger = ledger
        self._trainer = trainer
        self._max_runs = max_runs

    @staticmethod
    def manifest_hash(session_ids) -> str:
        return hashlib.sha1("\n".join(sorted(session_ids)).encode()).hexdigest()[:12]

    async def investigate(self, current: RunRecord) -> InvestigationResult:
        investigation_id = f"inv-{current.run_id}"
        result = InvestigationResult(investigation_id=investigation_id, outcome="no_regression")
        if self._ledger.is_flagged(current.manifest_hash):
            result.outcome = "skipped_cooldown"
            return result
        prior = self._ledger.best_accepted(eval_set_version=current.eval_set_version,
                                           exclude_run_id=current.run_id)
        if prior is None:
            return result
        budget = self._max_runs

        band = self._ledger.noise_band(current.manifest_hash,
                                       eval_set_version=current.eval_set_version)
        if band is None and budget > 0:
            self._run(current, [], current.seed + 1, "baseline", investigation_id)
            budget -= 1
            band = self._ledger.noise_band(current.manifest_hash,
                                           eval_set_version=current.eval_set_version)
        band = band if band is not None else 0.0

        if current.heldout_loss <= prior.heldout_loss + band:
            return result

        suspects = sorted(set(current.session_ids) - set(prior.session_ids))
        if not suspects:
            self._flag(current, [], investigation_id)
            result.outcome = "inconclusive"
            return result

        target = prior.heldout_loss
        candidates = suspects
        run_ids: list[str] = []
        while budget > 0:
            if len(candidates) == 1:
                record = self._run_or_resume(current, candidates, investigation_id)
                if record.run_id not in [r.run_id for r in self._ledger.records()]:
                    budget -= 1
                run_ids.append(record.run_id)
                if record.heldout_loss <= target + band:
                    result.convicted = [Conviction(
                        session_id=candidates[0],
                        delta=current.heldout_loss - record.heldout_loss,
                        noise_band=band, run_ids=run_ids)]
                    result.outcome = "convicted"
                else:
                    self._flag(current, candidates, investigation_id)
                    result.outcome = "inconclusive"
                result.runs_used = self._max_runs - budget
                return result
            half_a = candidates[: len(candidates) // 2]
            half_b = candidates[len(candidates) // 2:]
            rec_a = self._run_or_resume(current, half_a, investigation_id); budget -= 1
            run_ids.append(rec_a.run_id)
            if rec_a.heldout_loss <= target + band:
                candidates = half_a
                continue
            if budget <= 0:
                break
            rec_b = self._run_or_resume(current, half_b, investigation_id); budget -= 1
            run_ids.append(rec_b.run_id)
            if rec_b.heldout_loss <= target + band:
                candidates = half_b
                continue
            break  # harm spans both halves — v1 gives up
        self._flag(current, candidates, investigation_id)
        result.outcome = "inconclusive"
        result.runs_used = self._max_runs - budget
        return result
```

with `_run_or_resume` checking `ledger.find_run` first, and `_run` dispatching `self._trainer.run(...)`, appending a `RunRecord` (kind/excluded/investigation_id), and `_flag` appending a zero-loss? — no: `_flag` appends a `RunRecord(kind="ablation", status="completed", heldout_loss=None ... )` is invalid for `_comparable`; instead append a marker record with `flagged_cohort=candidates`, `heldout_loss=None`, `status="completed"`. Adjust `_comparable` filtering already excludes `heldout_loss is None`. Budget bookkeeping must only decrement on real trainer calls (resume hits are free) — track via a small helper returning `(record, was_cached)`.

- [ ] **Step 4:** Iterate until all 6 tests pass. Watch the budget arithmetic on the resume test.
- [ ] **Step 5:** Commit: `feat: ablation orchestrator with bisection and guardrails`

### Task 5: Tombstoning

**Files:** Modify `rehearse/train/curation/curate.py`; Test `tests/curation/test_tombstone.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tombstone: manifest rewrite + review flip, artifacts preserved."""
import json
from tests.curation.conftest import FakeJudgeBackend, FakeVolumeClient, approve_response, make_session_dir
from rehearse.train.curation.ablation import Conviction
from rehearse.train.curation.curate import curate_sessions, tombstone_sessions
from rehearse.train.curation.reviewer import SessionReviewer


async def test_tombstone_rewrites_manifest_and_flips_review(tmp_path):
    root = tmp_path / "sessions"
    for sid in ("keep-1", "bad-1"):
        make_session_dir(root, sid)
    out = tmp_path / "curated.jsonl"
    volume = FakeVolumeClient()
    await curate_sessions(root, criteria_dir=tmp_path / "criteria", out=out,
                          reviewer=SessionReviewer(backend=FakeJudgeBackend(lambda s, u: approve_response())),
                          volume_client=volume)
    conviction = Conviction(session_id="bad-1", delta=0.5, noise_band=0.02, run_ids=["r9"])

    tombstone_sessions(sessions_root=root, out=out, convictions=[conviction], volume_client=volume)

    manifest = out.read_text()
    assert "bad-1" not in manifest and "keep-1" in manifest
    remote = volume.pushed["data/curated_sessions.jsonl"].decode()
    assert "bad-1" not in remote and "keep-1" in remote
    review = json.loads((root / "bad-1" / "review.json").read_text())
    assert review["decision"] == "rejected"
    assert review["ablation"]["delta"] == 0.5 and review["ablation"]["run_ids"] == ["r9"]
    assert "0.5" in review["rationale"]
    assert json.loads(volume.pushed["data/sessions/bad-1/review.json"].decode())["decision"] == "rejected"
    assert (root / "bad-1" / "audio_stereo.wav").exists()  # artifacts untouched
```

- [ ] **Step 2:** Run — ImportError (`tombstone_sessions`).
- [ ] **Step 3: Implement** in `curate.py`:

```python
def tombstone_sessions(*, sessions_root: Path, out: Path, convictions, volume_client=None) -> None:
    convicted = {c.session_id: c for c in convictions}
    entries = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    kept = [e for e in entries if Path(e["path"]).parent.name not in convicted]
    with open(out, "w") as fh:
        for entry in kept:
            fh.write(json.dumps(entry) + "\n")
    files: list[tuple[str, bytes]] = []
    for session_id, conviction in convicted.items():
        session_dir = next(p.parent for p in sessions_root.rglob("audio_stereo.wav")
                           if p.parent.name == session_id)
        review = SessionReview.model_validate_json((session_dir / "review.json").read_text())
        review = review.model_copy(update={
            "decision": "rejected",
            "turning_point": None,
            "ablation": conviction.model_dump(),
            "rationale": (f"Tombstoned by ablation {conviction.run_ids}: held-out loss delta "
                          f"{conviction.delta} (noise band {conviction.noise_band}). "
                          f"Prior rationale: {review.rationale}"),
        })
        (session_dir / "review.json").write_text(review.model_dump_json(indent=2))
        files.append((f"data/sessions/{session_id}/review.json",
                      review.model_dump_json(indent=2).encode()))
    if volume_client is not None:
        remote_entries = [{**e, "path": f"/data/data/sessions/{Path(e['path']).parent.name}/audio_stereo.wav"}
                          for e in kept]
        manifest = ("\n".join(json.dumps(e) for e in remote_entries) + "\n").encode()
        files.append(("data/curated_sessions.jsonl", manifest))
        volume_client.push(files)
```

(Import `Conviction` only under `TYPE_CHECKING` or accept any object with the fields, to avoid a curate→ablation import cycle; ablation must not import curate.)

- [ ] **Step 4:** Run — pass. Run whole curation suite.
- [ ] **Step 5:** Commit: `feat: tombstone convicted sessions from the curated manifest`

### Task 6: Training-cycle composition + SIA hook + CLI

**Files:** Create `rehearse/train/curation/cli.py`; Test `tests/curation/test_cycle.py`

- [ ] **Step 1: Write failing test** — full loop with fakes: regression → conviction → tombstone → criteria v2.

```python
"""End-to-end training cycle: sweep -> train -> investigate -> tombstone -> SIA."""
import json
from tests.curation.conftest import (FakeJudgeBackend, FakeVolumeClient, FakeTrainer,
                                     approve_response, make_session_dir)
from rehearse.train.curation.cli import run_training_cycle
from rehearse.train.curation.criteria import load_criteria
from rehearse.train.curation.ledger import RunLedger
from rehearse.train.curation.reviewer import SessionReviewer


async def test_cycle_convicts_tombstones_and_updates_criteria(tmp_path):
    root = tmp_path / "sessions"
    for i in range(6):
        make_session_dir(root, f"s{i:02d}")
    judge = FakeJudgeBackend(lambda s, u: approve_response() if "criteria" not in u.lower()[:30]
                             else approve_response())
    # Criteria-update calls go to a separate fake returning the SIA JSON.
    feedback_backend = FakeJudgeBackend(
        lambda s, u: json.dumps({"criteria": "# v2", "improvement": "ablation evidence"}))
    ledger = RunLedger(tmp_path / "ledger.jsonl")
    volume = FakeVolumeClient()

    def trainer_factory(session_ids):
        return FakeTrainer(session_ids=session_ids, harm={sid: (0.5 if sid.endswith("3") else 0.0)
                                                          for sid in session_ids})

    # Cycle 1: establishes baseline ledger entries (no prior run to regress against).
    await run_training_cycle(
        sessions_root=root, criteria_dir=tmp_path / "criteria", out=tmp_path / "curated.jsonl",
        ledger=ledger, trainer_factory=trainer_factory, volume_client=volume,
        reviewer=SessionReviewer(backend=judge), feedback_backend=feedback_backend,
        seeds=(0, 1), max_steps=120,
    )
    # Remove the harmful session's harm? No — instead cycle 1 trains WITHOUT s03 approved yet... simpler:
    # see Step 3: run_training_cycle(first_cycle_baseline_seeds=2) runs two baseline seeds when the
    # ledger is empty, so cycle 1 alone gives band + best-accepted; the same cycle's training run then
    # regresses only if the corpus changed. For the test, delete s03's review and re-add the session
    # between cycles to make it "new".
```

The composition function signature locked in Step 3 makes this test concrete; write the final test against it:

```python
async def test_cycle_convicts_tombstones_and_updates_criteria(tmp_path):
    root = tmp_path / "sessions"
    for i in range(5):
        make_session_dir(root, f"s{i:02d}")
    reviewer = SessionReviewer(backend=FakeJudgeBackend(lambda s, u: approve_response()))
    feedback_backend = FakeJudgeBackend(
        lambda s, u: json.dumps({"criteria": "# v2", "improvement": "ablation evidence"}))
    ledger = RunLedger(tmp_path / "ledger.jsonl")
    volume = FakeVolumeClient()
    harm = {"s99": 0.5}
    kwargs = dict(sessions_root=root, criteria_dir=tmp_path / "criteria",
                  out=tmp_path / "curated.jsonl", ledger=ledger,
                  trainer_factory=lambda ids: FakeTrainer(session_ids=ids, harm=harm),
                  volume_client=volume, reviewer=reviewer,
                  feedback_backend=feedback_backend, max_steps=120)

    first = await run_training_cycle(**kwargs)          # clean corpus: baseline established
    assert first.investigation.outcome == "no_regression"

    make_session_dir(root, "s99")                        # harmful newcomer arrives
    second = await run_training_cycle(**kwargs)

    assert second.investigation.outcome == "convicted"
    assert [c.session_id for c in second.investigation.convicted] == ["s99"]
    assert "s99" not in (tmp_path / "curated.jsonl").read_text()
    assert json.loads((root / "s99" / "review.json").read_text())["decision"] == "rejected"
    assert load_criteria(tmp_path / "criteria")[0] == 2  # SIA fired
```

- [ ] **Step 2:** Run — ImportError.
- [ ] **Step 3: Implement `cli.py`**

```python
"""Compose the curation training cycle: sweep -> train -> check -> investigate -> tombstone -> SIA."""
from __future__ import annotations
import argparse, asyncio, logging, uuid
from dataclasses import dataclass
from pathlib import Path
from rehearse.train.curation.ablation import AblationOrchestrator, InvestigationResult
from rehearse.train.curation.curate import curate_sessions, tombstone_sessions
from rehearse.train.curation.feedback import update_criteria
from rehearse.train.curation.ledger import RunLedger, RunRecord

logger = logging.getLogger(__name__)


@dataclass
class CycleResult:
    run: RunRecord
    investigation: InvestigationResult


async def run_training_cycle(*, sessions_root: Path, criteria_dir: Path, out: Path,
                             ledger: RunLedger, trainer_factory, volume_client=None,
                             reviewer=None, feedback_backend=None, max_steps: int = 120,
                             max_runs: int = 6, seed: int = 0) -> CycleResult:
    curation = await curate_sessions(sessions_root, criteria_dir=criteria_dir, out=out,
                                     reviewer=reviewer, volume_client=volume_client)
    train_ids = sorted(r.session_id for r in curation.approved if r.split == "train")
    trainer = trainer_factory(train_ids)
    manifest_hash = AblationOrchestrator.manifest_hash(train_ids)
    loss = trainer.run(excluded_ids=[], seed=seed, max_steps=max_steps, kind="training")
    run = RunRecord(run_id=f"run-{uuid.uuid4().hex[:8]}", kind="training",
                    manifest_hash=manifest_hash, session_ids=train_ids, seed=seed,
                    max_steps=max_steps, heldout_loss=loss)
    ledger.append(run)
    orch = AblationOrchestrator(ledger=ledger, trainer=trainer, max_runs=max_runs)
    investigation = await orch.investigate(run)
    if investigation.convicted:
        tombstone_sessions(sessions_root=sessions_root, out=out,
                           convictions=investigation.convicted, volume_client=volume_client)
        reviews = curation.approved + curation.rejected
        summary = (f"run {run.run_id}: held-out loss {loss:.3f}; investigation "
                   f"{investigation.investigation_id} convicted "
                   f"{[c.session_id for c in investigation.convicted]} "
                   f"(deltas {[c.delta for c in investigation.convicted]})")
        await update_criteria(criteria_dir=criteria_dir, reviews=reviews,
                              training_summary=summary, backend=feedback_backend)
    return CycleResult(run=run, investigation=investigation)
```

plus an argparse `main()` with `train` / `ablate --run <id>` / `sweep` subcommands that build the live pieces (ModalTrainer, TrainingVolumeClient, ledger under `runs/curation/`).

- [ ] **Step 4:** Run the cycle test and full suite — pass. (First cycle must be `no_regression`: with an empty ledger `best_accepted` is None.)
- [ ] **Step 5:** Commit: `feat: curation training cycle with auto ablation and SIA hook`

### Task 7: ModalTrainer live adapter

**Files:** Modify `rehearse/train/curation/trainer.py`; Test `tests/curation/test_modal_trainer.py` (live_modal-marked, skipped by default)

- [ ] **Step 1:** Implement `ModalTrainer` — builds the ablated manifest in memory from the local curated manifest minus `excluded_ids`, pushes it to `data/ablation/<run_id>.jsonl` via `TrainingVolumeClient`, dispatches `rehearse.train.modal.run_training` with `data.train_data=/data/data/ablation/<run_id>.jsonl`, `data.eval_data=/data/data/heldout_sessions.jsonl`, `max_steps`, `seed`, `run_dir=/data/runs/<run_id>`, then reads the final eval loss from the run's metrics (confirm filename under `/data/runs/<run_id>/` at implementation; expected moshi-finetune writes `metrics.eval.jsonl`-style logs — use `modal volume get` and parse the last eval entry).
- [ ] **Step 2:** Live smoke test marked `@pytest.mark.live_modal` running one 10-step investigation against a tiny manifest; assert a float loss comes back. Default suite skips it.
- [ ] **Step 3:** Commit: `feat: modal-backed trainer adapter for ablation runs`

### Task 8: Wrap-up

- [ ] Run full suite (`.venv/bin/python -m pytest -q`) — no new failures vs the known `test_stereo_channel.py` baseline; `ruff check` clean.
- [ ] Push branch; PR #33 updates with spec + implementation. Update PR description with the ablation feature summary.

## Self-review notes

- Spec coverage: ledger (T1), split + heldout manifest (T2), trainer protocol (T3), orchestrator with band/bisection/guardrails/resume/cooldown (T4), tombstone (T5), auto-trigger + pre-training sweep + SIA hook (T6), live Modal adapter + open metrics item (T7). Eval-set-version freeze is carried in RunRecord and `_comparable`.
- Types consistent: `Trainer.run(excluded_ids, seed, max_steps, kind) -> float`; `Conviction{session_id, delta, noise_band, run_ids}`; `RunRecord.session_ids` added beyond spec JSON sketch (needed for suspect computation) — spec's ledger schema is a subset, acceptable extension.
- No placeholders remain except the explicitly flagged live metrics filename (verified in T7 against a real run, gated behind live_modal).
