"""Compose the curation training cycle and expose it as a CLI.

Cycle: pre-training sweep (curate_sessions) -> train -> regression check ->
auto ablation investigation -> tombstone convictions -> SIA criteria update.

Commands:
    python -m rehearse.train.curation.cli sweep            # curate + push only
    python -m rehearse.train.curation.cli train            # full cycle
    python -m rehearse.train.curation.cli ablate --run ID  # re-investigate a ledgered run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rehearse.train.curation.ablation import AblationOrchestrator, InvestigationResult
from rehearse.train.curation.curate import curate_sessions, tombstone_sessions
from rehearse.train.curation.feedback import update_criteria
from rehearse.train.curation.ledger import RunLedger, RunRecord

logger = logging.getLogger(__name__)

_DEFAULT_LEDGER = Path("runs/curation/run_ledger.jsonl")


@dataclass
class CycleResult:
    run: RunRecord
    investigation: InvestigationResult


async def run_training_cycle(
    *,
    sessions_root: Path,
    criteria_dir: Path,
    out: Path,
    ledger: RunLedger,
    trainer_factory: Any,
    volume_client: Any = None,
    reviewer: Any = None,
    feedback_backend: Any = None,
    max_steps: int = 120,
    max_runs: int = 6,
    seed: int = 0,
) -> CycleResult:
    curation = await curate_sessions(
        sessions_root, criteria_dir=criteria_dir, out=out, reviewer=reviewer,
        volume_client=volume_client,
    )
    train_ids = sorted(r.session_id for r in curation.approved if r.split == "train")
    trainer = trainer_factory(train_ids)

    loss = trainer.run(excluded_ids=[], seed=seed, max_steps=max_steps, kind="training")
    run = RunRecord(
        run_id=f"run-{uuid.uuid4().hex[:8]}",
        kind="training",
        manifest_hash=AblationOrchestrator.manifest_hash(train_ids),
        session_ids=train_ids,
        seed=seed,
        max_steps=max_steps,
        heldout_loss=loss,
    )
    ledger.append(run)
    logger.info("Training run %s: held-out loss %.4f over %d sessions", run.run_id, loss,
                len(train_ids))

    orchestrator = AblationOrchestrator(ledger=ledger, trainer=trainer, max_runs=max_runs)
    investigation = await orchestrator.investigate(run)
    logger.info("Investigation %s: %s", investigation.investigation_id, investigation.outcome)

    if investigation.convicted:
        tombstone_sessions(
            sessions_root=sessions_root, out=out, convictions=investigation.convicted,
            volume_client=volume_client,
        )
        summary = (
            f"run {run.run_id}: held-out loss {loss:.4f} regressed; investigation "
            f"{investigation.investigation_id} convicted "
            f"{[c.session_id for c in investigation.convicted]} with held-out loss deltas "
            f"{[round(c.delta, 4) for c in investigation.convicted]} "
            f"(noise band {investigation.convicted[0].noise_band}). These sessions passed "
            f"review but measurably hurt training; they are tombstoned from the manifest."
        )
        await update_criteria(
            criteria_dir=criteria_dir,
            reviews=curation.approved + curation.rejected,
            training_summary=summary,
            backend=feedback_backend,
        )

    return CycleResult(run=run, investigation=investigation)


def _build_live_pieces(args: argparse.Namespace) -> dict[str, Any]:
    from rehearse.train.curation.trainer import ModalTrainer
    from rehearse.train.curation.volume import TrainingVolumeClient

    return {
        "sessions_root": args.sessions_root,
        "criteria_dir": args.criteria_dir,
        "out": args.out,
        "ledger": RunLedger(args.ledger),
        "trainer_factory": lambda train_ids: ModalTrainer(
            manifest_path=args.out, train_ids=train_ids
        ),
        "volume_client": TrainingVolumeClient(),
        "max_steps": args.max_steps,
        "max_runs": args.max_runs,
    }


async def _ablate(args: argparse.Namespace) -> None:
    pieces = _build_live_pieces(args)
    ledger: RunLedger = pieces["ledger"]
    run = next((r for r in ledger.records() if r.run_id == args.run), None)
    if run is None:
        raise SystemExit(f"run {args.run!r} not found in {args.ledger}")
    trainer = pieces["trainer_factory"](run.session_ids)
    orchestrator = AblationOrchestrator(ledger=ledger, trainer=trainer, max_runs=args.max_runs)
    investigation = await orchestrator.investigate(run)
    print(investigation.model_dump_json(indent=2))
    if investigation.convicted:
        tombstone_sessions(
            sessions_root=args.sessions_root, out=args.out,
            convictions=investigation.convicted, volume_client=pieces["volume_client"],
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-root", type=Path, default=Path("sessions"))
    parser.add_argument("--criteria-dir", type=Path, default=Path("runs/curation/criteria"))
    parser.add_argument("--out", type=Path, default=Path("runs/curation/curated_sessions.jsonl"))
    parser.add_argument("--ledger", type=Path, default=_DEFAULT_LEDGER)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--max-runs", type=int, default=6)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sweep")
    sub.add_parser("train")
    ablate_cmd = sub.add_parser("ablate")
    ablate_cmd.add_argument("--run", required=True)
    args = parser.parse_args()

    if args.command == "sweep":
        from rehearse.train.curation.volume import TrainingVolumeClient

        asyncio.run(
            curate_sessions(
                args.sessions_root, criteria_dir=args.criteria_dir, out=args.out,
                volume_client=TrainingVolumeClient(),
            )
        )
    elif args.command == "train":
        result = asyncio.run(run_training_cycle(**_build_live_pieces(args)))
        print(result.investigation.model_dump_json(indent=2))
    elif args.command == "ablate":
        asyncio.run(_ablate(args))


if __name__ == "__main__":
    main()
