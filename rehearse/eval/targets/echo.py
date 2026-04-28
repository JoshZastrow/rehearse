"""Echo target — returns the example's payload unchanged.

Phase 1 skeleton. Lets the harness run without any model API.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rehearse.eval.protocols import BenchmarkExample, RolloutResult


class EchoTarget:
    name = "echo"
    version = "v0"

    def __init__(self, model_slots: dict[str, str] | None = None) -> None:
        self.model_slots = model_slots or {}

    async def rollout(
        self,
        example: BenchmarkExample,
        run_dir: Path,
        rng_seed: int,
    ) -> RolloutResult:
        started = datetime.now()
        completed = datetime.now()
        return RolloutResult(
            example_id=example.id,
            target_name=self.name,
            target_version=self.version,
            status="ok",
            started_at=started,
            completed_at=completed,
            duration_ms=int((completed - started).total_seconds() * 1000),
            payload={"echo": example.payload.get("echo")},
        )
