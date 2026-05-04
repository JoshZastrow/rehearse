"""InProcessExecutor — run rollouts in the harness process.

Used when test fixtures (monkeypatch) need to apply to the rollout, and when
``--verbose`` needs a live ``on_event`` hook on the transport (subprocess
boundaries would swallow the callback).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from rehearse.eval.environments import get_environment
from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.transports import TransportEvent


class InProcessExecutor:
    """Runs ``Environment.rollout`` in the current process. Honors timeout.

    If ``on_event`` is supplied and the resolved environment exposes an
    ``on_event`` attribute, it is wired through so transport events stream
    live during the rollout.
    """

    def __init__(
        self,
        *,
        on_event: Callable[[TransportEvent], None] | None = None,
    ) -> None:
        self.on_event = on_event

    async def submit(
        self,
        target_name: str,
        target_version: str,
        model_slots: dict[str, str],
        example: BenchmarkExample,
        run_dir: Path,
        timeout_s: int,
        rng_seed: int,
    ) -> RolloutResult:
        run_dir.mkdir(parents=True, exist_ok=True)
        env = get_environment(target_name, model_slots)
        if self.on_event is not None and hasattr(env, "on_event"):
            env.on_event = self.on_event
        started = datetime.now()
        try:
            return await asyncio.wait_for(
                env.rollout(example, run_dir, rng_seed), timeout=timeout_s
            )
        except TimeoutError:
            completed = datetime.now()
            return RolloutResult(
                example_id=example.id,
                target_name=target_name,
                target_version=target_version,
                status="timeout",
                started_at=started,
                completed_at=completed,
                duration_ms=int((completed - started).total_seconds() * 1000),
                error=f"in-process rollout exceeded {timeout_s}s",
            )
