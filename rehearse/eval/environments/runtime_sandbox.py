"""RuntimeSandboxEnvironment — wires the real RuntimeHost + LLMCustomerDriver.

This environment runs the same runtime code used in production (PhaseProcessor,
IntakeProcessor, PersonaCompiler, TranscriptWriter, TimingWriter) against an
LLM-simulated customer. No static coach prompt is used; the coach is driven by
TextOnlyCoachAdapter calling Anthropic directly.

Required environment variable: ANTHROPIC_API_KEY (used by both the LLM coach
and the synthetic customer).
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from rehearse.eval.customers.llm_customer import LLMCustomerDriver
from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.runtime import RuntimeHost, TextOnlyCoachAdapter
from rehearse.storage import LocalFilesystemStore
from rehearse.transport import InMemoryDuplexTransport


class RuntimeSandboxEnvironment:
    """Eval environment that runs the real runtime against an LLM customer."""

    name = "runtime-sandbox"
    version = "v1"

    def __init__(self, model_slots: dict[str, str] | None = None) -> None:
        self.model_slots = model_slots or {}
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "RuntimeSandboxEnvironment: ANTHROPIC_API_KEY is required "
                "(used by the LLM coach and synthetic customer). "
                "Set it in .env before running --environment runtime-sandbox."
            )

    async def rollout(
        self,
        example: BenchmarkExample,
        run_dir: Path,
        rng_seed: int,
    ) -> RolloutResult:
        started = datetime.now()
        run_dir.mkdir(parents=True, exist_ok=True)
        session_id = f"eval-{example.id}"

        store = LocalFilesystemStore(run_dir.parent, public_base_url="http://localhost")

        # Rename run_dir to match session_id so LocalFilesystemStore finds it.
        session_dir = run_dir.parent / session_id
        if not session_dir.exists():
            run_dir.rename(session_dir)
        else:
            session_dir = run_dir

        store = LocalFilesystemStore(session_dir.parent, public_base_url="http://localhost")

        coach = TextOnlyCoachAdapter()
        host = RuntimeHost(store, coach)

        scenario = example.payload.get("scenario", example.payload)
        customer = LLMCustomerDriver(
            scenario=scenario,
            run_dir=session_dir,
        )

        transport = InMemoryDuplexTransport()

        error: str | None = None
        try:
            await asyncio.gather(
                host.run(session_id=session_id, transport=transport.runtime),
                customer.run(
                    transport=transport.customer,
                    runtime_phase=lambda: host.current_phase,
                ),
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc(file=sys.stderr)

        completed = datetime.now()
        duration_ms = int((completed - started).total_seconds() * 1000)

        return RolloutResult(
            example_id=example.id,
            target_name=self.name,
            target_version=self.version,
            status="error" if error else "ok",
            started_at=started,
            completed_at=completed,
            duration_ms=duration_ms,
            artifacts_dir=session_dir,
            error=error,
        )
