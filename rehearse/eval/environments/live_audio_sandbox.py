"""Live-audio sandbox environment for eval rollouts.

Replaces RuntimeHost + AudioCustomerDriver + InMemoryTwoWayChannel with the
production conversation loop (run_session) and an EvalCallerParticipant.
The eval now runs exactly the same pipeline as a live Twilio call: phases,
consent (skipped via flag), outcome probe, survey, writers, and memory
recording all fire from the same code path.

Backend selection follows the same BACKEND_TYPE env var as production. Pass
model_slots={"backend_type": "pipeline"} to override per-run.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from rehearse.backends.factory import create_backend
from rehearse.config import RuntimeConfig
from rehearse.session.conversation import run_session
from rehearse.eval.drivers.eval_caller import EvalCallerParticipant
from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.environments.tts_bridge import TTSProvider, get_default_provider
from rehearse.memory.memory import NullCallerMemory
from rehearse.phases.phases import PhaseBudgets
from rehearse.session.session import SessionOrchestrator, TriggerEvent, utcnow
from rehearse.storage import LocalFilesystemStore
from rehearse.types import ConsentState, Session


class LiveAudioSandboxEnvironment:
    """Run the real conversation pipeline with a scripted caller and live backend.

    The eval caller generates turns via LLM + TTS (same as AudioCustomerDriver).
    The backend is selected via BACKEND_TYPE env var (same as production).
    run_session() is called with skip_consent=True so the eval skips the
    consent gate but still exercises phases, outcome, survey, and all writers.
    """

    name = "live-audio-sandbox"
    version = "v2"

    def __init__(
        self,
        model_slots: dict[str, str] | None = None,
        *,
        tts_provider: TTSProvider | None = None,
        llm_client: Any = None,
        customer_max_turns: int = 12,
        # Kept for test backward-compat; ignored in v2 (backend comes from factory)
        coach_adapter_factory: Any = None,
    ) -> None:
        self.model_slots = dict(model_slots or {})
        self._tts_provider = tts_provider
        self._llm_client = llm_client
        self._customer_max_turns = customer_max_turns

        missing = [
            name for name in ("ANTHROPIC_API_KEY", "HUME_API_KEY")
            if not os.environ.get(name) and tts_provider is None and llm_client is None
        ]
        if missing and coach_adapter_factory is None:
            raise RuntimeError(
                "LiveAudioSandboxEnvironment requires "
                + ", ".join(missing)
                + " (or inject tts_provider + llm_client for testing)"
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
        session_dir = run_dir.parent / session_id
        if not session_dir.exists():
            run_dir.rename(session_dir)
        else:
            session_dir = run_dir

        store = LocalFilesystemStore(session_dir.parent, public_base_url="http://localhost")
        scenario = example.payload.get("scenario", example.payload)

        tts = self._tts_provider or get_default_provider()
        if tts is None:
            raise RuntimeError("LiveAudioSandboxEnvironment requires a TTSProvider")

        # Write a minimal session.json so run_session() can read consent/phase state
        _session = Session(
            created_at=utcnow(),
            phone_number_hash=None,
            consent=ConsentState.GRANTED,
        )
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "session.json").write_text(_session.model_dump_json(indent=2))
        # Override session_id to match eval convention
        _session_store = LocalFilesystemStore(session_dir.parent, public_base_url="http://localhost")

        caller = EvalCallerParticipant(
            scenario=scenario,
            tts=tts,
            llm_client=self._llm_client,
            max_turns=self._customer_max_turns,
        )

        # Build backend from env (same factory as production)
        env_overrides = {k: v for k, v in self.model_slots.items()}
        config = _minimal_config(env_overrides)
        backend = create_backend(config)

        error: str | None = None
        try:
            async with backend:
                await run_session(
                    session_id,
                    caller,
                    backend,
                    store=_session_store,
                    memory=NullCallerMemory(),
                    caller_hash=None,
                    budgets=PhaseBudgets(),
                    skip_consent=True,
                    model_id=config.hume_config_id,
                    llm_client=None,
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc(file=sys.stderr)

        provenance = {
            "environment": self.name,
            "environment_version": self.version,
            "tier": "live-audio",
            "runtime_kernel": "real-run-session",
            "backend_type": getattr(config, "backend_type", "managed"),
            "synthetic_caller": "EvalCallerParticipant",
            "anthropic_api_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "hume_api_key_set": bool(os.environ.get("HUME_API_KEY")),
        }
        (session_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

        caller_usage = caller._usage  # noqa: SLF001
        token_usage: dict[str, int] = {
            "customer_prompt_tokens": caller_usage.get("prompt_tokens", 0),
            "customer_completion_tokens": caller_usage.get("completion_tokens", 0),
        }
        token_usage["total_tokens"] = sum(token_usage.values())

        completed = datetime.now()
        return RolloutResult(
            example_id=example.id,
            target_name=self.name,
            target_version=self.version,
            status="error" if error else "ok",
            started_at=started,
            completed_at=completed,
            duration_ms=int((completed - started).total_seconds() * 1000),
            artifacts_dir=session_dir,
            error=error,
            payload={"tts_provider": getattr(tts, "name", "unknown")},
            token_usage=token_usage if token_usage["total_tokens"] > 0 else None,
        )


def _minimal_config(overrides: dict[str, str]) -> RuntimeConfig:
    """Build a RuntimeConfig for eval with placeholder Twilio/Hume values."""
    import os
    from unittest.mock import MagicMock

    # RuntimeConfig.from_env() requires Twilio keys we don't have in eval.
    # Build a minimal config directly with only what create_backend() needs.
    cfg = MagicMock(spec=RuntimeConfig)
    cfg.backend_type = overrides.get("backend_type") or os.environ.get("BACKEND_TYPE", "managed")
    cfg.hume_api_key = os.environ.get("HUME_API_KEY", "")
    cfg.hume_config_id = overrides.get("hume_config_id") or os.environ.get("HUME_CONFIG_ID", "default")
    cfg.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    cfg.anthropic_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    cfg.pipeline_stt_model = overrides.get("stt_model") or os.environ.get("PIPELINE_STT_MODEL", "whisper-tiny")
    cfg.pipeline_tts_model = overrides.get("tts_model") or os.environ.get("PIPELINE_TTS_MODEL", "kokoro")
    cfg.clm_url = os.environ.get("CLM_URL", "http://localhost:8000")
    return cfg
