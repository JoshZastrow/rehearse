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
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

from rehearse.backends.factory import create_backend
from rehearse.config import RuntimeConfig
from rehearse.session.conversation import run_session
from rehearse.eval.customers.caller_clients import DEFAULT_CALLER_MODEL, make_caller_client
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
        self._customer_max_turns = customer_max_turns
        # Injected client is used as-is (tests); otherwise a fresh client is
        # created per rollout via _make_caller_client() to avoid shared _usage state.
        self._injected_llm_client = llm_client
        self._caller_model = self.model_slots.get("caller", DEFAULT_CALLER_MODEL)

        if not os.environ.get("HUME_API_KEY") and tts_provider is None and coach_adapter_factory is None:
            raise RuntimeError(
                "LiveAudioSandboxEnvironment requires HUME_API_KEY "
                "(or inject tts_provider for testing)"
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

        # Fresh client per rollout so _usage is isolated (no cross-rollout accumulation).
        llm_client = self._injected_llm_client or make_caller_client(self._caller_model)

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
            llm_client=llm_client,
            max_turns=self._customer_max_turns,
        )

        # Build backend from env (same factory as production)
        env_overrides = {k: v for k, v in self.model_slots.items()}
        config = _minimal_config(env_overrides)
        backend = create_backend(config)

        log.info(
            "[%s] starting rollout: backend_type=%s caller_model=%s "
            "anthropic_key=%s hume_key=%s gemini_key=%s",
            session_id,
            config.backend_type,
            self._caller_model,
            bool(os.environ.get("ANTHROPIC_API_KEY")),
            bool(os.environ.get("HUME_API_KEY")),
            bool(os.environ.get("GEMINI_API_KEY")),
        )

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
            log.error("[%s] rollout failed: %s", session_id, error)
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

        # Wrapper clients track usage on their own _usage dict.
        # EvalCallerParticipant._usage only updates on the legacy Anthropic SDK
        # fallback path — check both and take whichever is non-zero.
        wrapper_usage = getattr(llm_client, "_usage", {})
        caller_usage = caller._usage  # noqa: SLF001
        llm_usage = wrapper_usage if any(wrapper_usage.values()) else caller_usage
        token_usage: dict[str, int] = {
            "customer_prompt_tokens": llm_usage.get("prompt_tokens", 0),
            "customer_completion_tokens": llm_usage.get("completion_tokens", 0),
        }
        token_usage["total_tokens"] = sum(token_usage.values())

        completed = datetime.now()
        duration_s = (completed - started).total_seconds()

        # Log key outcome artifacts for post-run debugging
        transcript_path = session_dir / "transcript.jsonl"
        audio_path = session_dir / "audio.wav"
        transcript_lines = len(transcript_path.read_text().splitlines()) if transcript_path.exists() else 0
        audio_bytes = audio_path.stat().st_size if audio_path.exists() else 0
        log.info(
            "[%s] rollout done in %.1fs: status=%s transcript_lines=%d audio_bytes=%d tokens=%s",
            session_id,
            duration_s,
            "error" if error else "ok",
            transcript_lines,
            audio_bytes,
            token_usage if token_usage["total_tokens"] > 0 else "none",
        )

        return RolloutResult(
            example_id=example.id,
            target_name=self.name,
            target_version=self.version,
            status="error" if error else "ok",
            started_at=started,
            completed_at=completed,
            duration_ms=int(duration_s * 1000),
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
    hume_api_key = os.environ.get("HUME_API_KEY", "")
    hume_config_id = overrides.get("hume_config_id") or os.environ.get("HUME_CONFIG_ID", "default")
    cfg.hume_api_key = hume_api_key
    cfg.hume_config_id = hume_config_id
    # managed_api_key / managed_config_id are what create_backend() actually reads;
    # RuntimeConfig normally backfills these from hume_* via a validator that
    # doesn't run on MagicMock, so set them explicitly.
    cfg.managed_api_key = hume_api_key
    cfg.managed_config_id = hume_config_id
    cfg.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    cfg.anthropic_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    cfg.pipeline_stt_model = overrides.get("stt_model") or os.environ.get("PIPELINE_STT_MODEL", "whisper-tiny")
    cfg.pipeline_tts_model = overrides.get("tts_model") or os.environ.get("PIPELINE_TTS_MODEL", "kokoro")
    cfg.pipeline_speech_mode = os.environ.get("PIPELINE_SPEECH_MODE", "modular")
    cfg.pipeline_clm_url = os.environ.get("PIPELINE_CLM_URL") or os.environ.get("LITELLM_BASE_URL", "http://localhost:4000") + "/chat/completions"
    cfg.pipeline_clm_model = os.environ.get("PIPELINE_CLM_MODEL") or os.environ.get("LITELLM_MODEL", "coach")
    cfg.clm_url = os.environ.get("CLM_URL", "http://localhost:8000")
    cfg.interactive_checkpoint_path = os.environ.get("INTERACTIVE_CHECKPOINT_PATH", "")
    cfg.interactive_model_repo = os.environ.get("INTERACTIVE_MODEL_REPO", "kyutai/moshiko-pytorch-bf16")
    cfg.interactive_device = os.environ.get("INTERACTIVE_DEVICE", "cuda")
    cfg.interactive_asr_model = os.environ.get("INTERACTIVE_ASR_MODEL", "base")
    cfg.interactive_model_type = os.environ.get("INTERACTIVE_MODEL_TYPE", "moshi")
    cfg.interactive_provider_endpoint = os.environ.get("INTERACTIVE_PROVIDER_ENDPOINT", "")
    return cfg
