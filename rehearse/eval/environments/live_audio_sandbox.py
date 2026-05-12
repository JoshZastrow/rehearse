"""Live-audio sandbox environment for RuntimeHost eval rollouts."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from rehearse.eval.customers.audio_customer import AudioCustomerDriver
from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.tts_bridge import TTSProvider, get_default_provider
from rehearse.runtime import AudioCoachAdapter, HumeEVIAdapter, RuntimeHost
from rehearse.storage import LocalFilesystemStore
from rehearse.transport import InMemoryTwoWayChannel


class LiveAudioSandboxEnvironment:
    """Run the real runtime with caller PCM and an audio-capable coach adapter."""

    name = "live-audio-sandbox"
    version = "v1"

    def __init__(
        self,
        model_slots: dict[str, str] | None = None,
        *,
        coach_adapter_factory: Callable[[], AudioCoachAdapter] | None = None,
        tts_provider: TTSProvider | None = None,
        llm_client: Any = None,
        customer_max_turns: int = 12,
    ) -> None:
        self.model_slots = dict(model_slots or {})
        self._coach_adapter_factory = coach_adapter_factory
        self._tts_provider = tts_provider
        self._llm_client = llm_client
        self._customer_max_turns = customer_max_turns
        if coach_adapter_factory is None:
            missing = [
                name
                for name in ("ANTHROPIC_API_KEY", "HUME_API_KEY")
                if not os.environ.get(name)
            ]
            if missing:
                raise RuntimeError(
                    "LiveAudioSandboxEnvironment requires "
                    + ", ".join(missing)
                    + " unless coach_adapter_factory and test doubles are provided"
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
        transport = InMemoryTwoWayChannel()
        scenario = example.payload.get("scenario", example.payload)

        tts = self._tts_provider or get_default_provider()
        if tts is None:
            raise RuntimeError("LiveAudioSandboxEnvironment requires a TTSProvider")
        coach = self._build_coach(session_id)
        host = RuntimeHost(store, coach, enable_audio_recording=True)
        customer = AudioCustomerDriver(
            scenario=scenario,
            tts=tts,
            llm_client=self._llm_client,
            run_dir=session_dir,
            max_turns=self._customer_max_turns,
        )

        error: str | None = None
        try:
            async with coach:
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

        provenance = self._provenance(coach)
        (session_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )

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
        )

    def _build_coach(self, session_id: str) -> AudioCoachAdapter:
        if self._coach_adapter_factory is not None:
            return self._coach_adapter_factory()
        config_id = self.model_slots.get("hume_config_id") or os.environ.get(
            "HUME_CONFIG_ID", "default"
        )
        return HumeEVIAdapter(
            api_key=os.environ["HUME_API_KEY"],
            config_id=config_id,
            session_id=session_id,
        )

    def _provenance(self, coach: AudioCoachAdapter) -> dict[str, Any]:
        return {
            "environment": self.name,
            "environment_version": self.version,
            "tier": "live-audio",
            "runtime_kernel": "real",
            "intake_processor": "real (deterministic)",
            "persona_compiler": "real (deterministic)",
            "coach_voice_adapter": type(coach).__name__,
            "transport": "InMemoryTwoWayChannel",
            "synthetic_caller": "AudioCustomerDriver",
            "anthropic_api_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "hume_api_key_set": bool(os.environ.get("HUME_API_KEY")),
            "audio_stub": False,
        }
