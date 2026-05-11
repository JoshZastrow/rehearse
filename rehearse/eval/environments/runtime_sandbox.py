"""RuntimeSandboxEnvironment — wires the real RuntimeHost + SyntheticCaller.

This environment runs the same runtime code used in production (PhaseProcessor,
IntakeProcessor, PersonaCompiler, TranscriptWriter, TimingWriter) against an
LLM-simulated customer. No static coach prompt is used; the coach is driven by
TextOnlyCoachAdapter calling Anthropic directly.

After the runtime/customer gather completes, post-hoc TTS turns each transcript
turn into a per-turn WAV under `audio/{user,coach}/turn_<N>.wav` and writes
`timing.jsonl` from real WAV durations. This populates the audio judges
(`StrictAffectPerceptionJudgeScorer`, `StrictDeliveryJudgeScorer`) and the
timing-based `NaturalnessScorer` with real signal.

Required environment variable: ANTHROPIC_API_KEY (used by both the LLM coach
and the synthetic customer). Optional: HUME_API_KEY for real TTS — when unset,
silent WAVs sized by a word-count heuristic are written so the plumbing still
exercises and audio judges degrade gracefully.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from rehearse.eval.customers.llm_customer import SyntheticCaller
from rehearse.eval.environments._audio import (
    DEFAULT_COACH_DESCRIPTION,
    read_transcript,
    silent_audio,
    synthesize_turns,
    timing_from_frames,
)
from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.tts_bridge import TTSProvider, get_default_provider
from rehearse.phases import PhaseBudgets
from rehearse.runtime import RuntimeHost, TextOnlyCoachAdapter

# Phase budgets compressed for text-only eval: no audio playback time, so
# LLM turns happen ~10× faster than real phone exchanges. Production budgets
# (60s/180s/60s) exhaust _MAX_TURNS before INTAKE can transition.
_EVAL_BUDGETS = PhaseBudgets(
    intake_seconds=20,
    practice_seconds=60,
    intake_min_dwell_seconds=10,
    practice_min_dwell_seconds=20,
)
from rehearse.storage import LocalFilesystemStore
from rehearse.transport import InMemoryTwoWayChannel


class RuntimeSandboxEnvironment:
    """Eval environment that runs the real runtime against an LLM customer."""

    name = "runtime-sandbox"
    version = "v1"

    def __init__(
        self,
        model_slots: dict[str, str] | None = None,
        *,
        tts_provider: TTSProvider | None = None,
    ) -> None:
        self.model_slots = model_slots or {}
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "RuntimeSandboxEnvironment: ANTHROPIC_API_KEY is required "
                "(used by the LLM coach and synthetic customer). "
                "Set it in .env before running --environment runtime-sandbox."
            )
        # TTS provider is resolved lazily so unit tests don't need HUME_API_KEY
        # to instantiate the environment.
        self._tts_provider_override = tts_provider
        self._tts_provider_resolved = False
        self._tts_provider: TTSProvider | None = None

    def _get_tts_provider(self) -> TTSProvider | None:
        if not self._tts_provider_resolved:
            self._tts_provider = (
                self._tts_provider_override
                if self._tts_provider_override is not None
                else get_default_provider()
            )
            self._tts_provider_resolved = True
        return self._tts_provider

    async def rollout(
        self,
        example: BenchmarkExample,
        run_dir: Path,
        rng_seed: int,
    ) -> RolloutResult:
        started = datetime.now()
        run_dir.mkdir(parents=True, exist_ok=True)
        session_id = f"eval-{example.id}"

        # Rename run_dir to match session_id so LocalFilesystemStore finds it.
        session_dir = run_dir.parent / session_id
        if not session_dir.exists():
            run_dir.rename(session_dir)
        else:
            session_dir = run_dir

        store = LocalFilesystemStore(session_dir.parent, public_base_url="http://localhost")

        coach = TextOnlyCoachAdapter()
        host = RuntimeHost(store, coach, budgets=_EVAL_BUDGETS)

        scenario = example.payload.get("scenario", example.payload)
        customer = SyntheticCaller(
            scenario=scenario,
            run_dir=session_dir,
        )

        transport = InMemoryTwoWayChannel()

        print(f"[runtime-sandbox] {example.id} starting rollout …", file=sys.stderr, flush=True)
        error: str | None = None
        caller_result = None
        try:
            _, caller_result = await asyncio.gather(
                host.run(session_id=session_id, transport=transport.runtime),
                customer.run(
                    transport=transport.customer,
                    runtime_phase=lambda: host.current_phase,
                ),
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc(file=sys.stderr)

        elapsed = (datetime.now() - started).total_seconds()
        status_str = f"error: {error}" if error else "ok"
        print(f"[runtime-sandbox] {example.id} rollout done ({elapsed:.0f}s) → {status_str}", file=sys.stderr, flush=True)

        # Post-hoc TTS + timing.jsonl. Only attempted when the rollout produced
        # a transcript; if the runtime crashed mid-call, skip audio.
        audio_payload: dict[str, Any] = {}
        if error is None:
            audio_payload = await self._synthesize_audio(example, session_dir)

        provenance = self._provenance(audio_payload)
        (session_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )

        completed = datetime.now()
        duration_ms = int((completed - started).total_seconds() * 1000)

        # Merge token usage from coach and synthetic caller.
        coach_u = getattr(coach, "token_usage", {}) or {}
        caller_u = (getattr(caller_result, "token_usage", {}) if caller_result else {}) or {}
        token_usage: dict[str, int] = {
            "coach_prompt_tokens": coach_u.get("prompt_tokens", 0),
            "coach_completion_tokens": coach_u.get("completion_tokens", 0),
            "customer_prompt_tokens": caller_u.get("prompt_tokens", 0),
            "customer_completion_tokens": caller_u.get("completion_tokens", 0),
        }
        token_usage["total_tokens"] = sum(token_usage.values())

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
            payload=audio_payload,
            token_usage=token_usage,
        )

    async def _synthesize_audio(
        self, example: BenchmarkExample, session_dir: Path
    ) -> dict[str, Any]:
        transcript_path = session_dir / "transcript.jsonl"
        if not transcript_path.exists():
            return {"tts_provider": "skipped-no-transcript"}

        frames = read_transcript(transcript_path)
        if not frames:
            return {"tts_provider": "skipped-empty-transcript"}

        scenario = example.payload.get("scenario") or {}
        user_desc = scenario.get("emotional_state") or "natural conversational tone"
        coach_desc = (
            example.payload.get("coach_description")
            or self.model_slots.get("coach_description")
            or DEFAULT_COACH_DESCRIPTION
        )
        silence_between_s = float(
            example.payload.get("silence_between_turns_s", 1.5) or 1.5
        )

        provider = self._get_tts_provider()
        n_turns = len(frames)
        print(f"[runtime-sandbox] synthesizing TTS for {n_turns} turns ({provider.name if provider else 'silent'}) …", file=sys.stderr, flush=True)
        if provider is not None:
            user_durations, coach_durations = await synthesize_turns(
                run_dir=session_dir,
                frames=frames,
                user_description=user_desc,
                coach_description=coach_desc,
                provider=provider,
            )
            tts_provider_name = getattr(provider, "name", "unknown")
        else:
            user_durations, coach_durations = silent_audio(session_dir, frames)
            tts_provider_name = "silent-fallback"

        timing_events = timing_from_frames(
            frames=frames,
            user_durations_s=user_durations,
            coach_durations_s=coach_durations,
            silence_between_s=silence_between_s,
        )
        if timing_events:
            (session_dir / "timing.jsonl").write_text(
                "\n".join(json.dumps(e) for e in timing_events) + "\n"
            )

        return {
            "tts_provider": tts_provider_name,
            "user_audio_durations_s": user_durations,
            "coach_audio_durations_s": coach_durations,
        }

    def _provenance(self, audio_payload: dict[str, Any]) -> dict[str, Any]:
        """Document the eval substitutions for this rollout."""
        return {
            "environment": self.name,
            "environment_version": self.version,
            "tier": "text-plus-tts",
            "runtime_kernel": "real",
            "intake_processor": "real (deterministic)",
            "persona_compiler": "real (deterministic)",
            "coach_voice_adapter": "TextOnlyCoachAdapter",
            "transport": "InMemoryTwoWayChannel",
            "synthetic_caller": "SyntheticCaller",
            "tts_provider": audio_payload.get("tts_provider", "skipped"),
            "anthropic_api_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "hume_api_key_set": bool(os.environ.get("HUME_API_KEY")),
            "audio_stub": audio_payload.get("tts_provider") == "silent-fallback",
        }
