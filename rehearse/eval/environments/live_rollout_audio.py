"""`live-rollout-audio` environment.

Composes the LLM-driven sandbox dialogue with Hume Octave TTS so the
audio judges have real per-turn WAVs + a `timing.jsonl` to score.

Pipeline:
  1. Run `VoiceAgentSandboxEnvironment` with `customer_agent="llm"` and
     `coach_agent="llm"`. That writes `transcript.jsonl` and
     `conversation.jsonl` under `run_dir`.
  2. Read the transcript and synthesize each turn through the configured
     TTS provider (Hume Octave by default). Write
     `audio/{user,coach}/turn_<N>.wav` and a `timing.jsonl` derived from
     the real audio durations.

The audio + timing helpers live in `_audio.py` and are shared with
`runtime-sandbox`.

If no TTS provider is configured (`HUME_API_KEY` unset) the env falls
back to silent WAVs sized by a per-turn duration heuristic, so the
plumbing still exercises but the audio judges will degrade with
`audio_missing`-style flags.

Description hints for prosody come from `example.payload["scenario"]
["emotional_state"]` for user turns and a static "warm, steady, present"
for coach turns. Optional per-example `coach_description` overrides.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from rehearse.eval.environments._audio import (
    DEFAULT_COACH_DESCRIPTION,
    read_transcript,
    silent_audio,
    synthesize_turns,
    timing_from_frames,
)
from rehearse.eval.environments.voice_agent_sandbox import VoiceAgentSandboxEnvironment
from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.tts_bridge import TTSProvider, get_default_provider


class LiveRolloutAudioEnvironment:
    """LLM dialogue + post-hoc Hume TTS audio synthesis + timing.jsonl."""

    name = "live-rollout-audio"
    version = "v0"

    def __init__(
        self,
        model_slots: dict[str, str] | None = None,
        *,
        tts_provider: TTSProvider | None = None,
    ) -> None:
        self.model_slots = dict(model_slots or {})
        # Force LLM-driven agents by default; callers can still override
        # via `model_slots["customer_agent"]` / `["coach_agent"]`.
        self.model_slots.setdefault("customer_agent", "llm")
        self.model_slots.setdefault("coach_agent", "llm")
        self._sandbox = VoiceAgentSandboxEnvironment(model_slots=self.model_slots)
        self._tts_provider = tts_provider or get_default_provider()

    async def rollout(
        self,
        example: BenchmarkExample,
        run_dir: Path,
        rng_seed: int,
    ) -> RolloutResult:
        started = datetime.now()
        sandbox_result = await self._sandbox.rollout(example, run_dir, rng_seed)
        if sandbox_result.status != "ok":
            return sandbox_result.model_copy(
                update={
                    "target_name": self.name,
                    "target_version": self.version,
                }
            )

        transcript_path = run_dir / "transcript.jsonl"
        if not transcript_path.exists():
            completed = datetime.now()
            return RolloutResult(
                example_id=example.id,
                target_name=self.name,
                target_version=self.version,
                status="error",
                started_at=started,
                completed_at=completed,
                duration_ms=int((completed - started).total_seconds() * 1000),
                artifacts_dir=run_dir,
                error="sandbox produced no transcript.jsonl",
            )

        frames = read_transcript(transcript_path)
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

        if self._tts_provider is not None:
            user_durations, coach_durations = await synthesize_turns(
                run_dir=run_dir,
                frames=frames,
                user_description=user_desc,
                coach_description=coach_desc,
                provider=self._tts_provider,
            )
        else:
            user_durations, coach_durations = silent_audio(run_dir, frames)

        timing_events = timing_from_frames(
            frames=frames,
            user_durations_s=user_durations,
            coach_durations_s=coach_durations,
            silence_between_s=silence_between_s,
        )
        if timing_events:
            (run_dir / "timing.jsonl").write_text(
                "\n".join(json.dumps(e) for e in timing_events) + "\n"
            )

        completed = datetime.now()
        merged_payload = dict(sandbox_result.payload or {})
        merged_payload.update(
            {
                "user_audio_durations_s": user_durations,
                "coach_audio_durations_s": coach_durations,
                "tts_provider": getattr(self._tts_provider, "name", "silent-fallback"),
            }
        )
        return RolloutResult(
            example_id=example.id,
            target_name=self.name,
            target_version=self.version,
            status="ok",
            started_at=started,
            completed_at=completed,
            duration_ms=int((completed - started).total_seconds() * 1000),
            artifacts_dir=run_dir,
            payload=merged_payload,
        )
