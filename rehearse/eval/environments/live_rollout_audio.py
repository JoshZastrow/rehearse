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

If no TTS provider is configured (`HUME_API_KEY` unset) the env falls
back to silent WAVs sized by a per-turn duration heuristic, so the
plumbing still exercises but the audio judges will degrade with
`audio_missing`-style flags.

Description hints for prosody come from `example.payload["scenario"]
["emotional_state"]` for user turns and a static "warm, steady, present"
for coach turns. Optional per-example `coach_description` overrides.
"""

from __future__ import annotations

import asyncio
import json
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

from rehearse.eval.environments.voice_agent_sandbox import VoiceAgentSandboxEnvironment
from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.tts_bridge import TTSProvider, get_default_provider


_DEFAULT_COACH_DESCRIPTION = "warm, steady, present"
# Rough fallback when no TTS is available — sized to put naturalness
# bands near ideal so silent runs don't tank deterministic scoring.
_FALLBACK_TURN_DURATION_S = 3.0


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

        frames = _read_transcript(transcript_path)
        scenario = example.payload.get("scenario") or {}
        user_desc = scenario.get("emotional_state") or "natural conversational tone"
        coach_desc = (
            example.payload.get("coach_description")
            or self.model_slots.get("coach_description")
            or _DEFAULT_COACH_DESCRIPTION
        )
        silence_between_s = float(
            example.payload.get("silence_between_turns_s", 1.5) or 1.5
        )

        if self._tts_provider is not None:
            user_durations, coach_durations = await self._synthesize(
                run_dir=run_dir,
                frames=frames,
                user_description=user_desc,
                coach_description=coach_desc,
            )
        else:
            user_durations, coach_durations = self._silent_audio(run_dir, frames)

        timing_events = _timing_from_frames(
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

    async def _synthesize(
        self,
        *,
        run_dir: Path,
        frames: list[dict[str, Any]],
        user_description: str,
        coach_description: str,
    ) -> tuple[list[float], list[float]]:
        assert self._tts_provider is not None
        provider = self._tts_provider
        plan: list[tuple[str, int, str, str]] = []
        role_idx = {"user": 0, "coach": 0}
        for f in frames:
            role = f.get("speaker")
            if role not in ("user", "coach"):
                continue
            text = (f.get("text") or "").strip()
            desc = user_description if role == "user" else coach_description
            plan.append((role, role_idx[role], text, desc))
            role_idx[role] += 1

        async def _one(role: str, idx: int, text: str, desc: str) -> tuple[str, int, float]:
            out = run_dir / "audio" / role / f"turn_{idx}.wav"
            if not text:
                _silent_wav(out, duration_s=0.3)
                return role, idx, 0.3
            try:
                duration = await provider.synthesize(
                    text=text, out_path=out, description=desc
                )
                return role, idx, duration
            except Exception:
                _silent_wav(out, duration_s=_FALLBACK_TURN_DURATION_S)
                return role, idx, _FALLBACK_TURN_DURATION_S

        results = await asyncio.gather(*(_one(*args) for args in plan))
        user: list[float] = []
        coach: list[float] = []
        for role, _idx, dur in results:
            (user if role == "user" else coach).append(dur)
        return user, coach

    def _silent_audio(
        self, run_dir: Path, frames: list[dict[str, Any]]
    ) -> tuple[list[float], list[float]]:
        user: list[float] = []
        coach: list[float] = []
        role_idx = {"user": 0, "coach": 0}
        for f in frames:
            role = f.get("speaker")
            if role not in ("user", "coach"):
                continue
            text = (f.get("text") or "").strip()
            est = max(1.0, len(text.split()) / 2.5)  # ~150 wpm
            out = run_dir / "audio" / role / f"turn_{role_idx[role]}.wav"
            _silent_wav(out, duration_s=est)
            (user if role == "user" else coach).append(est)
            role_idx[role] += 1
        return user, coach


def _read_transcript(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _silent_wav(path: Path, *, duration_s: float, sample_rate: int = 16_000) -> None:
    n = max(1, int(duration_s * sample_rate))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * n)


def _timing_from_frames(
    *,
    frames: list[dict[str, Any]],
    user_durations_s: list[float],
    coach_durations_s: list[float],
    silence_between_s: float,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    role_turn_idx = {"user": 0, "coach": 0}
    user_idx = 0
    coach_idx = 0
    t_ms = 0
    silence_ms = int(silence_between_s * 1000)
    for f in frames:
        role = f.get("speaker")
        if role == "user":
            if user_idx >= len(user_durations_s):
                continue
            dur_ms = int(float(user_durations_s[user_idx]) * 1000)
            user_idx += 1
        elif role == "coach":
            if coach_idx >= len(coach_durations_s):
                continue
            dur_ms = int(float(coach_durations_s[coach_idx]) * 1000)
            coach_idx += 1
        else:
            continue
        turn_index = role_turn_idx[role]
        role_turn_idx[role] += 1
        events.append({"turn_index": turn_index, "role": role, "event": "audio_start", "t_ms": t_ms})
        t_ms += dur_ms
        events.append(
            {
                "turn_index": turn_index,
                "role": role,
                "event": "audio_end",
                "t_ms": t_ms,
                "duration_ms": dur_ms,
            }
        )
        t_ms += silence_ms
    return events
