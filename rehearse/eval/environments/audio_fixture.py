"""`audio-fixture` environment — produces per-turn audio + timing from a payload.

Useful for smoke-testing voice scorers (`AffectPerceptionJudgeScorer`,
`DeliveryJudgeScorer`, `NaturalnessScorer`) without standing up a full
sandbox or live phone runtime. The example payload describes the
trajectory shape:

    {
        "transcript": [
            {"speaker": "user", "text": "...", "description": "anxious"},
            {"speaker": "coach", "text": "...", "description": "warm, steady"},
            ...
        ],
        "user_audio_durations_s": [0.5, 0.5, ...],   # silent-fallback only
        "coach_audio_durations_s": [0.6, 0.6, ...],  # silent-fallback only
        "silence_between_turns_s": 0.0,              # optional pad between turns
    }

The environment writes:
    {run_dir}/transcript.jsonl
    {run_dir}/audio/user/turn_<N>.wav
    {run_dir}/audio/coach/turn_<N>.wav
    {run_dir}/timing.jsonl

Audio synthesis path: if a TTS provider is configured (see
`rehearse.eval.environments.tts_bridge.get_default_provider`), each turn's text is
synthesized to a real WAV and timing is computed from the *actual*
audio duration. If no provider is configured, the environment falls
back to silent WAVs sized by `*_audio_durations_s` so CI smoke runs
stay hermetic.
"""

from __future__ import annotations

import asyncio
import json
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.environments.tts_bridge import TTSProvider, get_default_provider


class AudioFixtureEnvironment:
    """Synthesize per-turn audio artifacts + timing.jsonl from an example payload."""

    name = "audio-fixture"
    version = "v1"

    def __init__(
        self,
        model_slots: dict[str, str] | None = None,
        *,
        tts_provider: TTSProvider | None = None,
    ) -> None:
        self.model_slots = model_slots or {}
        # If no provider is injected, resolve from env (None = silent fallback).
        # To force silent even when HUME_API_KEY is set, export
        # REHEARSE_TTS_PROVIDER=none.
        self._tts_provider = tts_provider or get_default_provider()

    async def rollout(
        self,
        example: BenchmarkExample,
        run_dir: Path,
        rng_seed: int,
    ) -> RolloutResult:
        started = datetime.now()
        run_dir.mkdir(parents=True, exist_ok=True)

        transcript = example.payload.get("transcript") or []
        user_durations = example.payload.get("user_audio_durations_s") or []
        coach_durations = example.payload.get("coach_audio_durations_s") or []
        silence_between_s = float(
            example.payload.get("silence_between_turns_s", 0.0) or 0.0
        )

        # Write transcript.jsonl.
        transcript_path = run_dir / "transcript.jsonl"
        transcript_path.write_text(
            "\n".join(json.dumps(_normalize_frame(f)) for f in transcript)
            + ("\n" if transcript else "")
        )

        # Synthesize per-turn audio. If a TTS provider is configured,
        # produce real speech and use the resulting durations. Otherwise
        # fall back to silent WAVs sized by the declared durations.
        if self._tts_provider is not None:
            real_user_durations, real_coach_durations = await self._synthesize_turns(
                run_dir=run_dir,
                transcript=transcript,
            )
            timing_user_durations = real_user_durations
            timing_coach_durations = real_coach_durations
        else:
            _write_silent_per_turn_audio(run_dir / "audio" / "user", user_durations)
            _write_silent_per_turn_audio(run_dir / "audio" / "coach", coach_durations)
            timing_user_durations = list(user_durations)
            timing_coach_durations = list(coach_durations)

        timing_events = _timing_from_transcript(
            transcript=transcript,
            user_durations_s=timing_user_durations,
            coach_durations_s=timing_coach_durations,
            silence_between_s=silence_between_s,
        )
        if timing_events:
            (run_dir / "timing.jsonl").write_text(
                "\n".join(json.dumps(e) for e in timing_events) + "\n"
            )

        completed = datetime.now()
        return RolloutResult(
            example_id=example.id,
            target_name=self.name,
            target_version=self.version,
            status="ok",
            started_at=started,
            completed_at=completed,
            duration_ms=int((completed - started).total_seconds() * 1000),
            artifacts_dir=run_dir,
        )

    async def _synthesize_turns(
        self,
        *,
        run_dir: Path,
        transcript: list[dict[str, Any]],
    ) -> tuple[list[float], list[float]]:
        """Synthesize each turn in parallel; return per-role real durations."""
        assert self._tts_provider is not None
        provider = self._tts_provider

        plan: list[tuple[str, int, dict[str, Any]]] = []
        role_idx = {"user": 0, "coach": 0}
        for frame in transcript:
            role = frame.get("speaker")
            if role not in ("user", "coach"):
                continue
            plan.append((role, role_idx[role], frame))
            role_idx[role] += 1

        async def _one(role: str, idx: int, frame: dict[str, Any]) -> tuple[str, int, float]:
            out = run_dir / "audio" / role / f"turn_{idx}.wav"
            text = (frame.get("text") or "").strip()
            if not text:
                _silent_wav(out, duration_s=0.2)
                return role, idx, 0.2
            duration = await provider.synthesize(
                text=text,
                out_path=out,
                description=frame.get("description"),
            )
            return role, idx, duration

        results = await asyncio.gather(*(_one(r, i, f) for r, i, f in plan))

        user_durations: list[float] = []
        coach_durations: list[float] = []
        for role, _idx, duration in results:
            (user_durations if role == "user" else coach_durations).append(duration)
        return user_durations, coach_durations


def _normalize_frame(frame: dict[str, Any]) -> dict[str, Any]:
    return {
        "speaker": frame.get("speaker", "user"),
        "text": frame.get("text", ""),
    }


def _write_silent_per_turn_audio(out_dir: Path, durations_s: list[float]) -> None:
    if not durations_s:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, dur in enumerate(durations_s):
        path = out_dir / f"turn_{idx}.wav"
        _silent_wav(path, duration_s=float(dur))


def _silent_wav(path: Path, *, duration_s: float, sample_rate: int = 16_000) -> None:
    n_samples = max(1, int(duration_s * sample_rate))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * n_samples)


def _timing_from_transcript(
    *,
    transcript: list[dict[str, Any]],
    user_durations_s: list[float],
    coach_durations_s: list[float],
    silence_between_s: float = 0.0,
) -> list[dict[str, Any]]:
    """Sequence transcript turns into timing events using per-role durations.

    The Nth user transcript frame consumes user_durations_s[role_turn_idx];
    the Nth coach frame consumes coach_durations_s[role_turn_idx]. Turns
    are placed back-to-back on the timeline, with `silence_between_s`
    inserted between consecutive turns. Frames for which no duration is
    configured are skipped (no audio, no timing event).
    """
    events: list[dict[str, Any]] = []
    role_turn_idx = {"user": 0, "coach": 0}
    user_idx = 0
    coach_idx = 0
    t_ms = 0
    silence_ms = int(silence_between_s * 1000)

    for frame in transcript:
        role = frame.get("speaker")
        if role not in ("user", "coach"):
            continue
        if role == "user":
            if user_idx >= len(user_durations_s):
                continue
            dur_ms = int(float(user_durations_s[user_idx]) * 1000)
            user_idx += 1
        else:
            if coach_idx >= len(coach_durations_s):
                continue
            dur_ms = int(float(coach_durations_s[coach_idx]) * 1000)
            coach_idx += 1

        turn_index = role_turn_idx[role]
        role_turn_idx[role] += 1

        events.append(
            {
                "turn_index": turn_index,
                "role": role,
                "event": "audio_start",
                "t_ms": t_ms,
            }
        )
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
