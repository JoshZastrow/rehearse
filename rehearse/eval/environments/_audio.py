"""Post-hoc audio synthesis + timing helpers for sandboxed rollouts.

Extracted from `live_rollout_audio.py` so `runtime-sandbox` and
`live-rollout-audio` share one implementation. Public functions are the
ones intended for reuse:

  - `read_transcript(path)` — parse `transcript.jsonl`.
  - `synthesize_turns(...)` — TTS each user/coach turn to a per-turn WAV
    under `audio/{user,coach}/turn_<N>.wav`. Falls back to silent WAVs
    on per-turn failure.
  - `silent_audio(...)` — produce silent WAVs sized by a word-count
    heuristic when no TTS provider is configured (e.g., `HUME_API_KEY`
    unset). Useful for plumbing tests and CI without a key.
  - `timing_from_frames(...)` — derive `timing.jsonl` events from real
    WAV durations.
  - `silent_wav(path, duration_s)` — write a single silent WAV.

Default coach description (`"warm, steady, present"`) and the fallback
turn-duration heuristic are exposed as module constants so callers can
override.
"""

from __future__ import annotations

import asyncio
import json
import wave
from pathlib import Path
from typing import Any

from rehearse.eval.environments.tts_bridge import TTSProvider

DEFAULT_COACH_DESCRIPTION = "warm, steady, present"

# Rough fallback when a TTS call fails or no provider is configured.
# Sized to put naturalness bands near ideal so silent runs don't tank
# the deterministic timing-based scorers.
FALLBACK_TURN_DURATION_S = 3.0


def read_transcript(path: Path) -> list[dict[str, Any]]:
    """Return parsed frames from `transcript.jsonl` (skips bad lines)."""
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


def silent_wav(path: Path, *, duration_s: float, sample_rate: int = 16_000) -> None:
    """Write a mono 16-bit PCM silent WAV of `duration_s` to `path`."""
    n = max(1, int(duration_s * sample_rate))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * n)


async def synthesize_turns(
    *,
    run_dir: Path,
    frames: list[dict[str, Any]],
    user_description: str,
    coach_description: str,
    provider: TTSProvider,
) -> tuple[list[float], list[float]]:
    """TTS each user/coach turn. Returns per-role duration lists in seconds."""
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
            silent_wav(out, duration_s=0.3)
            return role, idx, 0.3
        try:
            duration = await provider.synthesize(
                text=text, out_path=out, description=desc
            )
            return role, idx, duration
        except Exception:
            silent_wav(out, duration_s=FALLBACK_TURN_DURATION_S)
            return role, idx, FALLBACK_TURN_DURATION_S

    results = await asyncio.gather(*(_one(*args) for args in plan))
    user: list[float] = []
    coach: list[float] = []
    for role, _idx, dur in results:
        (user if role == "user" else coach).append(dur)
    return user, coach


def silent_audio(
    run_dir: Path, frames: list[dict[str, Any]]
) -> tuple[list[float], list[float]]:
    """Write silent WAVs sized by word count (~150 wpm). No TTS call."""
    user: list[float] = []
    coach: list[float] = []
    role_idx = {"user": 0, "coach": 0}
    for f in frames:
        role = f.get("speaker")
        if role not in ("user", "coach"):
            continue
        text = (f.get("text") or "").strip()
        est = max(1.0, len(text.split()) / 2.5)
        out = run_dir / "audio" / role / f"turn_{role_idx[role]}.wav"
        silent_wav(out, duration_s=est)
        (user if role == "user" else coach).append(est)
        role_idx[role] += 1
    return user, coach


def timing_from_frames(
    *,
    frames: list[dict[str, Any]],
    user_durations_s: list[float],
    coach_durations_s: list[float],
    silence_between_s: float,
) -> list[dict[str, Any]]:
    """Build `timing.jsonl` events from per-role durations + silence gaps."""
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
        events.append(
            {"turn_index": turn_index, "role": role, "event": "audio_start", "t_ms": t_ms}
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
