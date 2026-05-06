"""`audio-fixture` environment — produces per-turn audio artifacts from a payload.

Useful for smoke-testing audio judges (`AffectPerceptionJudgeScorer`,
`DeliveryJudgeScorer`) without standing up a full sandbox or live phone
runtime. The example payload describes the trajectory shape:

    {
        "transcript": [
            {"speaker": "user", "text": "..."},
            {"speaker": "coach", "text": "..."},
            ...
        ],
        "user_audio_durations_s": [0.5, 0.5, ...],   # per user turn
        "coach_audio_durations_s": [0.6, 0.6, ...],  # per coach turn
    }

The environment writes:
    {run_dir}/transcript.jsonl
    {run_dir}/audio/user/turn_<N>.wav   (silent PCM16 WAVs)
    {run_dir}/audio/coach/turn_<N>.wav

Audio is synthesized as silent WAVs of the requested durations so the
fixture has no external asset dependencies. For real audio, swap to a
proper sandbox environment.
"""

from __future__ import annotations

import json
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

from rehearse.eval.protocols import BenchmarkExample, RolloutResult


class AudioFixtureEnvironment:
    """Synthesize per-turn audio artifacts from an example payload."""

    name = "audio-fixture"
    version = "v0"

    def __init__(self, model_slots: dict[str, str] | None = None) -> None:
        self.model_slots = model_slots or {}

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

        # Write transcript.jsonl.
        transcript_path = run_dir / "transcript.jsonl"
        transcript_path.write_text(
            "\n".join(json.dumps(_normalize_frame(f)) for f in transcript)
            + ("\n" if transcript else "")
        )

        # Synthesize per-turn audio for each role.
        _write_per_turn_audio(run_dir / "audio" / "user", user_durations)
        _write_per_turn_audio(run_dir / "audio" / "coach", coach_durations)

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


def _normalize_frame(frame: dict[str, Any]) -> dict[str, Any]:
    return {
        "speaker": frame.get("speaker", "user"),
        "text": frame.get("text", ""),
    }


def _write_per_turn_audio(out_dir: Path, durations_s: list[float]) -> None:
    if not durations_s:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, dur in enumerate(durations_s):
        path = out_dir / f"turn_{idx}.wav"
        _silent_wav(path, duration_s=float(dur))


def _silent_wav(path: Path, *, duration_s: float, sample_rate: int = 16_000) -> None:
    n_samples = max(1, int(duration_s * sample_rate))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * n_samples)
