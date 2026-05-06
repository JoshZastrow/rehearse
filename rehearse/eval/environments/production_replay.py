"""`production-replay` environment — re-materialize a recorded session.

Production sessions are bundled as a single mixed `audio.wav` plus a
`transcript.jsonl` whose user turns carry real `[ts_start, ts_end]`
intervals; coach turns carry only an arrival timestamp (`ts_start`).
The audio judges and naturalness scorer expect per-turn-per-role WAVs
plus `timing.jsonl` (the same shape the `audio-fixture` env produces).

This environment closes that gap:

  1. Reads the source bundle from the example payload's `session_dir`.
  2. Slices `audio.wav` into `audio/{user,coach}/turn_<N>.wav`. Coach
     turn end-times are inferred from the next turn's `ts_start` (or
     end-of-audio for the trailing turn) since production transcripts
     don't yet record TTS duration.
  3. Writes a `timing.jsonl` synthesized from those intervals.
  4. Copies the source `transcript.jsonl` into `run_dir` so scorers
     find it at the same relative path as in sandbox rollouts.

The result is a `RolloutResult.artifacts_dir` that drops cleanly into
`AffectPerceptionJudgeScorer`, `DeliveryJudgeScorer`, and
`NaturalnessScorer` without any scorer-side changes.
"""

from __future__ import annotations

import json
import shutil
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

from rehearse.eval.protocols import BenchmarkExample, RolloutResult


class ProductionReplayEnvironment:
    """Re-materialize a stored production session as a per-turn artifact bundle."""

    name = "production-replay"
    version = "v1"

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

        session_dir = Path(example.payload.get("session_dir") or "")
        if not session_dir.is_dir():
            return _error(
                example, run_dir, started, f"session_dir not found: {session_dir}"
            )

        transcript_src = session_dir / "transcript.jsonl"
        audio_src = session_dir / "audio.wav"
        if not (transcript_src.exists() and audio_src.exists()):
            return _error(
                example,
                run_dir,
                started,
                "session bundle missing transcript.jsonl or audio.wav",
            )

        # Copy transcript verbatim — schema (speaker/text) matches what
        # the offline judges expect.
        shutil.copyfile(transcript_src, run_dir / "transcript.jsonl")

        frames = _read_transcript(transcript_src)
        if not frames:
            return _error(example, run_dir, started, "transcript.jsonl is empty")

        with wave.open(str(audio_src), "rb") as wav:
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            n_channels = wav.getnchannels()
            audio_duration_s = wav.getnframes() / float(sample_rate)
            audio_frames = wav.readframes(wav.getnframes())

        intervals = _infer_intervals(frames, audio_duration_s)
        _write_per_turn_audio(
            run_dir,
            intervals=intervals,
            audio_frames=audio_frames,
            sample_rate=sample_rate,
            sample_width=sample_width,
            n_channels=n_channels,
        )
        timing_events = _timing_events(intervals)
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
            payload={
                "source_session_id": example.payload.get("session_id"),
                "source_session_dir": str(session_dir),
                "n_user_turns": sum(1 for i in intervals if i["role"] == "user"),
                "n_coach_turns": sum(1 for i in intervals if i["role"] == "coach"),
                "audio_duration_s": audio_duration_s,
            },
        )


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


def _infer_intervals(
    frames: list[dict[str, Any]],
    audio_duration_s: float,
) -> list[dict[str, Any]]:
    """Return per-turn intervals with role, turn_index_within_role, start, end.

    User turns carry true (ts_start, ts_end). Coach turns carry only
    ts_start; we infer ts_end as the next frame's ts_start (regardless of
    role) and clamp to audio duration. Frames missing ts_start are skipped.
    """
    starts: list[float] = []
    for f in frames:
        try:
            starts.append(float(f.get("ts_start")))
        except (TypeError, ValueError):
            starts.append(float("nan"))

    role_idx = {"user": 0, "coach": 0}
    intervals: list[dict[str, Any]] = []
    for i, frame in enumerate(frames):
        speaker = frame.get("speaker")
        if speaker not in ("user", "coach"):
            continue
        ts_start = starts[i]
        if ts_start != ts_start:  # NaN
            continue

        ts_end_raw = frame.get("ts_end")
        try:
            ts_end = float(ts_end_raw) if ts_end_raw is not None else float("nan")
        except (TypeError, ValueError):
            ts_end = float("nan")

        # Coach turns (and any turn where ts_end is missing or collapsed
        # onto ts_start) get their end inferred from the next valid start.
        if speaker == "coach" or ts_end != ts_end or ts_end <= ts_start:
            next_start = _next_start(starts, i + 1, audio_duration_s)
            ts_end = max(ts_start, min(next_start, audio_duration_s))

        ts_start = max(0.0, min(ts_start, audio_duration_s))
        ts_end = max(ts_start, min(ts_end, audio_duration_s))
        if ts_end <= ts_start:
            continue

        intervals.append(
            {
                "role": speaker,
                "turn_index": role_idx[speaker],
                "start_s": ts_start,
                "end_s": ts_end,
            }
        )
        role_idx[speaker] += 1
    return intervals


def _next_start(starts: list[float], from_idx: int, fallback: float) -> float:
    for j in range(from_idx, len(starts)):
        s = starts[j]
        if s == s:  # not NaN
            return s
    return fallback


def _write_per_turn_audio(
    run_dir: Path,
    *,
    intervals: list[dict[str, Any]],
    audio_frames: bytes,
    sample_rate: int,
    sample_width: int,
    n_channels: int,
) -> None:
    bytes_per_frame = sample_width * n_channels
    for interval in intervals:
        role = interval["role"]
        idx = interval["turn_index"]
        start_frame = int(interval["start_s"] * sample_rate)
        end_frame = int(interval["end_s"] * sample_rate)
        slice_bytes = audio_frames[
            start_frame * bytes_per_frame : end_frame * bytes_per_frame
        ]
        out_path = run_dir / "audio" / role / f"turn_{idx}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_path), "wb") as out:
            out.setnchannels(n_channels)
            out.setsampwidth(sample_width)
            out.setframerate(sample_rate)
            out.writeframes(slice_bytes)


def _timing_events(intervals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Emit `timing.jsonl` rows compatible with NaturalnessScorer.

    Uses absolute call-time positions (start_s/end_s) directly. The
    scorer reads `t_ms` and `duration_ms` and cares about ordering plus
    relative spacing — both preserved.
    """
    events: list[dict[str, Any]] = []
    for interval in intervals:
        start_ms = int(interval["start_s"] * 1000)
        end_ms = int(interval["end_s"] * 1000)
        events.append(
            {
                "turn_index": interval["turn_index"],
                "role": interval["role"],
                "event": "audio_start",
                "t_ms": start_ms,
            }
        )
        events.append(
            {
                "turn_index": interval["turn_index"],
                "role": interval["role"],
                "event": "audio_end",
                "t_ms": end_ms,
                "duration_ms": end_ms - start_ms,
            }
        )
    return events


def _error(
    example: BenchmarkExample,
    run_dir: Path,
    started: datetime,
    message: str,
) -> RolloutResult:
    completed = datetime.now()
    return RolloutResult(
        example_id=example.id,
        target_name=ProductionReplayEnvironment.name,
        target_version=ProductionReplayEnvironment.version,
        status="error",
        started_at=started,
        completed_at=completed,
        duration_ms=int((completed - started).total_seconds() * 1000),
        artifacts_dir=run_dir,
        error=message,
    )
