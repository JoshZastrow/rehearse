"""`NaturalnessScorer` — deterministic, timing-derived voice metrics.

Spec 4 of the v2026-05-06 eval system roadmap. Reads `timing.jsonl`
from `rollout.artifacts_dir` and emits three banded sub-metric rows:

  - `naturalness.interruption_rate` — how often the coach starts speaking
    while the user is still talking (overlaps ≥ 250ms).
  - `naturalness.silence_after_affect` — average silence between user
    turn end and coach turn start. (v0 simplification: scored over all
    user turns, not just affect-flagged ones — that refinement waits on
    the audio judge feeding affect flags into timing data.)
  - `naturalness.speech_rate_band` — coach words per minute, computed
    from transcript text and coach audio duration.

No LLM, no calibration ρ. Thresholds are pinned and versioned via
`thresholds_version`; bump on any threshold change so old and new
scores cannot mix in training data.

Bands (Appendix A of v2026-05-05 spec):

  interruption_rate (events/user_turn):
    0.0 ideal · ≤0.2 acceptable · >0.5 pathological
  silence_after_affect (s):
    1.5–4.0 ideal · 1.0–1.5 or 4.0–6.0 acceptable · else pathological
  speech_rate_band (wpm):
    130–170 ideal · 100–130 or 170–200 acceptable · else pathological
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.types import RubricScore

_DEFAULT_THRESHOLDS_VERSION = "naturalness-v1"

# Overlaps shorter than this don't count as interruption (backchannels).
_BACKCHANNEL_THRESHOLD_MS = 250


@dataclass(frozen=True)
class _TimingEvent:
    turn_index: int
    role: str
    event: str
    t_ms: int


def _band(value: float, *, ideal: tuple[float, float], acceptable: tuple[float, float]) -> float:
    """Three-band scoring — 1.0 ideal, 0.5 acceptable, 0.0 pathological."""
    lo_i, hi_i = ideal
    lo_a, hi_a = acceptable
    if lo_i <= value <= hi_i:
        return 1.0
    if lo_a <= value <= hi_a:
        return 0.5
    return 0.0


class NaturalnessScorer:
    """Score three deterministic naturalness sub-metrics over per-turn timing."""

    name = "naturalness"
    dimension = "naturalness"

    def __init__(
        self,
        *,
        thresholds_version: str = _DEFAULT_THRESHOLDS_VERSION,
    ) -> None:
        self.thresholds_version = thresholds_version

    async def score(
        self,
        example: BenchmarkExample,
        rollout: RolloutResult,
        run_id: str,
    ) -> list[RubricScore]:
        if rollout.status != "ok" or rollout.artifacts_dir is None:
            return self._all_zero(
                example, run_id, rationale=f"rollout {rollout.status}"
            )

        artifacts_dir = Path(rollout.artifacts_dir)
        timing_path = artifacts_dir / "timing.jsonl"
        if not timing_path.exists():
            return self._all_zero(
                example,
                run_id,
                rationale=f"timing.jsonl not found at {timing_path}",
                flags=["timing_missing"],
            )

        events = _parse_timing(timing_path.read_text().splitlines())

        interruption_rate = _interruption_rate(events)
        silence = _avg_silence_after_user_turn(events)
        wpm = _coach_speech_rate_wpm(events, artifacts_dir / "transcript.jsonl")

        rows: list[RubricScore] = []
        rows.append(
            self._row(
                example,
                run_id,
                dimension="naturalness.interruption_rate",
                value=_band(
                    interruption_rate,
                    ideal=(0.0, 0.0),
                    acceptable=(0.0, 0.2),
                ),
                rationale=f"{interruption_rate:.3f} events/user_turn",
            )
        )
        rows.append(
            self._row(
                example,
                run_id,
                dimension="naturalness.silence_after_affect",
                value=_band(
                    silence,
                    ideal=(1.5, 4.0),
                    acceptable=(1.0, 6.0),
                )
                if silence is not None
                else 0.0,
                rationale=(
                    f"{silence:.2f}s mean silence" if silence is not None else "no user turns"
                ),
                flags=[] if silence is not None else ["timing_missing"],
            )
        )
        rows.append(
            self._row(
                example,
                run_id,
                dimension="naturalness.speech_rate_band",
                value=_band(
                    wpm,
                    ideal=(130.0, 170.0),
                    acceptable=(100.0, 200.0),
                )
                if wpm is not None
                else 0.0,
                rationale=(
                    f"{wpm:.1f} wpm" if wpm is not None else "no coach speech detected"
                ),
                flags=[] if wpm is not None else ["timing_missing"],
            )
        )
        return rows

    def _row(
        self,
        example: BenchmarkExample,
        run_id: str,
        *,
        dimension: str,
        value: float,
        rationale: str,
        flags: list[str] | None = None,
    ) -> RubricScore:
        return RubricScore(
            run_id=run_id,
            example_id=example.id,
            dimension=dimension,
            value=value,
            scorer="deterministic",
            rationale=rationale,
            modality="timing",
            judge_prompt_version=self.thresholds_version,
            flags=list(flags or []),
        )

    def _all_zero(
        self,
        example: BenchmarkExample,
        run_id: str,
        *,
        rationale: str,
        flags: list[str] | None = None,
    ) -> list[RubricScore]:
        flags = flags or []
        return [
            self._row(
                example,
                run_id,
                dimension=dim,
                value=0.0,
                rationale=rationale,
                flags=flags,
            )
            for dim in (
                "naturalness.interruption_rate",
                "naturalness.silence_after_affect",
                "naturalness.speech_rate_band",
            )
        ]


def _parse_timing(lines: list[str]) -> list[_TimingEvent]:
    out: list[_TimingEvent] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        try:
            out.append(
                _TimingEvent(
                    turn_index=int(obj.get("turn_index", 0)),
                    role=str(obj["role"]),
                    event=str(obj["event"]),
                    t_ms=int(obj["t_ms"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _interruption_rate(events: list[_TimingEvent]) -> float:
    """Count coach starts ≥250ms before a user end / total user turns."""

    user_turns = _intervals(events, role="user")
    coach_starts = [e for e in events if e.role == "coach" and e.event == "audio_start"]

    if not user_turns:
        return 0.0

    interruptions = 0
    for cs in coach_starts:
        for start_ms, end_ms in user_turns:
            overlap_ms = end_ms - cs.t_ms
            if start_ms < cs.t_ms < end_ms and overlap_ms >= _BACKCHANNEL_THRESHOLD_MS:
                interruptions += 1
                break
    return interruptions / len(user_turns)


def _avg_silence_after_user_turn(events: list[_TimingEvent]) -> float | None:
    """Mean seconds between a user turn ending and the next coach turn starting."""

    user_ends_sorted = sorted(
        e.t_ms for e in events if e.role == "user" and e.event == "audio_end"
    )
    coach_starts_sorted = sorted(
        e.t_ms for e in events if e.role == "coach" and e.event == "audio_start"
    )

    if not user_ends_sorted or not coach_starts_sorted:
        return None

    silences: list[float] = []
    for ue in user_ends_sorted:
        next_coach = next((cs for cs in coach_starts_sorted if cs >= ue), None)
        if next_coach is None:
            continue
        silences.append((next_coach - ue) / 1000.0)
    if not silences:
        return None
    return sum(silences) / len(silences)


_WORD_RE = re.compile(r"\b\w+\b")


def _coach_speech_rate_wpm(
    events: list[_TimingEvent], transcript_path: Path
) -> float | None:
    """Coach words per minute = (total coach words) / (total coach speech seconds * 60)."""

    coach_intervals = _intervals(events, role="coach")
    total_speech_s = sum((end - start) for start, end in coach_intervals) / 1000.0
    if total_speech_s <= 0:
        return None

    if not transcript_path.exists():
        return None

    word_count = 0
    for line in transcript_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if row.get("speaker") != "coach":
            continue
        text = str(row.get("text", ""))
        word_count += len(_WORD_RE.findall(text))

    if word_count == 0:
        return None

    return (word_count / total_speech_s) * 60.0


def _intervals(events: list[_TimingEvent], *, role: str) -> list[tuple[int, int]]:
    """Pair audio_start / audio_end events for a given role into intervals."""

    starts: dict[int, int] = {}
    intervals: list[tuple[int, int]] = []
    for e in events:
        if e.role != role:
            continue
        if e.event == "audio_start":
            starts[e.turn_index] = e.t_ms
        elif e.event == "audio_end":
            start_ms = starts.pop(e.turn_index, None)
            if start_ms is not None:
                intervals.append((start_ms, e.t_ms))
    return intervals
