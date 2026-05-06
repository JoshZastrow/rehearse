"""`AffectPerceptionJudgeScorer` — does the coach correctly read the user's affect?

Reads `transcript.jsonl` and per-turn user audio (`audio/user/turn_<N>.wav`)
from `rollout.artifacts_dir`, sends both to a multimodal LLM judge
(default Gemini 2.5), and emits one `affect_perception` `RubricScore`.

When user audio is missing, the scorer degrades to a *text-only* read of
the transcript and tags the score with `audio_missing` + `modality="text"`
so downstream filters (data card, calibration gate) can exclude it from
training data.

Spec reference: v2026-05-06 roadmap §6 (Mini-spec 2), v2026-05-05
parent spec §3.1 + §5.2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.scorers.audio_judge import AudioJudge, AudioJudgeError
from rehearse.types import RubricScore

_DEFAULT_PROMPT_VERSION = "affect-perception-v1"

_SYSTEM_PROMPT = (
    "You are scoring a multi-turn coaching conversation on whether the "
    "coach correctly perceived the user's emotional state. The user came "
    "in with a specific emotional opening and may have shifted during the "
    "call. You will receive the user's audio (per-turn) and the full "
    "transcript.\n\n"
    "Score `affect_perception` ∈ [0.0, 1.0] (continuous):\n"
    "  • 1.0 — Coach demonstrably reads the user's affect (named, mirrored, "
    "or acted on) and updates as the user shifts.\n"
    "  • 0.5 — Generic empathy; not grounded in the user's actual state. "
    "Doesn't update on shifts.\n"
    "  • 0.0 — Misreads, escalates, dismisses, or ignores affect.\n\n"
    "Respond with ONLY this JSON object, nothing else:\n"
    '{"affect_perception": {"score": <0.0-1.0>, "rationale": "<one '
    'sentence>"}}'
)


class AffectPerceptionJudgeScorer:
    """Score `affect_perception` ∈ [0, 1] using a multimodal LLM judge."""

    name = "affect_perception_judge"
    dimension = "affect_perception"

    def __init__(
        self,
        *,
        judge: AudioJudge | Any | None = None,
        prompt_version: str = _DEFAULT_PROMPT_VERSION,
    ) -> None:
        self.judge = judge or AudioJudge()
        self.prompt_version = prompt_version

    async def score(
        self,
        example: BenchmarkExample,
        rollout: RolloutResult,
        run_id: str,
    ) -> list[RubricScore]:
        if rollout.status != "ok" or rollout.artifacts_dir is None:
            return [self._zero(example, run_id, f"rollout {rollout.status}")]

        artifacts_dir = Path(rollout.artifacts_dir)
        transcript_path = artifacts_dir / "transcript.jsonl"
        if not transcript_path.exists():
            return [
                self._zero(
                    example, run_id, f"transcript not found at {transcript_path}"
                )
            ]

        rendered_transcript = _render_transcript(
            transcript_path.read_text().splitlines()
        )

        user_audio_paths = _user_audio_paths(artifacts_dir)
        flags: list[str] = []
        modality: str = "audio+text"
        if not user_audio_paths:
            flags.append("audio_missing")
            modality = "text"

        scenario = example.payload.get("scenario") or {}
        opening_emotion = (
            example.expected.get("opening_emotion")
            or example.payload.get("opening_emotion")
            or scenario.get("emotional_state")
            or "unknown"
        )
        user_prompt = (
            f"Scenario:\n  Situation: {scenario.get('situation', '<unspecified>')}\n"
            f"  User's opening emotional state: {opening_emotion}\n\n"
            f"Transcript:\n{rendered_transcript}\n"
        )

        try:
            judge_output = await self.judge.judge(
                system=_SYSTEM_PROMPT,
                user=user_prompt,
                audio_paths=user_audio_paths,
            )
        except AudioJudgeError as exc:
            return [self._zero(example, run_id, str(exc), flags=flags, modality=modality)]

        dim = judge_output.get("affect_perception")
        if not isinstance(dim, dict) or "score" not in dim:
            return [
                self._zero(
                    example,
                    run_id,
                    "judge response missing affect_perception",
                    flags=flags,
                    modality=modality,
                )
            ]

        return [
            RubricScore(
                run_id=run_id,
                example_id=example.id,
                dimension=self.dimension,
                value=float(dim["score"]),
                scorer="llm_judge",
                rationale=str(dim.get("rationale", "")) or None,
                modality=modality,  # type: ignore[arg-type]
                judge_prompt_version=self.prompt_version,
                flags=flags,
            )
        ]

    def _zero(
        self,
        example: BenchmarkExample,
        run_id: str,
        rationale: str,
        *,
        flags: list[str] | None = None,
        modality: str = "text",
    ) -> RubricScore:
        return RubricScore(
            run_id=run_id,
            example_id=example.id,
            dimension=self.dimension,
            value=0.0,
            scorer="llm_judge",
            rationale=rationale,
            modality=modality,  # type: ignore[arg-type]
            judge_prompt_version=self.prompt_version,
            flags=list(flags or []),
        )


def _render_transcript(jsonl_lines: list[str]) -> str:
    """Render JSONL TranscriptFrames as `[idx] SPEAKER: text`."""
    out: list[str] = []
    for idx, line in enumerate(jsonl_lines):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        speaker = row.get("speaker", "unknown")
        label = "USER" if speaker == "user" else "COACH" if speaker == "coach" else speaker.upper()
        out.append(f"[{idx}] {label}: {row.get('text', '').strip()}")
    return "\n".join(out)


def _user_audio_paths(artifacts_dir: Path) -> list[Path]:
    """Return per-turn user-audio WAV paths, ordered by turn index."""
    user_dir = artifacts_dir / "audio" / "user"
    if not user_dir.exists():
        return []
    return sorted(user_dir.glob("turn_*.wav"), key=lambda p: p.stem)
