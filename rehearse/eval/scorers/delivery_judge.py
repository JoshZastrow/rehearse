"""`DeliveryJudgeScorer` — did the coach's *delivery* match the moment?

Scores `delivery_quality` ∈ [0, 1] from per-turn user audio paired with
per-turn coach audio. Covers two facets in one number (folded for v0):
prosodic *attunement* (does coach prosody track user prosody?) and
*expressiveness* (range, animation, warmth regardless of user state).

Without both legs of audio, the dimension is not meaningfully scoreable —
the judge returns 0 with an `audio_missing` flag so the data card filter
can exclude the score from training data.

Spec reference: v2026-05-06 roadmap §6 (Mini-spec 2), v2026-05-05
parent spec §3.1 + §5.3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.scorers.audio_judge import AudioJudge, AudioJudgeError
from rehearse.types import RubricScore

_DEFAULT_PROMPT_VERSION = "delivery-quality-v1"

_SYSTEM_PROMPT = (
    "You are scoring a multi-turn coaching conversation on the *delivery* "
    "of the coach's responses — prosody, pacing, warmth, and expressive "
    "range — relative to the user's emotional state. You receive both the "
    "user's audio and the coach's audio, per turn. Do not score what was "
    "said; that is scored elsewhere.\n\n"
    "Score `delivery_quality` ∈ [0.0, 1.0] (continuous):\n"
    "  • 1.0 — Coach prosody has expressive range that meets the moment: "
    "warmer on disclosure, grounded on escalation, animated when "
    "celebrating. Pitch, rate, energy, and pause all track emotional "
    "shifts. Holds space.\n"
    "  • 0.5 — Audible delivery, flat range. Neutral, not load-bearing.\n"
    "  • 0.0 — Monotone, tonally miscalibrated, rushed, talks over the "
    "user, or fills every silence.\n\n"
    "Respond with ONLY this JSON object, nothing else:\n"
    '{"delivery_quality": {"score": <0.0-1.0>, "rationale": "<one '
    'sentence>"}}'
)


class DeliveryJudgeScorer:
    """Score `delivery_quality` ∈ [0, 1] using a multimodal LLM judge."""

    name = "delivery_judge"
    dimension = "delivery_quality"

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
        user_paths = _per_turn_audio(artifacts_dir, "user")
        coach_paths = _per_turn_audio(artifacts_dir, "coach")

        if not user_paths or not coach_paths:
            return [
                self._zero(
                    example,
                    run_id,
                    "delivery requires both user and coach audio",
                    flags=["audio_missing"],
                )
            ]

        transcript_path = artifacts_dir / "transcript.jsonl"
        rendered = (
            _render_transcript(transcript_path.read_text().splitlines())
            if transcript_path.exists()
            else "<no transcript>"
        )
        scenario = example.payload.get("scenario") or {}
        user_prompt = (
            f"Scenario:\n  Situation: {scenario.get('situation', '<unspecified>')}\n\n"
            f"Transcript (for grounding only — score the audio):\n{rendered}\n"
        )

        # Interleave user, coach for each turn so the model sees the pair.
        ordered_audio: list[Path] = []
        for u, c in zip(user_paths, coach_paths, strict=False):
            ordered_audio.extend([u, c])
        # Tail any remaining unmatched audio so nothing is lost.
        ordered_audio.extend(user_paths[len(coach_paths):])
        ordered_audio.extend(coach_paths[len(user_paths):])

        try:
            judge_output = await self.judge.judge(
                system=_SYSTEM_PROMPT,
                user=user_prompt,
                audio_paths=ordered_audio,
            )
        except AudioJudgeError as exc:
            return [self._zero(example, run_id, str(exc))]

        dim = judge_output.get("delivery_quality")
        if not isinstance(dim, dict) or "score" not in dim:
            return [
                self._zero(
                    example,
                    run_id,
                    "judge response missing delivery_quality",
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
                modality="audio",
                judge_prompt_version=self.prompt_version,
            )
        ]

    def _zero(
        self,
        example: BenchmarkExample,
        run_id: str,
        rationale: str,
        *,
        flags: list[str] | None = None,
    ) -> RubricScore:
        return RubricScore(
            run_id=run_id,
            example_id=example.id,
            dimension=self.dimension,
            value=0.0,
            scorer="llm_judge",
            rationale=rationale,
            modality="audio",
            judge_prompt_version=self.prompt_version,
            flags=list(flags or []),
        )


def _per_turn_audio(artifacts_dir: Path, role: str) -> list[Path]:
    role_dir = artifacts_dir / "audio" / role
    if not role_dir.exists():
        return []
    return sorted(role_dir.glob("turn_*.wav"), key=lambda p: p.stem)


def _render_transcript(jsonl_lines: list[str]) -> str:
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
