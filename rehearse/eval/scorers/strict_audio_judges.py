"""Strict audio judges for `voice-rollout-judges` calibration.

Same shape as `AffectPerceptionJudgeScorer` / `DeliveryJudgeScorer` but
with deduction-based rubrics that pull scores down off the default 1.0
ceiling. Used to keep aggregate `weighted_reward` near 0.5 across a
mixed-difficulty scenario set.
"""

from __future__ import annotations

import json
from pathlib import Path

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.scorers.audio_judge import AudioJudge, AudioJudgeError
from rehearse.types import RubricScore


_AFFECT_PROMPT_VERSION = "strict-affect-v1"
_DELIVERY_PROMPT_VERSION = "strict-delivery-v1"


_AFFECT_SYSTEM = """You are scoring whether the coach correctly perceived the user's affect
across a multi-turn coaching call. You will see per-turn user audio plus
the transcript. Be a strict, demanding judge — the goal is to differentiate
"actually attuned" from "competent but generic empathy". Most coaches
deserve 0.4–0.7. Reserve 0.9+ for transcripts where the coach demonstrably
named the user's specific state in their own words and updated when it shifted.

Score `affect_perception` ∈ [0.0, 1.0] (continuous). Start from 0.85 and
deduct explicitly:
  -0.15: coach never named or mirrored the user's specific opening state.
  -0.15: coach offered generic empathy that any persona could have given.
  -0.10 each: missed an audible shift in user state (e.g. user softens or
              hardens between turns and the coach proceeds as if nothing changed).
  -0.10 each: coach's response felt scripted/templated rather than reactive.
  -0.05 each: coach hedged ("you may be feeling…") instead of naming.
  +0.05 once: coach named a non-obvious state (numb, brittle, conflicted)
              accurately and the user accepted the read.
  +0.05 once: coach updated the read mid-call when the user shifted.

Respond with ONLY this JSON, nothing else:
{"affect_perception": {"score": <0.0-1.0>, "deductions": ["<each applied
deduction or credit, briefly>"], "rationale": "<one sentence>"}}
"""

_DELIVERY_SYSTEM = """You are scoring whether the coach's audible delivery (prosody, pacing,
warmth, pauses) matched the moment across a multi-turn coaching call.
You will hear both the user's audio and the coach's audio per turn. Be a
strict, demanding judge — generic-warm-AI-voice should not score above
0.6. Most coaches deserve 0.4–0.7. Reserve 0.9+ for transcripts where the
coach varied their delivery to fit specific moments (slowing into grief,
energizing into a breakthrough, holding silence after a hard line).

Score `delivery_quality` ∈ [0.0, 1.0] (continuous). Start from 0.80 and
deduct explicitly:
  -0.15: coach used the same flat delivery across all turns regardless of
         what the user was doing.
  -0.10 each: rushed through a moment that called for slowness.
  -0.10 each: spoke too softly/slowly when the user needed energy or
              redirection.
  -0.10 each: filler/uptalk/sing-song that read as AI-default-warm.
  -0.05 each: coach interrupted or stepped on the user's pause.
  +0.05 once: coach matched a specific moment's energy (slowed for grief,
              energized for excitement) audibly, not just lexically.
  +0.05 once: coach used silence/pacing as a tool rather than always filling.

Respond with ONLY this JSON, nothing else:
{"delivery_quality": {"score": <0.0-1.0>, "deductions": ["<each applied
deduction or credit, briefly>"], "rationale": "<one sentence>"}}
"""


def _render_transcript(lines: list[str]) -> str:
    out: list[str] = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            f = json.loads(line)
        except json.JSONDecodeError:
            continue
        spk = f.get("speaker", "?")
        label = "USER" if spk == "user" else "COACH" if spk == "coach" else spk.upper()
        out.append(f"[{i}] {label}: {(f.get('text') or '').strip()}")
    return "\n".join(out)


def _user_audio(artifacts_dir: Path) -> list[Path]:
    d = artifacts_dir / "audio" / "user"
    if not d.exists():
        return []
    return sorted(d.glob("turn_*.wav"), key=lambda p: p.stem)


def _both_audio(artifacts_dir: Path) -> list[Path]:
    user = _user_audio(artifacts_dir)
    coach_dir = artifacts_dir / "audio" / "coach"
    coach = (
        sorted(coach_dir.glob("turn_*.wav"), key=lambda p: p.stem)
        if coach_dir.exists()
        else []
    )
    interleaved: list[Path] = []
    for i in range(max(len(user), len(coach))):
        if i < len(user):
            interleaved.append(user[i])
        if i < len(coach):
            interleaved.append(coach[i])
    return interleaved


class StrictAffectPerceptionJudgeScorer:
    name = "strict_affect_judge"
    dimension = "affect_perception"

    def __init__(
        self,
        *,
        judge: AudioJudge | None = None,
        prompt_version: str = _AFFECT_PROMPT_VERSION,
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
            return [self._zero(example, run_id, "no transcript.jsonl")]
        transcript = _render_transcript(transcript_path.read_text().splitlines())
        audio_paths = _user_audio(artifacts_dir)
        flags: list[str] = []
        modality = "audio+text"
        if not audio_paths:
            flags.append("audio_missing")
            modality = "text"
        scenario = example.payload.get("scenario") or {}
        user_prompt = (
            f"Scenario:\n  Situation: {scenario.get('situation','')}\n"
            f"  Opening emotional state: {scenario.get('emotional_state','')}\n\n"
            f"Transcript:\n{transcript}\n"
        )
        try:
            output = await self.judge.judge(
                system=_AFFECT_SYSTEM, user=user_prompt, audio_paths=audio_paths
            )
        except AudioJudgeError as exc:
            return [self._zero(example, run_id, str(exc), flags=flags, modality=modality)]
        dim = output.get("affect_perception")
        if not isinstance(dim, dict) or "score" not in dim:
            return [self._zero(example, run_id, "judge missing affect_perception", flags=flags, modality=modality)]
        rationale = dim.get("rationale") or ""
        deductions = dim.get("deductions") or []
        if isinstance(deductions, list) and deductions:
            rationale = f"{rationale} | deductions: {'; '.join(map(str, deductions))}"
        return [
            RubricScore(
                run_id=run_id,
                example_id=example.id,
                dimension=self.dimension,
                value=float(dim["score"]),
                scorer="llm_judge",
                rationale=rationale or None,
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


class StrictDeliveryJudgeScorer:
    name = "strict_delivery_judge"
    dimension = "delivery_quality"

    def __init__(
        self,
        *,
        judge: AudioJudge | None = None,
        prompt_version: str = _DELIVERY_PROMPT_VERSION,
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
            return [self._zero(example, run_id, "no transcript.jsonl")]
        transcript = _render_transcript(transcript_path.read_text().splitlines())
        audio_paths = _both_audio(artifacts_dir)
        flags: list[str] = []
        modality = "audio"
        if not audio_paths:
            flags.append("audio_missing")
            modality = "text"
        scenario = example.payload.get("scenario") or {}
        user_prompt = (
            f"Scenario:\n  Situation: {scenario.get('situation','')}\n"
            f"  Opening state: {scenario.get('emotional_state','')}\n\n"
            f"Transcript:\n{transcript}\n\n"
            f"You will hear audio interleaved as user/coach/user/coach by turn.\n"
        )
        try:
            output = await self.judge.judge(
                system=_DELIVERY_SYSTEM, user=user_prompt, audio_paths=audio_paths
            )
        except AudioJudgeError as exc:
            return [self._zero(example, run_id, str(exc), flags=flags, modality=modality)]
        dim = output.get("delivery_quality")
        if not isinstance(dim, dict) or "score" not in dim:
            return [self._zero(example, run_id, "judge missing delivery_quality", flags=flags, modality=modality)]
        rationale = dim.get("rationale") or ""
        deductions = dim.get("deductions") or []
        if isinstance(deductions, list) and deductions:
            rationale = f"{rationale} | deductions: {'; '.join(map(str, deductions))}"
        return [
            RubricScore(
                run_id=run_id,
                example_id=example.id,
                dimension=self.dimension,
                value=float(dim["score"]),
                scorer="llm_judge",
                rationale=rationale or None,
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
