"""Audio judge machinery — Gemini client, base scorers, and calibrated scorers.

Four public scorer classes, two client classes:

  AudioJudge              — Gemini-backed multimodal LLM client (production)
  StubAudioJudge          — deterministic test double for AudioJudge

  AffectPerceptionJudgeScorer  — affect scorer, lenient 3-point rubric
  DeliveryJudgeScorer          — delivery scorer, lenient 3-point rubric

  CalibratedAffectScorer   — affect scorer, deduction-based rubric
  CalibratedDeliveryScorer — delivery scorer, deduction-based rubric

The lenient scorers are used by smoke and replay evals where the goal is
to verify the scoring pipeline runs end-to-end. The calibrated scorers are
used by the primary product eval (`voice-rollout-judges`) — their
deduction-based rubrics are tuned to land the aggregate reward near 0.5
across a mixed-difficulty scenario set, giving the signal range to
differentiate mediocre from genuinely good coaching.

Both scorer pairs score the same dimensions (`affect_perception`,
`delivery_quality`) and emit the same `RubricScore` shape. The only
difference is the judge prompt sent to Gemini.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Protocol

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.types import RubricScore

# ── Model selection ───────────────────────────────────────────────────────────

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_FALLBACK_MODEL = "gemini-2.5-flash"


def _default_model() -> str:
    return os.environ.get("REHEARSE_AUDIO_JUDGE_MODEL", _FALLBACK_MODEL)


# ── Judge client ──────────────────────────────────────────────────────────────

class AudioJudgeError(RuntimeError):
    """Raised when an audio-judge call fails or its output cannot be parsed."""


class _AudioClient(Protocol):
    async def generate(
        self,
        *,
        model: str,
        system: str,
        user: str,
        audio_paths: list[Path],
    ) -> str: ...


class AudioJudge:
    """Audio-aware judge: calls a multimodal LLM with audio + prompt, returns parsed JSON.

    Pass `client=` for tests; omit for production (a real Gemini client is
    constructed on first call).
    """

    def __init__(self, *, model: str | None = None, client: _AudioClient | None = None) -> None:
        self.model = model or _default_model()
        self._client = client

    async def judge(self, *, system: str, user: str, audio_paths: list[Path]) -> dict[str, Any]:
        client = self._client or self._lazy_default_client()
        try:
            text = await client.generate(model=self.model, system=system, user=user, audio_paths=list(audio_paths))
        except Exception as exc:
            raise AudioJudgeError(f"audio judge call failed: {exc}") from exc
        return _parse_json_block(text)

    def _lazy_default_client(self) -> _AudioClient:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise AudioJudgeError(
                "GEMINI_API_KEY not set; pass `client=` for tests or set the key for live runs"
            )
        return _GeminiAudioClient(api_key=api_key)


class StubAudioJudge:
    """Deterministic stand-in for `AudioJudge`. No network calls."""

    def __init__(self, *, response: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        if response is None and error is None:
            response = {}
        if response is not None and error is not None:
            raise ValueError("StubAudioJudge takes either response or error, not both")
        self._response = response
        self._error = error
        self.last_system: str | None = None
        self.last_user: str | None = None
        self.last_audio_paths: list[Path] = []

    async def judge(self, *, system: str, user: str, audio_paths: list[Path]) -> dict[str, Any]:
        self.last_system = system
        self.last_user = user
        self.last_audio_paths = list(audio_paths)
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


class _GeminiAudioClient:
    def __init__(self, *, api_key: str) -> None:
        self._api_key = api_key

    async def generate(self, *, model: str, system: str, user: str, audio_paths: list[Path]) -> str:
        try:
            from google import genai  # type: ignore[import-not-found]
            from google.genai import types  # type: ignore[import-not-found]
        except Exception as exc:
            raise AudioJudgeError("google-genai is not installed") from exc

        client = genai.Client(api_key=self._api_key)
        uploaded = [await _maybe_await(client.aio.files.upload(file=str(p))) for p in audio_paths if Path(p).exists()]
        if len(uploaded) != len(audio_paths):
            missing = [p for p in audio_paths if not Path(p).exists()]
            raise AudioJudgeError(f"audio files not found: {missing}")

        contents: list[Any] = [*uploaded, f"{system}\n\n{user}" if system else user]
        response = await _maybe_await(
            client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(max_output_tokens=8192, temperature=0.0),
            )
        )
        return getattr(response, "text", "") or ""


def _parse_json_block(text: str) -> dict[str, Any]:
    if not text:
        raise AudioJudgeError("audio judge returned empty text")
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(text)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    raise AudioJudgeError(f"could not parse JSON from audio judge output: {text[:300]!r}")


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


# ── Shared helpers ────────────────────────────────────────────────────────────

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


def _audio_paths(artifacts_dir: Path, role: str) -> list[Path]:
    role_dir = artifacts_dir / "audio" / role
    if not role_dir.exists():
        return []
    return sorted(role_dir.glob("turn_*.wav"), key=lambda p: p.stem)


def _interleaved_audio(artifacts_dir: Path) -> list[Path]:
    """Return user+coach WAVs interleaved by turn index."""
    user = _audio_paths(artifacts_dir, "user")
    coach = _audio_paths(artifacts_dir, "coach")
    result: list[Path] = []
    for i in range(max(len(user), len(coach))):
        if i < len(user):
            result.append(user[i])
        if i < len(coach):
            result.append(coach[i])
    return result


# ── Base scorers — lenient 3-point rubric ─────────────────────────────────────
# Used by smoke and replay evals. Easier to pass; validates the pipeline runs.

_AFFECT_PROMPT_V1 = "affect-perception-v1"
_AFFECT_SYSTEM_V1 = (
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
    '{"affect_perception": {"score": <0.0-1.0>, "rationale": "<one sentence>"}}'
)

_DELIVERY_PROMPT_V1 = "delivery-quality-v1"
_DELIVERY_SYSTEM_V1 = (
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
    '{"delivery_quality": {"score": <0.0-1.0>, "rationale": "<one sentence>"}}'
)


class AffectPerceptionJudgeScorer:
    """Score `affect_perception` ∈ [0, 1] — lenient rubric, for smoke and replay evals."""

    name = "affect_perception_judge"
    dimension = "affect_perception"

    def __init__(self, *, judge: AudioJudge | Any | None = None, prompt_version: str = _AFFECT_PROMPT_V1) -> None:
        self.judge = judge or AudioJudge()
        self.prompt_version = prompt_version

    async def score(self, example: BenchmarkExample, rollout: RolloutResult, run_id: str) -> list[RubricScore]:
        if rollout.status != "ok" or rollout.artifacts_dir is None:
            return [self._zero(example, run_id, f"rollout {rollout.status}")]
        artifacts_dir = Path(rollout.artifacts_dir)
        transcript_path = artifacts_dir / "transcript.jsonl"
        if not transcript_path.exists():
            return [self._zero(example, run_id, f"transcript not found at {transcript_path}")]

        transcript = _render_transcript(transcript_path.read_text().splitlines())
        user_audio = _audio_paths(artifacts_dir, "user")
        flags: list[str] = []
        modality = "audio+text"
        if not user_audio:
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
            f"Transcript:\n{transcript}\n"
        )
        try:
            output = await self.judge.judge(system=_AFFECT_SYSTEM_V1, user=user_prompt, audio_paths=user_audio)
        except AudioJudgeError as exc:
            return [self._zero(example, run_id, str(exc), flags=flags, modality=modality)]

        dim = output.get("affect_perception")
        if not isinstance(dim, dict) or "score" not in dim:
            return [self._zero(example, run_id, "judge missing affect_perception", flags=flags, modality=modality)]
        return [RubricScore(
            run_id=run_id, example_id=example.id, dimension=self.dimension,
            value=float(dim["score"]), scorer="llm_judge",
            rationale=str(dim.get("rationale", "")) or None,
            modality=modality,  # type: ignore[arg-type]
            judge_prompt_version=self.prompt_version, flags=flags,
        )]

    def _zero(self, example: BenchmarkExample, run_id: str, rationale: str, *, flags: list[str] | None = None, modality: str = "text") -> RubricScore:
        return RubricScore(
            run_id=run_id, example_id=example.id, dimension=self.dimension,
            value=0.0, scorer="llm_judge", rationale=rationale,
            modality=modality,  # type: ignore[arg-type]
            judge_prompt_version=self.prompt_version, flags=list(flags or []),
        )


class DeliveryJudgeScorer:
    """Score `delivery_quality` ∈ [0, 1] — lenient rubric, for smoke and replay evals."""

    name = "delivery_judge"
    dimension = "delivery_quality"

    def __init__(self, *, judge: AudioJudge | Any | None = None, prompt_version: str = _DELIVERY_PROMPT_V1) -> None:
        self.judge = judge or AudioJudge()
        self.prompt_version = prompt_version

    async def score(self, example: BenchmarkExample, rollout: RolloutResult, run_id: str) -> list[RubricScore]:
        if rollout.status != "ok" or rollout.artifacts_dir is None:
            return [self._zero(example, run_id, f"rollout {rollout.status}")]
        artifacts_dir = Path(rollout.artifacts_dir)
        user_audio = _audio_paths(artifacts_dir, "user")
        coach_audio = _audio_paths(artifacts_dir, "coach")
        if not user_audio or not coach_audio:
            return [self._zero(example, run_id, "delivery requires both user and coach audio", flags=["audio_missing"])]

        transcript_path = artifacts_dir / "transcript.jsonl"
        rendered = _render_transcript(transcript_path.read_text().splitlines()) if transcript_path.exists() else "<no transcript>"
        scenario = example.payload.get("scenario") or {}
        user_prompt = (
            f"Scenario:\n  Situation: {scenario.get('situation', '<unspecified>')}\n\n"
            f"Transcript (for grounding only — score the audio):\n{rendered}\n"
        )
        ordered = _interleaved_audio(artifacts_dir)
        try:
            output = await self.judge.judge(system=_DELIVERY_SYSTEM_V1, user=user_prompt, audio_paths=ordered)
        except AudioJudgeError as exc:
            return [self._zero(example, run_id, str(exc))]

        dim = output.get("delivery_quality")
        if not isinstance(dim, dict) or "score" not in dim:
            return [self._zero(example, run_id, "judge missing delivery_quality")]
        return [RubricScore(
            run_id=run_id, example_id=example.id, dimension=self.dimension,
            value=float(dim["score"]), scorer="llm_judge",
            rationale=str(dim.get("rationale", "")) or None,
            modality="audio", judge_prompt_version=self.prompt_version,
        )]

    def _zero(self, example: BenchmarkExample, run_id: str, rationale: str, *, flags: list[str] | None = None) -> RubricScore:
        return RubricScore(
            run_id=run_id, example_id=example.id, dimension=self.dimension,
            value=0.0, scorer="llm_judge", rationale=rationale,
            modality="audio", judge_prompt_version=self.prompt_version,
            flags=list(flags or []),
        )


# ── Calibrated scorers — deduction-based rubric ───────────────────────────────
# Used by the primary product eval (voice-rollout-judges). Start-high-then-deduct
# rubrics keep the aggregate reward near 0.5 across a mixed-difficulty set,
# giving enough range to differentiate mediocre from genuinely good coaching.

_AFFECT_PROMPT_CALIBRATED = "strict-affect-v1"
_AFFECT_SYSTEM_CALIBRATED = """You are scoring whether the coach correctly perceived the user's affect
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
{"affect_perception": {"score": <0.0-1.0>, "deductions": ["<each applied deduction or credit, briefly>"], "rationale": "<one sentence>"}}
"""

_DELIVERY_PROMPT_CALIBRATED = "strict-delivery-v1"
_DELIVERY_SYSTEM_CALIBRATED = """You are scoring whether the coach's audible delivery (prosody, pacing,
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
{"delivery_quality": {"score": <0.0-1.0>, "deductions": ["<each applied deduction or credit, briefly>"], "rationale": "<one sentence>"}}
"""


class CalibratedAffectScorer:
    """Score `affect_perception` ∈ [0, 1] — deduction rubric, for the primary product eval."""

    name = "strict_affect_judge"
    dimension = "affect_perception"

    def __init__(self, *, judge: AudioJudge | None = None, prompt_version: str = _AFFECT_PROMPT_CALIBRATED) -> None:
        self.judge = judge or AudioJudge()
        self.prompt_version = prompt_version

    async def score(self, example: BenchmarkExample, rollout: RolloutResult, run_id: str) -> list[RubricScore]:
        if rollout.status != "ok" or rollout.artifacts_dir is None:
            return [self._zero(example, run_id, f"rollout {rollout.status}")]
        artifacts_dir = Path(rollout.artifacts_dir)
        transcript_path = artifacts_dir / "transcript.jsonl"
        if not transcript_path.exists():
            return [self._zero(example, run_id, "no transcript.jsonl")]

        transcript = _render_transcript(transcript_path.read_text().splitlines())
        user_audio = _audio_paths(artifacts_dir, "user")
        flags: list[str] = []
        modality = "audio+text"
        if not user_audio:
            flags.append("audio_missing")
            modality = "text"

        scenario = example.payload.get("scenario") or {}
        user_prompt = (
            f"Scenario:\n  Situation: {scenario.get('situation', '')}\n"
            f"  Opening emotional state: {scenario.get('emotional_state', '')}\n\n"
            f"Transcript:\n{transcript}\n"
        )
        try:
            output = await self.judge.judge(system=_AFFECT_SYSTEM_CALIBRATED, user=user_prompt, audio_paths=user_audio)
        except AudioJudgeError as exc:
            return [self._zero(example, run_id, str(exc), flags=flags, modality=modality)]

        dim = output.get("affect_perception")
        if not isinstance(dim, dict) or "score" not in dim:
            return [self._zero(example, run_id, "judge missing affect_perception", flags=flags, modality=modality)]

        rationale = dim.get("rationale") or ""
        deductions = dim.get("deductions") or []
        if isinstance(deductions, list) and deductions:
            rationale = f"{rationale} | deductions: {'; '.join(map(str, deductions))}"
        return [RubricScore(
            run_id=run_id, example_id=example.id, dimension=self.dimension,
            value=float(dim["score"]), scorer="llm_judge", rationale=rationale or None,
            modality=modality,  # type: ignore[arg-type]
            judge_prompt_version=self.prompt_version, flags=flags,
        )]

    def _zero(self, example: BenchmarkExample, run_id: str, rationale: str, *, flags: list[str] | None = None, modality: str = "text") -> RubricScore:
        return RubricScore(
            run_id=run_id, example_id=example.id, dimension=self.dimension,
            value=0.0, scorer="llm_judge", rationale=rationale,
            modality=modality,  # type: ignore[arg-type]
            judge_prompt_version=self.prompt_version, flags=list(flags or []),
        )


class CalibratedDeliveryScorer:
    """Score `delivery_quality` ∈ [0, 1] — deduction rubric, for the primary product eval."""

    name = "strict_delivery_judge"
    dimension = "delivery_quality"

    def __init__(self, *, judge: AudioJudge | None = None, prompt_version: str = _DELIVERY_PROMPT_CALIBRATED) -> None:
        self.judge = judge or AudioJudge()
        self.prompt_version = prompt_version

    async def score(self, example: BenchmarkExample, rollout: RolloutResult, run_id: str) -> list[RubricScore]:
        if rollout.status != "ok" or rollout.artifacts_dir is None:
            return [self._zero(example, run_id, f"rollout {rollout.status}")]
        artifacts_dir = Path(rollout.artifacts_dir)
        transcript_path = artifacts_dir / "transcript.jsonl"
        if not transcript_path.exists():
            return [self._zero(example, run_id, "no transcript.jsonl")]

        transcript = _render_transcript(transcript_path.read_text().splitlines())
        audio_paths = _interleaved_audio(artifacts_dir)
        flags: list[str] = []
        modality = "audio"
        if not audio_paths:
            flags.append("audio_missing")
            modality = "text"

        scenario = example.payload.get("scenario") or {}
        user_prompt = (
            f"Scenario:\n  Situation: {scenario.get('situation', '')}\n"
            f"  Opening state: {scenario.get('emotional_state', '')}\n\n"
            f"Transcript:\n{transcript}\n\n"
            f"You will hear audio interleaved as user/coach/user/coach by turn.\n"
        )
        try:
            output = await self.judge.judge(system=_DELIVERY_SYSTEM_CALIBRATED, user=user_prompt, audio_paths=audio_paths)
        except AudioJudgeError as exc:
            return [self._zero(example, run_id, str(exc), flags=flags, modality=modality)]

        dim = output.get("delivery_quality")
        if not isinstance(dim, dict) or "score" not in dim:
            return [self._zero(example, run_id, "judge missing delivery_quality", flags=flags, modality=modality)]

        rationale = dim.get("rationale") or ""
        deductions = dim.get("deductions") or []
        if isinstance(deductions, list) and deductions:
            rationale = f"{rationale} | deductions: {'; '.join(map(str, deductions))}"
        return [RubricScore(
            run_id=run_id, example_id=example.id, dimension=self.dimension,
            value=float(dim["score"]), scorer="llm_judge", rationale=rationale or None,
            modality=modality,  # type: ignore[arg-type]
            judge_prompt_version=self.prompt_version, flags=flags,
        )]

    def _zero(self, example: BenchmarkExample, run_id: str, rationale: str, *, flags: list[str] | None = None, modality: str = "text") -> RubricScore:
        return RubricScore(
            run_id=run_id, example_id=example.id, dimension=self.dimension,
            value=0.0, scorer="llm_judge", rationale=rationale,
            modality=modality,  # type: ignore[arg-type]
            judge_prompt_version=self.prompt_version, flags=list(flags or []),
        )
