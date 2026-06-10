"""Judge-backed session reviewer.

Reads one prepared session directory, renders its transcript (falling back to
word alignments when the transcript is empty), and asks the judge backend to
approve/reject against the selection criteria. Approvals must carry a turning
point inside the audio duration — otherwise the review is coerced to rejected,
since an approval without a usable credit-assignment anchor is worthless to RL.

Writes review.json into the session directory.
"""

from __future__ import annotations

import json
import re
import wave
from pathlib import Path
from typing import Any

from rehearse.train.curation.types import SessionReview, TurningPoint

_DEFAULT_MODEL = "claude-opus-4-7"
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

_REVIEW_SYSTEM = (
    "You are a training-data curator for a voice coaching model. Review one recorded "
    "coaching session against the selection criteria and decide whether it enters the "
    "training corpus.\n"
    "Respond with ONLY this JSON object (no prose, no code fences):\n"
    "{\n"
    '  "decision": "approve" | "reject",\n'
    '  "turning_point": {"ts_start": <seconds>, "ts_end": <seconds>, '
    '"rationale": "<one sentence>"} | null,\n'
    '  "rationale": "<why this decision, citing the criteria>"\n'
    "}\n"
    "An approval MUST include a turning_point whose span lies within the audio duration; "
    "it anchors RL credit assignment. If you cannot locate one, reject."
)


class ReviewError(RuntimeError):
    """Raised when the judge response cannot be parsed."""


def _parse_json(text: str) -> dict[str, Any]:
    match = _JSON_BLOCK.search(text)
    if not match:
        raise ReviewError(f"no JSON object in judge response: {text[:200]}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ReviewError(f"malformed JSON in judge response: {exc}") from exc


def _audio_duration(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def _render_alignments(alignments: list) -> str:
    """Group word alignments into per-speaker utterance lines."""
    lines: list[str] = []
    speaker, words, start, end = None, [], 0.0, 0.0
    for word, (ws, we), label in alignments:
        if label != speaker and words:
            lines.append(f"[{start:.1f}-{end:.1f}] {speaker}: {' '.join(words)}")
            words = []
        if not words:
            speaker, start = label, ws
        words.append(word)
        end = we
    if words:
        lines.append(f"[{start:.1f}-{end:.1f}] {speaker}: {' '.join(words)}")
    return "\n".join(lines)


def _render_transcript(session_dir: Path) -> str:
    transcript_path = session_dir / "transcript.jsonl"
    lines: list[str] = []
    if transcript_path.exists():
        for raw in transcript_path.read_text().splitlines():
            if not raw.strip():
                continue
            d = json.loads(raw)
            if d.get("is_interim"):
                continue
            lines.append(f"[{d['ts_start']:.1f}-{d['ts_end']:.1f}] {d['speaker']}: {d['text']}")
    if lines:
        return "\n".join(lines)
    for name in ("audio_stereo.json", "audio.json"):
        path = session_dir / name
        if path.exists():
            alignments = json.loads(path.read_text()).get("alignments", [])
            if alignments:
                return _render_alignments(alignments)
    return "(no transcript available)"


def _render_prosody(session_dir: Path, limit: int = 50) -> str:
    path = session_dir / "prosody.jsonl"
    if not path.exists():
        return "(none)"
    lines = [line for line in path.read_text().splitlines() if line.strip()][:limit]
    return "\n".join(lines) if lines else "(none)"


def _build_user_prompt(session_dir: Path, criteria_text: str, duration: float) -> str:
    return (
        f"## Selection criteria\n{criteria_text}\n\n"
        f"## Session: {session_dir.name}\n"
        f"Audio duration: {duration:.1f}s\n\n"
        f"## Transcript\n{_render_transcript(session_dir)}\n\n"
        f"## Prosody events\n{_render_prosody(session_dir)}\n"
    )


def _validate(
    raw: dict[str, Any],
    *,
    session_id: str,
    criteria_version: int,
    duration: float,
    model: str | None,
) -> SessionReview:
    decision = raw.get("decision")
    rationale = str(raw.get("rationale", ""))
    tp_raw = raw.get("turning_point")
    turning_point = TurningPoint(**tp_raw) if tp_raw else None

    if decision == "approve":
        problem = None
        if turning_point is None:
            problem = "approval missing a turning point"
        elif not (0 <= turning_point.ts_start < turning_point.ts_end <= duration + 0.5):
            problem = (
                f"turning point [{turning_point.ts_start:.1f}, {turning_point.ts_end:.1f}] "
                f"outside audio duration {duration:.1f}s"
            )
        if problem is None:
            return SessionReview(
                session_id=session_id,
                decision="approved",
                turning_point=turning_point,
                rationale=rationale,
                criteria_version=criteria_version,
                audio_duration=duration,
                model=model,
            )
        decision, turning_point = "reject", None
        rationale = f"Coerced to rejected — {problem}. Judge rationale: {rationale}"

    if decision != "reject":
        raise ReviewError(f"unexpected decision {decision!r} for session {session_id}")
    return SessionReview(
        session_id=session_id,
        decision="rejected",
        turning_point=None,
        rationale=rationale,
        criteria_version=criteria_version,
        audio_duration=duration,
        model=model,
    )


class SessionReviewer:
    """Reviews prepared sessions via a judge backend (see judge_backend.backend_from_env)."""

    def __init__(self, *, backend: Any = None, model: str = _DEFAULT_MODEL, max_tokens: int = 1024):
        self.model = model
        self._backend = backend
        self._max_tokens = max_tokens

    def _get_backend(self) -> Any:
        if self._backend is None:
            from rehearse.eval.scorers.judge_backend import backend_from_env

            self._backend = backend_from_env(model=self.model, max_tokens=self._max_tokens)
        return self._backend

    async def review(
        self, session_dir: Path, *, criteria_text: str, criteria_version: int
    ) -> SessionReview:
        duration = _audio_duration(session_dir / "audio_stereo.wav")
        user = _build_user_prompt(session_dir, criteria_text, duration)
        text = await self._get_backend().complete(system=_REVIEW_SYSTEM, user=user)
        review = _validate(
            _parse_json(text),
            session_id=session_dir.name,
            criteria_version=criteria_version,
            duration=duration,
            model=self.model,
        )
        (session_dir / "review.json").write_text(review.model_dump_json(indent=2))
        return review
