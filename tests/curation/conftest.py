"""Shared fixtures for session curation tests."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np


class FakeJudgeBackend:
    """Judge backend stub: records calls, answers via a (system, user) -> str callable."""

    def __init__(self, respond):
        self._respond = respond
        self.calls: list[dict[str, str]] = []

    async def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        return self._respond(system, user)


class FakeTrainer:
    """Planted loss model: loss = base + sum(harm of included sessions) + seed jitter."""

    def __init__(self, *, session_ids, harm=None, base=3.0, seed_jitter=None):
        self.session_ids = list(session_ids)
        self._harm = harm or {}
        self._base = base
        self._jitter = seed_jitter or {}
        self.calls: list[dict] = []

    def run(self, *, excluded_ids, seed, max_steps, kind):
        included = [s for s in self.session_ids if s not in excluded_ids]
        loss = (
            self._base
            + sum(self._harm.get(s, 0.0) for s in included)
            + self._jitter.get(seed, 0.0)
        )
        self.calls.append({"excluded": sorted(excluded_ids), "seed": seed, "kind": kind})
        return loss


class FakeVolumeClient:
    """Records pushed (volume_path, bytes) pairs instead of hitting Modal."""

    def __init__(self):
        self.pushed: dict[str, bytes] = {}

    def push(self, files: list[tuple[str, bytes]]) -> None:
        self.pushed.update(dict(files))


def approve_response(ts_start: float = 5.0, ts_end: float = 8.0) -> str:
    return json.dumps(
        {
            "decision": "approve",
            "turning_point": {
                "ts_start": ts_start,
                "ts_end": ts_end,
                "rationale": "Caller shifts from deflection to ownership.",
            },
            "rationale": "Clear arc with a usable credit-assignment moment.",
        }
    )


def reject_response() -> str:
    return json.dumps(
        {
            "decision": "reject",
            "turning_point": None,
            "rationale": "Provider breaks persona; no coaching arc.",
        }
    )


def make_session_dir(
    root: Path,
    session_id: str,
    *,
    duration_sec: float = 20.0,
    transcript_lines: list[dict] | None = None,
    alignments: list | None = None,
) -> Path:
    """Build a minimal prepared session dir: audio_stereo.wav (+json), transcript.jsonl."""
    session_dir = root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    sr = 16000
    n = int(duration_sec * sr)
    silence = np.zeros(n, dtype=np.int16)
    stereo = np.stack([silence, silence], axis=1)
    with wave.open(str(session_dir / "audio_stereo.wav"), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(stereo.tobytes())

    if alignments is None:
        alignments = [
            ["hello", [0.5, 1.0], "provider"],
            ["i", [2.0, 2.2], "caller"],
            ["froze", [2.2, 2.6], "caller"],
        ]
    (session_dir / "audio_stereo.json").write_text(json.dumps({"alignments": alignments}))

    if transcript_lines is None:
        transcript_lines = [
            {
                "session_id": session_id,
                "speaker": "provider",
                "ts_start": 0.5,
                "ts_end": 1.0,
                "phase": "practice",
                "text": "Hello, let's begin your rehearsal.",
                "is_interim": False,
            },
            {
                "session_id": session_id,
                "speaker": "caller",
                "ts_start": 2.0,
                "ts_end": 2.6,
                "phase": "practice",
                "text": "I froze when my manager pushed back.",
                "is_interim": False,
            },
        ]
    with open(session_dir / "transcript.jsonl", "w") as fh:
        for line in transcript_lines:
            fh.write(json.dumps(line) + "\n")

    (session_dir / "prosody.jsonl").write_text("")
    (session_dir / "session.json").write_text(json.dumps({"id": session_id}))
    return session_dir
