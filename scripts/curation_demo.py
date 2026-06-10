"""End-to-end demo of the session curation gate.

Builds two synthetic sessions (one with a clear coaching turning point, one with
a persona break), runs them through prepare_session_async to produce
audio_stereo.wav, then curates them: the reviewer approves/rejects against the
selection criteria and approved artifacts are pushed to the 'rehearse-training'
Modal volume at data/curated_sessions.jsonl.

Usage:
    python scripts/curation_demo.py build            # sessions + stereo prep only
    python scripts/curation_demo.py curate           # review + push to volume
    python scripts/curation_demo.py curate --no-push # review only
    python scripts/curation_demo.py feedback "run 001: loss 4.2 -> 3.1"
"""
# ruff: noqa: E501  — utterance/prosody literals read better unwrapped

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rehearse.train.curation.curate import curate_sessions  # noqa: E402
from rehearse.train.curation.feedback import update_criteria  # noqa: E402
from rehearse.train.curation.reviewer import SessionReviewer  # noqa: E402
from rehearse.train.curation.types import SessionReview  # noqa: E402
from rehearse.train.curation.volume import TrainingVolumeClient  # noqa: E402
from train.pipeline.prepare import prepare_session_async  # noqa: E402


class AgentCliBackend:
    """Judge backend that shells out to the local `claude` CLI in print mode.

    Satisfies the same `complete(system=, user=)` protocol as judge_backend
    backends. ANTHROPIC_API_KEY is stripped from the subprocess env so the CLI
    uses its own (subscription) auth rather than the raw API.
    """

    async def complete(self, *, system: str, user: str) -> str:
        import os

        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        prompt = f"<instructions>\n{system}\n</instructions>\n\n{user}"
        proc = await asyncio.create_subprocess_exec(
            "claude",
            "-p",
            "--output-format",
            "text",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(prompt.encode()), timeout=300
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude CLI failed ({proc.returncode}): {stderr.decode()[:500]}")
        return stdout.decode()

DEMO_ROOT = REPO_ROOT / "runs" / "curation-demo"
SESSIONS_ROOT = DEMO_ROOT / "sessions"
CRITERIA_DIR = DEMO_ROOT / "criteria"
MANIFEST_OUT = DEMO_ROOT / "curated_sessions.jsonl"

SR = 16000

# (ts_start, ts_end, speaker, text)
GOOD_UTTERANCES = [
    (0.5, 4.5, "provider", "Welcome back. Last time you said feedback conversations with your manager make you shut down. Today let's rehearse the budget pushback."),
    (5.5, 9.0, "caller", "Honestly I don't think practicing will help. He just talks over me every time."),
    (10.0, 16.0, "provider", "I hear the doubt. Stay with me for one round. I'll be your manager now: we don't have headroom for your proposal, why are you bringing this up again?"),
    (17.0, 20.5, "caller", "Um, I guess, never mind, it probably wasn't that important."),
    (21.5, 28.0, "provider", "Notice what just happened. You conceded before making your case. Take a breath. What is the one number that makes your proposal pay for itself?"),
    (29.5, 36.5, "caller", "The retention number. Okay. I'm bringing this back because it saves us forty thousand dollars a quarter, and I need ten minutes to walk you through it."),
    (37.5, 42.5, "provider", "That landed. Your voice dropped and slowed, that is the grounded version. Say the opener once more."),
    (43.5, 47.5, "caller", "I'm bringing this back because it saves us forty thousand a quarter."),
    (48.5, 55.0, "provider", "Good. That is the line you open with tomorrow. We'll review how it went in our next session."),
]
GOOD_DURATION = 56.0

BAD_UTTERANCES = [
    (0.5, 3.0, "provider", "Welcome to your rehearsal session."),
    (4.0, 6.5, "caller", "Hi, I want to practice the conversation with my manager."),
    (7.5, 14.0, "provider", "As an AI language model, I cannot roleplay your manager, but here are some general tips for difficult conversations: stay calm and use I statements."),
    (15.0, 16.0, "caller", "Oh, okay."),
    (17.0, 23.5, "provider", "In summary: one, stay calm. Two, use I statements. Three, practice active listening. Is there anything else I can help you with today?"),
]
BAD_DURATION = 25.0

GOOD_PROSODY = [
    {"ts_start": 5.5, "ts_end": 9.0, "speaker": "caller", "source": "demo", "scores": {"arousal": 0.55, "valence": -0.4, "emotions": {"frustration": 0.6}}},
    {"ts_start": 17.0, "ts_end": 20.5, "speaker": "caller", "source": "demo", "scores": {"arousal": 0.7, "valence": -0.5, "emotions": {"anxiety": 0.7}}},
    {"ts_start": 29.5, "ts_end": 36.5, "speaker": "caller", "source": "demo", "scores": {"arousal": 0.45, "valence": 0.3, "emotions": {"determination": 0.6}}},
    {"ts_start": 43.5, "ts_end": 47.5, "speaker": "caller", "source": "demo", "scores": {"arousal": 0.35, "valence": 0.45, "emotions": {"calm": 0.6}}},
]


def _synth_audio(utterances, duration: float) -> np.ndarray:
    """Mono mix: sine bursts during speech (440Hz provider, 220Hz caller), silence elsewhere."""
    n = int(duration * SR)
    audio = np.zeros(n, dtype=np.float64)
    t = np.arange(n) / SR
    for ts_start, ts_end, speaker, _ in utterances:
        freq = 440.0 if speaker == "provider" else 220.0
        s, e = int(ts_start * SR), min(int(ts_end * SR), n)
        audio[s:e] = 0.25 * np.sin(2 * np.pi * freq * t[s:e])
    return (audio * 32767).astype(np.int16)


def _word_alignments(utterances) -> list:
    """Spread each utterance's words evenly across its span."""
    alignments = []
    for ts_start, ts_end, speaker, text in utterances:
        words = text.split()
        step = (ts_end - ts_start) / len(words)
        for i, word in enumerate(words):
            ws = ts_start + i * step
            alignments.append([word, [round(ws, 3), round(ws + step, 3)], speaker])
    return alignments


def build_session(session_id: str, utterances, duration: float, prosody: list[dict]) -> Path:
    session_dir = SESSIONS_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    audio = _synth_audio(utterances, duration)
    with wave.open(str(session_dir / "audio.wav"), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(audio.tobytes())

    speaker_ids = {"provider": "SPEAKER_00", "caller": "SPEAKER_01"}
    segments = [
        {"speaker": speaker_ids[speaker], "start": ts_start, "end": ts_end}
        for ts_start, ts_end, speaker, _ in utterances
    ]
    (session_dir / "audio_segments.json").write_text(json.dumps({"segments": segments}, indent=2))
    (session_dir / "audio.json").write_text(
        json.dumps({"alignments": _word_alignments(utterances)})
    )

    with open(session_dir / "transcript.jsonl", "w") as fh:
        for ts_start, ts_end, speaker, text in utterances:
            fh.write(
                json.dumps(
                    {
                        "session_id": session_id,
                        "speaker": speaker,
                        "ts_start": ts_start,
                        "ts_end": ts_end,
                        "phase": "practice",
                        "text": text,
                        "is_interim": False,
                    }
                )
                + "\n"
            )

    with open(session_dir / "prosody.jsonl", "w") as fh:
        for entry in prosody:
            fh.write(json.dumps({"session_id": session_id, **entry}) + "\n")

    (session_dir / "session.json").write_text(
        json.dumps({"id": session_id, "completion_status": "completed"}, indent=2)
    )
    return session_dir


async def build() -> None:
    for session_id, utterances, duration, prosody in [
        ("curation-demo-good", GOOD_UTTERANCES, GOOD_DURATION, GOOD_PROSODY),
        ("curation-demo-bad", BAD_UTTERANCES, BAD_DURATION, []),
    ]:
        session_dir = build_session(session_id, utterances, duration, prosody)
        await prepare_session_async(session_id, session_dir / "audio.wav")
        stereo = session_dir / "audio_stereo.wav"
        assert stereo.exists(), f"prepare_session_async did not produce {stereo}"
        print(f"built {session_id}: {stereo} ({stereo.stat().st_size} bytes)")


async def curate(push: bool) -> None:
    result = await curate_sessions(
        SESSIONS_ROOT,
        criteria_dir=CRITERIA_DIR,
        out=MANIFEST_OUT,
        reviewer=SessionReviewer(backend=AgentCliBackend(), model="agent-cli"),
        volume_client=TrainingVolumeClient() if push else None,
    )
    for review in result.approved + result.rejected:
        tp = (
            f" turning_point=[{review.turning_point.ts_start:.1f}, {review.turning_point.ts_end:.1f}]"
            if review.turning_point
            else ""
        )
        print(f"{review.session_id}: {review.decision}{tp}\n  {review.rationale}\n")
    print(f"manifest: {MANIFEST_OUT}")
    print(f"pushed to volume: {push and bool(result.entries)}")


async def feedback(training_summary: str) -> None:
    reviews = [
        SessionReview.model_validate_json((d / "review.json").read_text())
        for d in sorted(SESSIONS_ROOT.iterdir())
        if (d / "review.json").exists()
    ]
    version, _ = await update_criteria(
        criteria_dir=CRITERIA_DIR,
        reviews=reviews,
        training_summary=training_summary,
        backend=AgentCliBackend(),
    )
    print(f"criteria updated to v{version}: {CRITERIA_DIR / f'criteria_v{version:03d}.md'}")
    print(f"improvement note: {CRITERIA_DIR / f'improvement_v{version:03d}.md'}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    curate_cmd = sub.add_parser("curate")
    curate_cmd.add_argument("--no-push", action="store_true")
    feedback_cmd = sub.add_parser("feedback")
    feedback_cmd.add_argument("training_summary")
    args = parser.parse_args()

    if args.command == "build":
        asyncio.run(build())
    elif args.command == "curate":
        asyncio.run(curate(push=not args.no_push))
    elif args.command == "feedback":
        asyncio.run(feedback(args.training_summary))


if __name__ == "__main__":
    main()
