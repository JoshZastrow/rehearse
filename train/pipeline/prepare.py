"""
Stereo audio preparation for moshi fine-tuning.

Splits a mixed mono session audio.wav into a two-channel stereo file:
  - Left  channel: coach speech
  - Right channel: user speech

Reads audio_segments.json (diarization segment boundaries) and audio.json
(word alignments with coach/user labels) to determine which pyannote speaker
ID maps to which role. Each segment's audio is copied from the mono mix into
the appropriate channel; all other samples remain zero.

Also writes audio_stereo.json as a copy of audio.json so the moshi-finetune
interleaver can find the transcript via the {wav_stem}.json convention.

── Standalone CLI ─────────────────────────────────────────────────────────────
    python train/pipeline/prepare.py egs=/path/to/sessions.jsonl
    python train/pipeline/prepare.py egs=/path/to/sessions.jsonl rerun=true

── Pipeline order ─────────────────────────────────────────────────────────────
    1. diarize.py   → audio_segments.json
    2. annotate.py  → audio.json          (chains to prepare automatically)
    3. prepare.py   → audio_stereo.wav + audio_stereo.json

── Configuration (PrepareConfig) ─────────────────────────────────────────────
    egs         Path to JSONL manifest (required)
    rerun       Re-process sessions that already have audio_stereo.wav
    verbose     Enable DEBUG-level logging
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path

import chz
import numpy as np

logger = logging.getLogger(__name__)


# ─── Audio helpers ─────────────────────────────────────────────────────────────


def _load_pcm16_mono(wav_path: Path) -> tuple[np.ndarray, int]:
    """Load a mono PCM16 WAV; return (int16 samples, sample_rate)."""
    with wave.open(str(wav_path), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"Expected mono audio, got {wf.getnchannels()} channels: {wav_path}")
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16).copy(), sr


def _write_stereo_wav(path: Path, left: np.ndarray, right: np.ndarray, sr: int) -> None:
    """Write a stereo PCM16 WAV from two equal-length int16 arrays."""
    stereo = np.stack([left, right], axis=1)  # (N, 2) interleaved
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(stereo.tobytes())


# ─── Speaker mapping ───────────────────────────────────────────────────────────


def _derive_speaker_map(segments: list[dict], alignments: list) -> dict[str, str]:
    """Map pyannote SPEAKER_XX ids → 'coach'/'user' via word alignment majority vote.

    For each speaker ID, finds all words whose midpoint falls inside one of its
    segments and takes the most common label from audio.json.
    """
    speaker_labels: dict[str, list[str]] = defaultdict(list)
    for word, (ws, we), label in alignments:
        mid = (ws + we) / 2.0
        for seg in segments:
            if seg["start"] <= mid <= seg["end"]:
                speaker_labels[seg["speaker"]].append(label)
                break

    mapping: dict[str, str] = {}
    for sid, labels in speaker_labels.items():
        # Normalize any legacy "user"/"coach" labels from existing audio.json files.
        _compat = {"user": "caller", "coach": "provider", "caller": "caller", "provider": "provider"}
        raw = Counter(labels).most_common(1)[0][0]
        mapping[sid] = _compat.get(raw, raw)

    # Guarantee every segment speaker has an entry
    all_ids = sorted({seg["speaker"] for seg in segments})
    assigned_roles = set(mapping.values())
    for sid in all_ids:
        if sid not in mapping:
            mapping[sid] = "caller" if "provider" in assigned_roles else "provider"

    return mapping


# ─── Core preparation ──────────────────────────────────────────────────────────


def prepare_session(session_dir: Path) -> Path:
    """Create audio_stereo.wav and audio_stereo.json in session_dir.

    Requires:
        audio.wav              — mono mixed recording
        audio_segments.json    — pyannote diarization output
        audio.json             — word alignments with coach/user labels

    Returns:
        Path to the written audio_stereo.wav.
    """
    wav_path = session_dir / "audio.wav"
    segments_path = session_dir / "audio_segments.json"
    alignments_path = session_dir / "audio.json"
    out_wav = session_dir / "audio_stereo.wav"
    out_json = session_dir / "audio_stereo.json"

    mono, sr = _load_pcm16_mono(wav_path)
    segments = json.loads(segments_path.read_text())["segments"]
    alignments = json.loads(alignments_path.read_text())["alignments"]

    speaker_map = _derive_speaker_map(segments, alignments)
    logger.debug("Speaker map: %s", speaker_map)

    left = np.zeros_like(mono)   # provider
    right = np.zeros_like(mono)  # caller

    for seg in segments:
        role = speaker_map.get(seg["speaker"], "caller")
        s = int(seg["start"] * sr)
        e = min(int(seg["end"] * sr), len(mono))
        if role == "provider":
            left[s:e] = mono[s:e]
        else:
            right[s:e] = mono[s:e]

    _write_stereo_wav(out_wav, left, right, sr)

    # audio_stereo.json mirrors audio.json so the moshi-finetune interleaver
    # can resolve the transcript via the {wav_stem}.json convention.
    out_json.write_text(alignments_path.read_text())

    return out_wav


# ─── Async session hook ────────────────────────────────────────────────────────


async def prepare_session_async(session_id: str, audio_path: Path) -> None:
    """Async hook: create audio_stereo.wav after annotate_session_async writes audio.json.

    Runs prepare_session in a thread executor so the event loop is not blocked.
    Skips silently if audio_stereo.wav already exists or required inputs are missing.
    """
    session_dir = audio_path.parent
    out_path = session_dir / "audio_stereo.wav"

    if out_path.exists():
        logger.debug("prepare_session_async: already prepared, skipping %s", session_id)
        return

    required = [audio_path, session_dir / "audio_segments.json", session_dir / "audio.json"]
    if not all(p.exists() for p in required):
        logger.warning("prepare_session_async: missing inputs, skipping %s", session_id)
        return

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, prepare_session, session_dir)
        logger.info("prepare_session_async: wrote %s", out_path)
    except Exception:
        logger.exception("prepare_session_async: failed for session %s", session_id)


# ─── Config + batch runner ─────────────────────────────────────────────────────


@chz.chz
class PrepareConfig:
    """Batch stereo preparation configuration.

    python train/pipeline/prepare.py egs=/path/to/sessions.jsonl
    """

    egs: Path
    """JSONL manifest; each line must be {"path": "/abs/path/to/audio.wav"}."""

    rerun: bool = False
    """Re-process sessions that already have audio_stereo.wav."""

    verbose: bool = False
    """Enable DEBUG-level logging."""


def _init_logging(verbose: bool) -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _run(config: PrepareConfig) -> None:
    _init_logging(config.verbose)

    open_fn = gzip.open if str(config.egs).endswith(".gz") else open
    with open_fn(config.egs, "rb") as fh:
        paths = [Path(json.loads(line)["path"]) for line in fh]

    pending = []
    for p in paths:
        out = p.parent / "audio_stereo.wav"
        if out.exists() and not config.rerun:
            logger.debug("Skipping (already prepared): %s", p)
            continue
        required = [p, p.parent / "audio_segments.json", p.parent / "audio.json"]
        if not all(r.exists() for r in required):
            logger.warning("Skipping (missing inputs): %s", p)
            continue
        pending.append(p)

    logger.info("%d / %d sessions to prepare", len(pending), len(paths))
    if not pending:
        logger.info("Nothing to do.")
        return

    for idx, path in enumerate(pending):
        logger.info("[%d/%d] %s", idx + 1, len(pending), path)
        try:
            out = prepare_session(path.parent)
            logger.info("  → wrote %s", out)
        except Exception:
            logger.exception("  Error preparing %s", path)


if __name__ == "__main__":
    chz.nested_entrypoint(_run)
