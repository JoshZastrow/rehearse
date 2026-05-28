"""
Annotate audio files with word-level transcripts using Whisper on Modal GPU.

── Modal run (recommended) ──────────────────────────────────────────────────
    modal run train/pipeline/annotate.py --egs /path/to/sessions.jsonl
    modal run train/pipeline/annotate.py --egs /path/to/sessions.jsonl \\
        --lang fr --whisper-model large-v2

── Standalone chz CLI ────────────────────────────────────────────────────────
    python train/pipeline/annotate.py egs=/path/to/sessions.jsonl
    python train/pipeline/annotate.py egs=/path/to/sessions.jsonl lang=fr

── Output ───────────────────────────────────────────────────────────────────
    Each audio.wav gets a sibling audio.json:
    {"alignments": [["word", [start_sec, end_sec], "SPEAKER_MAIN"], ...]}

    Already-annotated files are skipped (audio.json exists).
    Failed files write a audio.json.err sentinel; use rerun_errors=true to retry.

── Schema validation ─────────────────────────────────────────────────────────
    session.json is validated against Session before processing begins.
    Each annotation result is validated against AnnotationOutput before writing.

── Configuration (AnnotateConfig) ───────────────────────────────────────────
    egs                       Path to JSONL manifest (required)
                              Each line: {"path": "/abs/path/to/audio.wav"}

    lang                      Whisper language code (default: "en")

    whisper_model             Whisper model size (default: "medium")
                              Use "medium" for stereo audio with VAD.
                              Use "large-v2" for higher accuracy.

    keep_silence_in_segments  Seconds of silence to retain at VAD segment
                              boundaries (default: 0.5). Prevents edge words
                              from being clipped by the VAD pass.

    rerun_errors              Re-process files that previously failed
                              (default: False)

    verbose                   Enable debug logging (default: False)
"""

from __future__ import annotations

import gc
import gzip
import importlib
import json
import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import chz
import modal

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
_GPU_TYPE = os.environ.get("REHEARSE_GPU", "T4")


# ─── Modal image ──────────────────────────────────────────────────────────────

app = modal.App("rehearse-annotate")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "torchaudio",
        "sphn==0.1.12",
        "auditok==0.2",
        "whisper_timestamped",
        "numba>=0.61",
        "llvmlite>=0.44",
        # chz and pydantic are needed so Modal can import this module remotely
        # (module-level imports run on both local and remote sides).
        "chz",
        "pydantic",
    )
)


# ─── Modal GPU function ───────────────────────────────────────────────────────


@app.function(image=image, gpu=_GPU_TYPE, timeout=1800)
def process_remote(
    audio_bytes: bytes,
    lang: str,
    whisper_model: str,
    keep_silence: float,
) -> dict:
    """Transcribe audio on a Modal GPU worker using Whisper with forced alignment.

    Args:
        audio_bytes: Raw WAV file bytes.
        lang: Whisper language code (e.g. "en", "fr").
        whisper_model: Whisper model size (e.g. "medium", "large-v2").
        keep_silence: Seconds of silence to pad around VAD segment boundaries.

    Returns:
        {"alignments": [["word", [start_sec, end_sec], "SPEAKER_MAIN"], ...]}
    """
    import torch
    import torchaudio.functional as F
    import whisper_timestamped as whisper
    import sphn

    transcribe_mod = importlib.import_module("whisper_timestamped.transcribe")
    old_get_vad = transcribe_mod.get_vad_segments

    device = torch.device("cuda:0")
    model = whisper.load_model(whisper_model, device=device)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name

    try:
        if keep_silence > 0:
            d = int(SAMPLE_RATE * keep_silence)

            def _patched_get_vad(*args, **kwargs):
                segs = old_get_vad(*args, **kwargs)
                out, last = [], 0
                for seg in segs:
                    out.append({"start": max(last, seg["start"] - d), "end": seg["end"] + d})
                    last = out[-1]["end"]
                return out

            transcribe_mod.get_vad_segments = _patched_get_vad

        gc.collect()
        torch.cuda.empty_cache()

        x, sr = sphn.read(tmp)
        x = torch.from_numpy(x).cuda()
        vocals = x[0][None]
        vocals = F.resample(vocals, sr, SAMPLE_RATE)
        vocals = vocals.cpu().numpy()[0]
        dur = vocals.shape[-1] / SAMPLE_RATE

        result = whisper.transcribe(
            model,
            vocals,
            language=lang,
            vad="auditok" if dur > 10 else None,
            best_of=5,
            beam_size=5,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            verbose=None,
        )
    finally:
        os.unlink(tmp)
        if keep_silence > 0:
            transcribe_mod.get_vad_segments = old_get_vad

    alignments = []
    for segment in result["segments"]:
        if "words" not in segment:
            logger.warning("No word-level timestamps in segment: %r", segment)
            continue
        for word in segment["words"]:
            alignments.append([word["text"], [word["start"], word["end"]], "SPEAKER_MAIN"])

    return {"alignments": alignments}


# ─── Config ───────────────────────────────────────────────────────────────────


@chz.chz
class AnnotateConfig:
    """Annotation job configuration.

    Modal run CLI:
        modal run train/pipeline/annotate.py --egs /path/to/sessions.jsonl

    Standalone chz CLI:
        python train/pipeline/annotate.py egs=/path/to/sessions.jsonl

    Nested fields (chz dot-notation):
        python train/pipeline/annotate.py egs=... whisper_model=large-v2 lang=fr
    """

    egs: Path
    """JSONL manifest; each line must be {"path": "/abs/path/to/audio.wav"}."""

    lang: str = "en"
    """Whisper language code. Pass 'fr' for French, 'es' for Spanish, etc."""

    whisper_model: str = "medium"
    """Whisper model size. 'medium' is recommended for stereo with VAD detection.
    Options: tiny, base, small, medium, large-v2, large-v3."""

    keep_silence_in_segments: float = 0.5
    """Seconds of silence to pad at VAD segment boundaries.
    Prevents words at the edges of speech segments from being clipped.
    Set to 0.0 to disable padding."""

    rerun_errors: bool = False
    """Re-process files that previously errored (audio.json.err exists).
    By default, errored files are skipped to avoid infinite retries."""

    verbose: bool = False
    """Enable DEBUG-level logging."""


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _init_logging(verbose: bool) -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _load_audio_paths(egs: Path) -> list[Path]:
    """Read audio paths from a JSONL manifest (plain or gzipped)."""
    open_fn = gzip.open if str(egs).endswith(".gz") else open
    with open_fn(egs, "rb") as fh:
        return [Path(json.loads(line)["path"]) for line in fh]


@contextmanager
def _write_and_rename(path: Path, mode: str = "w") -> Generator:
    """Write to a .tmp file then atomically rename to avoid partial writes."""
    tmp = str(path) + f".tmp.{os.getpid()}"
    with open(tmp, mode) as fh:
        yield fh
    os.rename(tmp, path)


def _validate_session(session_dir: Path):
    """Parse and validate session.json against the Session schema.

    Raises pydantic.ValidationError if the file does not match the schema.
    """
    try:
        from train.pipeline.schemas import Session
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from schemas import Session  # type: ignore[no-redef]

    data = json.loads((session_dir / "session.json").read_text())
    return Session.model_validate(data)


def _validate_output(result: dict):
    """Validate an annotation result dict against the AnnotationOutput schema.

    Raises pydantic.ValidationError if the structure is unexpected.
    """
    try:
        from train.pipeline.schemas import AnnotationOutput
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from schemas import AnnotationOutput  # type: ignore[no-redef]

    return AnnotationOutput.model_validate(result)


# ─── Core runner ─────────────────────────────────────────────────────────────


def _run(config: AnnotateConfig) -> None:
    """Process all pending audio files in the manifest."""
    _init_logging(config.verbose)

    paths = _load_audio_paths(config.egs)
    pending = []
    for p in paths:
        out = p.with_suffix(".json")
        err = p.with_suffix(".json.err")
        if out.exists():
            logger.debug("Skipping (already annotated): %s", p)
            continue
        if err.exists() and not config.rerun_errors:
            logger.debug("Skipping (previous error): %s", p)
            continue
        if p.stat().st_size < 1000:
            logger.warning("Skipping small file (<1KB): %s", p)
            continue
        pending.append(p)

    logger.info("%d / %d files to process", len(pending), len(paths))
    if not pending:
        logger.info("Nothing to do.")
        return

    # Validate session.json schema for the first session directory encountered.
    sample_session_dir = pending[0].parent
    if (sample_session_dir / "session.json").exists():
        try:
            session = _validate_session(sample_session_dir)
            logger.info(
                "Session schema OK: id=%s status=%s", session.id, session.completion_status
            )
        except Exception as exc:
            logger.warning("session.json failed schema validation (continuing): %s", exc)

    for idx, path in enumerate(pending):
        logger.info("[%d/%d] %s", idx + 1, len(pending), path)
        out_path = path.with_suffix(".json")
        err_path = path.with_suffix(".json.err")
        try:
            audio_bytes = path.read_bytes()
            result = process_remote.remote(
                audio_bytes,
                config.lang,
                config.whisper_model,
                config.keep_silence_in_segments,
            )

            logger.info("  → remote returned %d alignments", len(result.get("alignments", [])))
            annotation = _validate_output(result)
            logger.info("  → schema OK: %d words", len(annotation.alignments))

            with _write_and_rename(out_path) as fh:
                json.dump(result, fh, ensure_ascii=False)

            logger.info("  → wrote %s", out_path)

        except Exception as exc:
            if "cuda" in repr(exc).lower():
                raise
            logger.exception("Error processing %s", path)
            err_path.touch()


# ─── Entrypoints ─────────────────────────────────────────────────────────────


@app.local_entrypoint()  # runs locally; chz available here
def cli(
    egs: str,
    lang: str = "en",
    whisper_model: str = "medium",
    keep_silence_in_segments: float = 0.5,
    rerun_errors: bool = False,
    verbose: bool = False,
):
    """Annotate audio files with Whisper on Modal GPU.

    Usage:
        modal run train/pipeline/annotate.py --egs /path/to/sessions.jsonl
        modal run train/pipeline/annotate.py --egs /path/to/sessions.jsonl \\
            --lang fr --whisper-model large-v2

    Override GPU type via environment variable (default: T4):
        REHEARSE_GPU=A10G modal run train/pipeline/annotate.py --egs ...
    """
    _run(
        AnnotateConfig(
            egs=Path(egs),
            lang=lang,
            whisper_model=whisper_model,
            keep_silence_in_segments=keep_silence_in_segments,
            rerun_errors=rerun_errors,
            verbose=verbose,
        )
    )


if __name__ == "__main__":
    # Standalone chz CLI — runs process_remote via Modal if authenticated.
    # Usage: python train/pipeline/annotate.py egs=/path/to/sessions.jsonl
    chz.nested_entrypoint(_run)
