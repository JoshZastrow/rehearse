"""
Speaker diarization for audio files using pyannote on Modal GPU.

── Modal run (recommended) ──────────────────────────────────────────────────
    modal run train/pipeline/diarize.py --egs /path/to/sessions.jsonl
    modal run train/pipeline/diarize.py --egs /path/to/sessions.jsonl \\
        --num-speakers 2

── Standalone chz CLI ────────────────────────────────────────────────────────
    python train/pipeline/diarize.py egs=/path/to/sessions.jsonl
    python train/pipeline/diarize.py egs=/path/to/sessions.jsonl num_speakers=3

── Output ───────────────────────────────────────────────────────────────────
    Each audio.wav gets a sibling audio_segments.json:
    {"segments": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.3}, ...]}

    Already-diarized files are skipped (audio_segments.json exists).
    Failed files write a audio_segments.json.err sentinel; use rerun_errors=true to retry.

── Pipeline order ────────────────────────────────────────────────────────────
    1. diarize.py  — produces audio_segments.json with speaker segments
    2. annotate.py — uses audio_segments.json to assign speaker labels to each word

── Requirements ─────────────────────────────────────────────────────────────
    Requires a Modal secret named HF_TOKEN containing a HuggingFace token
    that has accepted the pyannote/speaker-diarization-3.1 model agreement at:
    https://huggingface.co/pyannote/speaker-diarization-3.1

── Configuration (DiarizeConfig) ────────────────────────────────────────────
    egs                       Path to JSONL manifest (required)
                              Each line: {"path": "/abs/path/to/audio.wav"}

    num_speakers              Expected number of speakers (default: 2)

    rerun_errors              Re-process files that previously failed
                              (default: False)

    verbose                   Enable debug logging (default: False)
"""

from __future__ import annotations

import gzip
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

_GPU_TYPE = os.environ.get("REHEARSE_GPU", "T4")


# ─── Modal image ──────────────────────────────────────────────────────────────

app = modal.App("rehearse-diarize")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "numpy<2",
        "torch==2.2.2",
        "torchaudio==2.2.2",
        "pyannote.audio==3.3.2",
        "sphn==0.1.12",
        # chz and pydantic are needed so Modal can import this module remotely
        # (module-level imports run on both local and remote sides).
        "chz",
        "pydantic",
    )
)


# ─── Modal GPU function ───────────────────────────────────────────────────────


@app.function(
    image=image,
    gpu=_GPU_TYPE,
    timeout=1800,
    secrets=[modal.Secret.from_name("HF_TOKEN")],
)
def diarize_remote(audio_bytes: bytes, num_speakers: int = 2) -> list[dict]:
    """Run speaker diarization on a Modal GPU worker using pyannote.

    Args:
        audio_bytes: Raw WAV file bytes.
        num_speakers: Expected number of distinct speakers in the audio.

    Returns:
        List of segments: [{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.3}, ...]
    """
    import torch
    import pyannote.audio.core.pipeline as _pap

    # pyannote.audio==3.3.2 calls hf_hub_download(use_auth_token=...) which was
    # removed in huggingface_hub>=0.23.0. Patch pyannote's module-level reference
    # to remap use_auth_token → token before the validator runs.
    _orig_dl = _pap.hf_hub_download
    def _compat_dl(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs.setdefault("token", kwargs.pop("use_auth_token"))
        return _orig_dl(*args, **kwargs)
    _pap.hf_hub_download = _compat_dl

    from pyannote.audio import Pipeline

    device = torch.device("cuda:0")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.environ["HF_TOKEN"],
    )
    pipeline.to(device)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name

    try:
        diarization = pipeline(tmp, num_speakers=num_speakers)
    finally:
        os.unlink(tmp)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({"speaker": speaker, "start": turn.start, "end": turn.end})

    return segments


# ─── Config ───────────────────────────────────────────────────────────────────


@chz.chz
class DiarizeConfig:
    """Diarization job configuration.

    Modal run CLI:
        modal run train/pipeline/diarize.py --egs /path/to/sessions.jsonl

    Standalone chz CLI:
        python train/pipeline/diarize.py egs=/path/to/sessions.jsonl

    Nested fields (chz dot-notation):
        python train/pipeline/diarize.py egs=... num_speakers=3
    """

    egs: Path
    """JSONL manifest; each line must be {"path": "/abs/path/to/audio.wav"}."""

    num_speakers: int = 2
    """Expected number of distinct speakers in the audio."""

    rerun_errors: bool = False
    """Re-process files that previously errored (audio_segments.json.err exists).
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


def _validate_output(result: list[dict]):
    """Validate a diarization result list against the DiarizationOutput schema.

    Raises pydantic.ValidationError if the structure is unexpected.
    """
    try:
        from train.pipeline.schemas import DiarizationOutput
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from schemas import DiarizationOutput  # type: ignore[no-redef]

    return DiarizationOutput.model_validate({"segments": result})


# ─── Core runner ─────────────────────────────────────────────────────────────


def _run(config: DiarizeConfig) -> None:
    """Process all pending audio files in the manifest."""
    _init_logging(config.verbose)

    paths = _load_audio_paths(config.egs)
    pending = []
    for p in paths:
        out = p.parent / "audio_segments.json"
        err = p.parent / "audio_segments.json.err"
        if out.exists():
            logger.debug("Skipping (already diarized): %s", p)
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

    for idx, path in enumerate(pending):
        logger.info("[%d/%d] %s", idx + 1, len(pending), path)
        out_path = path.parent / "audio_segments.json"
        err_path = path.parent / "audio_segments.json.err"
        try:
            audio_bytes = path.read_bytes()
            result = diarize_remote.remote(audio_bytes, config.num_speakers)

            logger.info("  → remote returned %d segments", len(result))
            output = _validate_output(result)
            logger.info("  → schema OK: %d segments", len(output.segments))

            payload = {"segments": [s.model_dump() for s in output.segments]}
            with _write_and_rename(out_path) as fh:
                json.dump(payload, fh, ensure_ascii=False)

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
    num_speakers: int = 2,
    rerun_errors: bool = False,
    verbose: bool = False,
):
    """Diarize audio files with pyannote on Modal GPU.

    Usage:
        modal run train/pipeline/diarize.py --egs /path/to/sessions.jsonl
        modal run train/pipeline/diarize.py --egs /path/to/sessions.jsonl \\
            --num-speakers 3

    Override GPU type via environment variable (default: T4):
        REHEARSE_GPU=A10G modal run train/pipeline/diarize.py --egs ...

    Requires a Modal secret named HF_TOKEN with a HuggingFace token that has
    accepted the pyannote/speaker-diarization-3.1 model agreement.
    """
    _run(
        DiarizeConfig(
            egs=Path(egs),
            num_speakers=num_speakers,
            rerun_errors=rerun_errors,
            verbose=verbose,
        )
    )


if __name__ == "__main__":
    # Standalone chz CLI — runs diarize_remote via Modal if authenticated.
    # Usage: python train/pipeline/diarize.py egs=/path/to/sessions.jsonl
    chz.nested_entrypoint(_run)
