"""
Build audio_stereo.wav for moshi-finetune from the separated PCM streams
written by the interactive inference server (infra/interactive.py).

The interactive server saves raw PCM16 16kHz streams to the rehearse-sessions
Modal Volume after each call:

    rehearse-sessions/{session_id}/provider_stream.pcm  (Moshi output, 16kHz)
    rehearse-sessions/{session_id}/caller_stream.pcm    (user input, 16kHz)

This pipeline reads those streams, resamples to 24kHz (Mimi's sample rate),
and writes a stereo WAV to the rehearse-training Modal Volume:

    /data/data/sessions/{session_id}/audio_stereo.wav
    ch0 (left)  = provider (coach / Moshi output)
    ch1 (right) = caller  (user input)

Sessions that have no PCM streams (non-interactive-backend calls) are skipped
silently — the diarization-based prepare.py handles those.

After prepare_stereo, run annotate.py to produce audio_stereo.json, then
dataset.py to build the training manifest.

── Modal run ────────────────────────────────────────────────────────────────
    modal run train/pipeline/prepare_stereo.py
    modal run train/pipeline/prepare_stereo.py --rerun   # reprocess existing

── Standalone (dry-run, no Modal) ───────────────────────────────────────────
    python train/pipeline/prepare_stereo.py --local --provider /path/to/provider.pcm \\
        --caller /path/to/caller.pcm --out /path/to/audio_stereo.wav
"""
from __future__ import annotations

import logging
import sys
import wave
from math import gcd
from pathlib import Path

import modal
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SRC_RATE = 16_000   # PCM streams are recorded at 16kHz
_DST_RATE = 24_000   # Mimi sample rate expected by moshi-finetune

_SESSIONS_MOUNT = "/mnt/sessions"
_TRAINING_MOUNT = "/data"

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

sessions_vol = modal.Volume.from_name("rehearse-sessions", create_if_missing=False)
training_vol = modal.Volume.from_name("rehearse-training", create_if_missing=True)

app = modal.App("rehearse-prepare-stereo")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy>=1.26", "scipy>=1.12")
)

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _resample_pcm16(raw: bytes, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample raw PCM16 bytes from src_rate to dst_rate.

    Uses polyphase resampling (exact rational ratio). Returns int16 array.
    """
    from scipy.signal import resample_poly

    g = gcd(dst_rate, src_rate)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    resampled = resample_poly(samples, dst_rate // g, src_rate // g)
    return np.clip(resampled, -32768, 32767).astype(np.int16)


def _write_stereo_wav(
    path: Path,
    left: np.ndarray,
    right: np.ndarray,
    rate: int,
) -> None:
    """Write a stereo PCM16 WAV from two int16 arrays, padding to equal length."""
    n = max(len(left), len(right))
    if len(left) < n:
        left = np.pad(left, (0, n - len(left)))
    if len(right) < n:
        right = np.pad(right, (0, n - len(right)))

    stereo = np.stack([left, right], axis=1)  # (N, 2) interleaved
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(stereo.tobytes())


def build_stereo_wav(
    provider_pcm: bytes,
    caller_pcm: bytes,
    out_path: Path,
) -> None:
    """Core transform: two raw PCM16 16kHz bytestrings → stereo 24kHz WAV."""
    provider = _resample_pcm16(provider_pcm, _SRC_RATE, _DST_RATE)
    caller = _resample_pcm16(caller_pcm, _SRC_RATE, _DST_RATE)
    _write_stereo_wav(out_path, provider, caller, _DST_RATE)


# ---------------------------------------------------------------------------
# Modal function
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    timeout=600,
    volumes={
        _SESSIONS_MOUNT: sessions_vol,
        _TRAINING_MOUNT: training_vol,
    },
)
def prepare_one(session_id: str, rerun: bool = False) -> str | None:
    """Build audio_stereo.wav for one session from its Modal Volume PCM streams.

    Returns the output path on success, None if skipped (already exists or no
    PCM streams found).
    """
    _init_logging()

    out_path = (
        Path(_TRAINING_MOUNT) / "data" / "sessions" / session_id / "audio_stereo.wav"
    )
    if out_path.exists() and not rerun:
        logger.debug("already prepared, skipping: %s", session_id)
        return None

    provider_path = Path(_SESSIONS_MOUNT) / session_id / "provider_stream.pcm"
    caller_path = Path(_SESSIONS_MOUNT) / session_id / "caller_stream.pcm"

    if not provider_path.exists() or not caller_path.exists():
        logger.debug("no PCM streams found, skipping: %s", session_id)
        return None

    provider_pcm = provider_path.read_bytes()
    caller_pcm = caller_path.read_bytes()

    if not provider_pcm or not caller_pcm:
        logger.warning("empty PCM stream for session %s, skipping", session_id)
        return None

    build_stereo_wav(provider_pcm, caller_pcm, out_path)
    training_vol.commit()

    duration_s = len(provider_pcm) / (_SRC_RATE * 2)  # PCM16 = 2 bytes/sample
    logger.info("wrote %s (%.1fs)", out_path, duration_s)
    return str(out_path)


# ---------------------------------------------------------------------------
# Batch entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def cli(rerun: bool = False) -> None:
    """Prepare stereo WAVs for all sessions in the rehearse-sessions volume.

    Usage:
        modal run train/pipeline/prepare_stereo.py
        modal run train/pipeline/prepare_stereo.py --rerun
    """
    _init_logging()

    entries = sessions_vol.listdir("/")
    # Keep only directory-like entries (session IDs are UUID hex strings, no dots)
    session_ids = [
        e.path.strip("/")
        for e in entries
        if "." not in Path(e.path).name
    ]

    if not session_ids:
        print("No sessions found in rehearse-sessions volume.")
        return

    logger.info("Processing %d sessions...", len(session_ids))

    prepared, skipped, errors = 0, 0, 0
    for session_id, result in zip(
        session_ids,
        prepare_one.map(
            session_ids,
            kwargs={"rerun": rerun},
            return_exceptions=True,
        ),
    ):
        if isinstance(result, Exception):
            logger.error("error processing %s: %s", session_id, result)
            errors += 1
        elif result is None:
            skipped += 1
        else:
            prepared += 1

    print(f"Done — prepared: {prepared}, skipped: {skipped}, errors: {errors}")


# ---------------------------------------------------------------------------
# Async hook (for future wiring into session finalize)
# ---------------------------------------------------------------------------


async def prepare_stereo_async(session_id: str) -> None:
    """Fire-and-forget: build audio_stereo.wav on Modal after a session ends.

    Intended to be called as an asyncio background task from
    SessionOrchestrator.finalize(). Skips silently if PCM streams are absent
    (non-interactive-backend sessions handled by diarization pipeline instead).
    """
    import asyncio

    logger.info("prepare_stereo_async: starting for session %s", session_id)
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: prepare_one.remote(session_id),
        )
        if result:
            logger.info("prepare_stereo_async: wrote %s", result)
        else:
            logger.debug("prepare_stereo_async: skipped session %s", session_id)
    except Exception:
        logger.exception("prepare_stereo_async: failed for session %s", session_id)


# ---------------------------------------------------------------------------
# Local dry-run (no Modal)
# ---------------------------------------------------------------------------


def _local_main() -> None:
    """Test the transform locally without Modal.

    Usage:
        python train/pipeline/prepare_stereo.py --local \\
            --provider /path/to/provider.pcm \\
            --caller  /path/to/caller.pcm \\
            --out     /path/to/audio_stereo.wav
    """
    import argparse

    parser = argparse.ArgumentParser(description="Build stereo WAV from PCM streams (local)")
    parser.add_argument("--provider", required=True, type=Path)
    parser.add_argument("--caller", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(sys.argv[2:])  # skip '--local'

    _init_logging(verbose=True)
    build_stereo_wav(
        args.provider.read_bytes(),
        args.caller.read_bytes(),
        args.out,
    )
    print(f"Wrote {args.out}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--local":
        _local_main()
    else:
        print("Use: modal run train/pipeline/prepare_stereo.py")
        print("Or:  python train/pipeline/prepare_stereo.py --local --provider ... --caller ... --out ...")
