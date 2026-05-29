"""
Build a training manifest (JSONL) from session audio files.

Scans a sessions root directory for audio.wav files that have a sibling
audio.json annotation, computes duration for each, and writes a manifest
in the same format as DailyTalk's dailytalk.jsonl.

── Usage ────────────────────────────────────────────────────────────────────
    python train/pipeline/dataset.py \\
        sessions_root=/path/to/sessions \\
        out=/path/to/sessions.jsonl

    # Only include sessions that already have audio.json annotations:
    python train/pipeline/dataset.py \\
        sessions_root=/path/to/sessions \\
        out=/path/to/sessions.jsonl \\
        require_annotation=true

    # Push data to Modal Volume after building (default: true):
    python train/pipeline/dataset.py sessions_root=sessions/ out=runs/sessions.jsonl

── Output format ────────────────────────────────────────────────────────────
    Each line: {"path": "/abs/path/to/audio.wav", "duration": 12.34}

    This is identical to DailyTalk's dailytalk.jsonl and can be passed
    directly to data.train_data in the training config.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import chz
import sphn

logger = logging.getLogger(__name__)


@chz.chz
class ManifestConfig:
    sessions_root: Path
    """Root directory containing session subdirectories, each with audio.wav."""

    out: Path
    """Output JSONL path."""

    require_annotation: bool = True
    """Only include sessions that have a sibling audio.json annotation file."""

    min_duration: float = 1.0
    """Skip audio files shorter than this many seconds."""

    push_to_volume: bool = True
    """After writing the manifest, sync session audio files to Modal Volume
    'rehearse-training'. The local manifest is unchanged; a rewritten copy
    with /data/data/... paths is written to the Volume at
    /data/data/sessions.jsonl."""

    verbose: bool = False


def _init_logging(verbose: bool) -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _run(config: ManifestConfig) -> None:
    _init_logging(config.verbose)

    wavs = sorted(config.sessions_root.rglob("audio.wav"))
    logger.info("Found %d audio.wav files under %s", len(wavs), config.sessions_root)

    entries = []
    skipped = 0
    for wav in wavs:
        annotation = wav.with_suffix(".json")
        if config.require_annotation and not annotation.exists():
            logger.debug("Skipping (no annotation): %s", wav)
            skipped += 1
            continue

        try:
            x, sr = sphn.read(str(wav))
            duration = x.shape[-1] / sr
        except Exception as exc:
            logger.warning("Failed to read %s: %s", wav, exc)
            skipped += 1
            continue

        if duration < config.min_duration:
            logger.debug("Skipping short file (%.2fs): %s", duration, wav)
            skipped += 1
            continue

        entries.append({"path": str(wav), "duration": duration})
        logger.debug("%.2fs  %s", duration, wav)

    logger.info("%d entries, %d skipped", len(entries), skipped)

    config.out.parent.mkdir(parents=True, exist_ok=True)
    with open(config.out, "w") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")

    logger.info("Wrote %s", config.out)

    if config.push_to_volume and entries:
        files: list[tuple[str, bytes]] = []
        rewritten_entries = []
        for entry in entries:
            wav = Path(entry["path"])
            session_id = wav.parent.name
            remote_wav = f"/data/data/sessions/{session_id}/audio.wav"
            files.append((remote_wav, wav.read_bytes()))
            ann = wav.with_suffix(".json")
            if ann.exists():
                remote_ann = f"/data/data/sessions/{session_id}/audio.json"
                files.append((remote_ann, ann.read_bytes()))
            rewritten_entries.append({"path": remote_wav, "duration": entry["duration"]})
        manifest_content = (
            "\n".join(json.dumps(e) for e in rewritten_entries) + "\n"
        ).encode()
        try:
            from rehearse.train.modal import push_data
            push_data(files, manifest_content)
            logger.info("Pushed %d files to Modal Volume 'rehearse-training'", len(files))
        except Exception as exc:
            logger.error(
                "Failed to push to Modal Volume (is modal authenticated? run `modal token new`): %s",
                exc,
            )


if __name__ == "__main__":
    chz.nested_entrypoint(_run)
