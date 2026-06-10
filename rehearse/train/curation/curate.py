"""Curate prepared sessions into a training manifest for the Modal volume.

Scans a sessions root for prepared sessions (audio_stereo.wav present), reviews
each against the current selection criteria, writes a local curated manifest
containing only approved sessions (with turning-point metadata), and pushes the
approved artifacts plus a path-rewritten manifest to the training volume at
data/curated_sessions.jsonl. The existing data/sessions.jsonl is never touched.

Manifest entry shape (superset of the {path, duration} the data loader reads):
    {"path": ..., "duration": ..., "turning_point": {...}, "criteria_version": N}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rehearse.train.curation.criteria import load_criteria
from rehearse.train.curation.reviewer import SessionReviewer
from rehearse.train.curation.types import SessionReview

logger = logging.getLogger(__name__)

_VOLUME_SESSIONS_PREFIX = "data/sessions"
_VOLUME_MANIFEST_PATH = "data/curated_sessions.jsonl"
_PUSHED_SESSION_FILES = ("audio_stereo.wav", "audio_stereo.json", "review.json")


@dataclass
class CurationResult:
    approved: list[SessionReview]
    rejected: list[SessionReview]
    manifest_path: Path
    entries: list[dict[str, Any]]


def _load_existing_review(session_dir: Path, criteria_version: int) -> SessionReview | None:
    path = session_dir / "review.json"
    if not path.exists():
        return None
    try:
        review = SessionReview.model_validate_json(path.read_text())
    except Exception:
        return None
    return review if review.criteria_version == criteria_version else None


async def curate_sessions(
    sessions_root: Path,
    *,
    criteria_dir: Path,
    out: Path,
    reviewer: SessionReviewer | None = None,
    volume_client: Any = None,
    rerun: bool = False,
) -> CurationResult:
    criteria_version, criteria_text = load_criteria(criteria_dir)
    reviewer = reviewer or SessionReviewer()

    session_dirs = sorted({p.parent for p in sessions_root.rglob("audio_stereo.wav")})
    logger.info("Curating %d prepared sessions under %s", len(session_dirs), sessions_root)

    approved: list[SessionReview] = []
    rejected: list[SessionReview] = []
    entries: list[dict[str, Any]] = []

    for session_dir in session_dirs:
        review = None if rerun else _load_existing_review(session_dir, criteria_version)
        if review is None:
            review = await reviewer.review(
                session_dir, criteria_text=criteria_text, criteria_version=criteria_version
            )
        if review.decision == "approved":
            approved.append(review)
            assert review.turning_point is not None
            entries.append(
                {
                    "path": str(session_dir / "audio_stereo.wav"),
                    "duration": review.audio_duration,
                    "turning_point": review.turning_point.model_dump(),
                    "criteria_version": criteria_version,
                }
            )
        else:
            rejected.append(review)
        logger.info("%s: %s — %s", review.session_id, review.decision, review.rationale[:120])

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")
    logger.info("Wrote %s (%d approved, %d rejected)", out, len(approved), len(rejected))

    if volume_client is not None and entries:
        files: list[tuple[str, bytes]] = []
        remote_entries = []
        for entry in entries:
            session_dir = Path(entry["path"]).parent
            session_id = session_dir.name
            for name in _PUSHED_SESSION_FILES:
                local = session_dir / name
                if local.exists():
                    files.append(
                        (f"{_VOLUME_SESSIONS_PREFIX}/{session_id}/{name}", local.read_bytes())
                    )
            remote_entries.append(
                {**entry, "path": f"/data/{_VOLUME_SESSIONS_PREFIX}/{session_id}/audio_stereo.wav"}
            )
        manifest = ("\n".join(json.dumps(e) for e in remote_entries) + "\n").encode()
        files.append((_VOLUME_MANIFEST_PATH, manifest))
        volume_client.push(files)
        logger.info("Pushed %d files to training volume", len(files))

    return CurationResult(approved=approved, rejected=rejected, manifest_path=out, entries=entries)
