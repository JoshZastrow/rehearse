"""Versioned selection criteria for the curation gate.

Criteria live as criteria_vNNN.md files in a directory; the highest version is
current. v1 is materialized from DEFAULT_CRITERIA on first load. New versions
are written by the feedback loop (see feedback.py) after each training run.
"""

from __future__ import annotations

import re
from pathlib import Path

DEFAULT_CRITERIA = """\
# Session selection criteria — v1

A session is APPROVED for the training corpus only if ALL of the following hold:

1. Coaching arc: the caller's stance visibly changes during the call
   (resistance -> openness, vague -> specific commitment, panic -> grounded plan).
2. Turning point: there is one identifiable span where that change happens,
   attributable to a provider intervention. It must be annotated with start/end
   timestamps; it is used downstream for RL credit assignment.
3. Persona integrity: the provider stays in character for the whole session —
   no assistant-style breaks (meta commentary, "as an AI", scaffolding leakage).
4. Signal density: both speakers contribute substantively; no degenerate audio
   (near-total silence, repeated loops, cut off mid-sentence).

REJECT if any of: no discernible turning point; persona break; degenerate or
trivially short content; transcript and audio metadata contradict each other.

When approving, set turning_point ts_start/ts_end (seconds, within the audio
duration) to the tightest span containing the shift, with a one-sentence
rationale naming the provider intervention that caused it.
"""

_VERSION_RE = re.compile(r"criteria_v(\d+)\.md$")


def _version_path(criteria_dir: Path, version: int) -> Path:
    return criteria_dir / f"criteria_v{version:03d}.md"


def load_criteria(criteria_dir: Path) -> tuple[int, str]:
    """Return (version, text) of the current criteria, materializing v1 if absent."""
    criteria_dir.mkdir(parents=True, exist_ok=True)
    versions = sorted(
        (int(m.group(1)), p)
        for p in criteria_dir.glob("criteria_v*.md")
        if (m := _VERSION_RE.search(p.name))
    )
    if not versions:
        _version_path(criteria_dir, 1).write_text(DEFAULT_CRITERIA)
        return 1, DEFAULT_CRITERIA
    version, path = versions[-1]
    return version, path.read_text()


def save_criteria(criteria_dir: Path, version: int, text: str) -> Path:
    criteria_dir.mkdir(parents=True, exist_ok=True)
    path = _version_path(criteria_dir, version)
    path.write_text(text)
    return path
