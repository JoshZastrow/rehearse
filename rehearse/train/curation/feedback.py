"""SIA feedback loop: revise the selection criteria after a training run.

Harness-update path of the SIA pattern — the judge reads the review history and
the training-run outcome, then emits the next criteria version plus an
improvement note explaining what changed and why.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rehearse.train.curation.criteria import load_criteria, save_criteria
from rehearse.train.curation.reviewer import _DEFAULT_MODEL, _parse_json
from rehearse.train.curation.types import SessionReview

_FEEDBACK_SYSTEM = (
    "You maintain the selection criteria for a training-data curation gate. Given the "
    "current criteria, the curation decisions made under them, and the outcome of the "
    "training run that consumed the approved sessions, emit a revised criteria document "
    "that should select better training data next cycle.\n"
    "Respond with ONLY this JSON object (no prose, no code fences):\n"
    "{\n"
    '  "criteria": "<full updated criteria markdown, with the version header bumped>",\n'
    '  "improvement": "<what changed and why, grounded in the reviews and run outcome>"\n'
    "}"
)


def _render_reviews(reviews: list[SessionReview]) -> str:
    lines = []
    for r in reviews:
        tp = (
            f" turning_point=[{r.turning_point.ts_start:.1f}, {r.turning_point.ts_end:.1f}]"
            if r.turning_point
            else ""
        )
        lines.append(f"- {r.session_id}: {r.decision}{tp} — {r.rationale}")
    return "\n".join(lines) if lines else "(no reviews)"


async def update_criteria(
    *,
    criteria_dir: Path,
    reviews: list[SessionReview],
    training_summary: str,
    backend: Any = None,
    model: str = _DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> tuple[int, str]:
    """Write criteria_v{n+1}.md + improvement_v{n+1}.md; return (new_version, new_text)."""
    version, current = load_criteria(criteria_dir)
    if backend is None:
        from rehearse.eval.scorers.judge_backend import backend_from_env

        backend = backend_from_env(model=model, max_tokens=max_tokens)

    user = (
        f"## Current criteria (v{version})\n{current}\n\n"
        f"## Curation decisions under v{version}\n{_render_reviews(reviews)}\n\n"
        f"## Training run outcome\n{training_summary}\n"
    )
    raw = _parse_json(await backend.complete(system=_FEEDBACK_SYSTEM, user=user))

    new_version = version + 1
    save_criteria(criteria_dir, new_version, raw["criteria"])
    (criteria_dir / f"improvement_v{new_version:03d}.md").write_text(raw["improvement"])
    return new_version, raw["criteria"]
