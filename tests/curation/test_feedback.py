"""SIA feedback loop: update selection criteria after a training run."""

from __future__ import annotations

import json

from rehearse.train.curation.criteria import load_criteria
from rehearse.train.curation.feedback import update_criteria
from rehearse.train.curation.types import SessionReview
from tests.curation.conftest import FakeJudgeBackend


def _review(session_id: str, decision: str) -> SessionReview:
    return SessionReview(
        session_id=session_id,
        decision=decision,
        turning_point=None,
        rationale="test",
        criteria_version=1,
        audio_duration=20.0,
    )


async def test_update_criteria_writes_next_version_and_improvement(tmp_path):
    criteria_dir = tmp_path / "criteria"
    load_criteria(criteria_dir)  # materialize v1

    new_criteria = "# Selection criteria v2\nRequire >=30s of caller speech."
    backend = FakeJudgeBackend(
        lambda s, u: json.dumps(
            {"criteria": new_criteria, "improvement": "Short sessions hurt loss; raise floor."}
        )
    )

    version, text = await update_criteria(
        criteria_dir=criteria_dir,
        reviews=[_review("good-session", "approved"), _review("bad-session", "rejected")],
        training_summary="run 001: loss 4.2 -> 3.1 on 1 curated session",
        backend=backend,
    )

    assert version == 2
    assert text == new_criteria
    assert (criteria_dir / "criteria_v002.md").read_text() == new_criteria
    improvement = (criteria_dir / "improvement_v002.md").read_text()
    assert "raise floor" in improvement

    # The prompt grounds the update in reviews and the training outcome (SIA pattern).
    [call] = backend.calls
    assert "good-session" in call["user"]
    assert "loss 4.2 -> 3.1" in call["user"]

    # Subsequent loads pick up the new version.
    assert load_criteria(criteria_dir) == (2, new_criteria)
