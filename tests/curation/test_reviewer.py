"""SessionReviewer: judge-backed approve/reject with turning-point annotation."""

from __future__ import annotations

import json

from rehearse.train.curation.reviewer import SessionReviewer
from tests.curation.conftest import (
    FakeJudgeBackend,
    approve_response,
    make_session_dir,
    reject_response,
)

CRITERIA = "# Selection criteria v1\nApprove sessions with a clear coaching turning point."


async def test_approve_writes_review_with_turning_point(tmp_path):
    session_dir = make_session_dir(tmp_path, "good-session", duration_sec=20.0)
    backend = FakeJudgeBackend(lambda s, u: approve_response(ts_start=5.0, ts_end=8.0))
    reviewer = SessionReviewer(backend=backend)

    review = await reviewer.review(session_dir, criteria_text=CRITERIA, criteria_version=1)

    assert review.decision == "approved"
    assert review.session_id == "good-session"
    assert review.criteria_version == 1
    assert review.turning_point is not None
    assert review.turning_point.ts_start == 5.0
    assert review.turning_point.ts_end == 8.0
    assert 0 <= review.turning_point.ts_start < review.turning_point.ts_end
    assert review.turning_point.ts_end <= review.audio_duration

    saved = json.loads((session_dir / "review.json").read_text())
    assert saved["decision"] == "approved"
    assert saved["turning_point"]["ts_start"] == 5.0


async def test_reject_writes_review_without_turning_point(tmp_path):
    session_dir = make_session_dir(tmp_path, "bad-session")
    backend = FakeJudgeBackend(lambda s, u: reject_response())
    reviewer = SessionReviewer(backend=backend)

    review = await reviewer.review(session_dir, criteria_text=CRITERIA, criteria_version=1)

    assert review.decision == "rejected"
    assert review.turning_point is None
    assert review.rationale
    saved = json.loads((session_dir / "review.json").read_text())
    assert saved["decision"] == "rejected"


async def test_approval_without_valid_turning_point_is_coerced_to_rejected(tmp_path):
    """An approval is only valid with a turning point inside the audio; else reject."""
    session_dir = make_session_dir(tmp_path, "overrun-session", duration_sec=10.0)
    # Turning point ends past the 10s audio -> unusable for RL credit assignment.
    backend = FakeJudgeBackend(lambda s, u: approve_response(ts_start=8.0, ts_end=15.0))
    reviewer = SessionReviewer(backend=backend)

    review = await reviewer.review(session_dir, criteria_text=CRITERIA, criteria_version=1)

    assert review.decision == "rejected"
    assert "turning point" in review.rationale.lower()


async def test_prompt_contains_criteria_and_transcript(tmp_path):
    session_dir = make_session_dir(tmp_path, "prompted-session")
    backend = FakeJudgeBackend(lambda s, u: reject_response())
    reviewer = SessionReviewer(backend=backend)

    await reviewer.review(session_dir, criteria_text=CRITERIA, criteria_version=1)

    [call] = backend.calls
    blob = call["system"] + call["user"]
    assert "Selection criteria v1" in blob
    assert "I froze when my manager pushed back." in blob


async def test_prompt_falls_back_to_alignments_when_transcript_empty(tmp_path):
    session_dir = make_session_dir(
        tmp_path,
        "aligned-session",
        transcript_lines=[],
        alignments=[["resilience", [1.0, 1.6], "caller"]],
    )
    backend = FakeJudgeBackend(lambda s, u: reject_response())
    reviewer = SessionReviewer(backend=backend)

    await reviewer.review(session_dir, criteria_text=CRITERIA, criteria_version=1)

    [call] = backend.calls
    assert "resilience" in call["user"]
