"""curate_sessions: review prepared sessions, build curated manifest, push approved."""

from __future__ import annotations

import json

from rehearse.train.curation.curate import curate_sessions
from rehearse.train.curation.reviewer import SessionReviewer
from tests.curation.conftest import (
    FakeJudgeBackend,
    FakeVolumeClient,
    approve_response,
    make_session_dir,
    reject_response,
)


def _backend_approving_only(approved_id: str) -> FakeJudgeBackend:
    return FakeJudgeBackend(
        lambda s, u: approve_response() if approved_id in u else reject_response()
    )


async def test_curate_pushes_only_approved_session(tmp_path):
    sessions_root = tmp_path / "sessions"
    make_session_dir(sessions_root, "good-session", duration_sec=20.0)
    make_session_dir(sessions_root, "bad-session", duration_sec=20.0)

    reviewer = SessionReviewer(backend=_backend_approving_only("good-session"))
    volume = FakeVolumeClient()
    out = tmp_path / "curated_sessions.jsonl"

    result = await curate_sessions(
        sessions_root,
        criteria_dir=tmp_path / "criteria",
        out=out,
        reviewer=reviewer,
        volume_client=volume,
    )

    assert [r.session_id for r in result.approved] == ["good-session"]
    assert [r.session_id for r in result.rejected] == ["bad-session"]

    # Local curated manifest: only the approved session, with turning point metadata.
    [entry] = [json.loads(line) for line in out.read_text().splitlines()]
    assert entry["path"].endswith("good-session/audio_stereo.wav")
    assert entry["duration"] > 0
    assert entry["turning_point"]["ts_start"] == 5.0
    assert entry["criteria_version"] == 1

    # Volume receives the approved session's artifacts and the rewritten manifest.
    assert "data/sessions/good-session/audio_stereo.wav" in volume.pushed
    assert "data/sessions/good-session/audio_stereo.json" in volume.pushed
    assert "data/sessions/good-session/review.json" in volume.pushed
    [remote_entry] = [
        json.loads(line)
        for line in volume.pushed["data/curated_sessions.jsonl"].decode().splitlines()
    ]
    assert remote_entry["path"] == "/data/data/sessions/good-session/audio_stereo.wav"
    assert remote_entry["turning_point"]["ts_start"] == 5.0

    # The rejected session never reaches the volume.
    assert not any("bad-session" in path for path in volume.pushed)


async def test_curate_skips_sessions_already_reviewed_under_same_criteria(tmp_path):
    sessions_root = tmp_path / "sessions"
    session_dir = make_session_dir(sessions_root, "good-session")

    backend = _backend_approving_only("good-session")
    reviewer = SessionReviewer(backend=backend)
    volume = FakeVolumeClient()
    out = tmp_path / "curated_sessions.jsonl"

    await curate_sessions(
        sessions_root,
        criteria_dir=tmp_path / "criteria",
        out=out,
        reviewer=reviewer,
        volume_client=volume,
    )
    assert len(backend.calls) == 1
    assert (session_dir / "review.json").exists()

    # Second run: review.json with matching criteria_version short-circuits the judge.
    result = await curate_sessions(
        sessions_root,
        criteria_dir=tmp_path / "criteria",
        out=out,
        reviewer=reviewer,
        volume_client=volume,
    )
    assert len(backend.calls) == 1
    assert [r.session_id for r in result.approved] == ["good-session"]


async def test_curate_ignores_unprepared_sessions(tmp_path):
    """Sessions without audio_stereo.wav (not yet through prepare) are not reviewed."""
    sessions_root = tmp_path / "sessions"
    make_session_dir(sessions_root, "good-session")
    unprepared = sessions_root / "unprepared-session"
    unprepared.mkdir(parents=True)
    (unprepared / "audio.wav").write_bytes(b"")

    backend = _backend_approving_only("good-session")
    result = await curate_sessions(
        sessions_root,
        criteria_dir=tmp_path / "criteria",
        out=tmp_path / "curated_sessions.jsonl",
        reviewer=SessionReviewer(backend=backend),
        volume_client=FakeVolumeClient(),
    )

    reviewed = {r.session_id for r in result.approved + result.rejected}
    assert reviewed == {"good-session"}
