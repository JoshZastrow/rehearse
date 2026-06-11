"""Tombstone: manifest rewrite + review flip, artifacts preserved."""

from __future__ import annotations

import json

from rehearse.train.curation.ablation import Conviction
from rehearse.train.curation.curate import curate_sessions, tombstone_sessions
from rehearse.train.curation.reviewer import SessionReviewer
from tests.curation.conftest import (
    FakeJudgeBackend,
    FakeVolumeClient,
    approve_response,
    make_session_dir,
)


async def test_tombstone_rewrites_manifest_and_flips_review(tmp_path):
    root = tmp_path / "sessions"
    for sid in ("keep-1", "bad-1"):
        make_session_dir(root, sid)
    out = tmp_path / "curated.jsonl"
    volume = FakeVolumeClient()
    await curate_sessions(
        root,
        criteria_dir=tmp_path / "criteria",
        out=out,
        reviewer=SessionReviewer(backend=FakeJudgeBackend(lambda s, u: approve_response())),
        volume_client=volume,
    )
    conviction = Conviction(session_id="bad-1", delta=0.5, noise_band=0.02, run_ids=["r9"])

    tombstone_sessions(sessions_root=root, out=out, convictions=[conviction], volume_client=volume)

    manifest = out.read_text()
    assert "bad-1" not in manifest and "keep-1" in manifest
    remote = volume.pushed["data/curated_sessions.jsonl"].decode()
    assert "bad-1" not in remote and "keep-1" in remote

    review = json.loads((root / "bad-1" / "review.json").read_text())
    assert review["decision"] == "rejected"
    assert review["ablation"]["delta"] == 0.5 and review["ablation"]["run_ids"] == ["r9"]
    assert "0.5" in review["rationale"]
    remote_review = json.loads(volume.pushed["data/sessions/bad-1/review.json"].decode())
    assert remote_review["decision"] == "rejected"

    assert (root / "bad-1" / "audio_stereo.wav").exists()  # artifacts untouched
