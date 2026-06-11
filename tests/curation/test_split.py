"""Held-out split: sticky, deterministic, ~20% min 2 once corpus >= 4 approved."""

from __future__ import annotations

import json

from rehearse.train.curation.curate import curate_sessions
from rehearse.train.curation.reviewer import SessionReviewer
from tests.curation.conftest import (
    FakeJudgeBackend,
    FakeVolumeClient,
    approve_response,
    make_session_dir,
)


async def _curate(tmp_path, n, volume=None):
    root = tmp_path / "sessions"
    for i in range(n):
        make_session_dir(root, f"s{i:02d}")
    return await curate_sessions(
        root,
        criteria_dir=tmp_path / "criteria",
        out=tmp_path / "curated.jsonl",
        reviewer=SessionReviewer(backend=FakeJudgeBackend(lambda s, u: approve_response())),
        volume_client=volume,
    )


async def test_small_corpus_has_no_heldout(tmp_path):
    result = await _curate(tmp_path, 3)
    assert all(r.split == "train" for r in result.approved)


async def test_split_is_two_heldout_at_six_and_sticky(tmp_path):
    result = await _curate(tmp_path, 6)
    heldout = sorted(r.session_id for r in result.approved if r.split == "heldout")
    assert len(heldout) == 2
    # Re-running curation (cached reviews) keeps the same assignment.
    again = await _curate(tmp_path, 6)
    assert sorted(r.session_id for r in again.approved if r.split == "heldout") == heldout
    # Stored in review.json so it survives manifest rebuilds.
    saved = json.loads((tmp_path / "sessions" / heldout[0] / "review.json").read_text())
    assert saved["split"] == "heldout"


async def test_manifests_separate_train_and_heldout(tmp_path):
    volume = FakeVolumeClient()
    result = await _curate(tmp_path, 6, volume=volume)
    train_ids = {
        json.loads(line)["path"].split("/")[-2]
        for line in (tmp_path / "curated.jsonl").read_text().splitlines()
    }
    heldout_ids = {r.session_id for r in result.approved if r.split == "heldout"}
    assert train_ids.isdisjoint(heldout_ids) and len(train_ids) == 4
    remote_heldout = volume.pushed["data/heldout_sessions.jsonl"].decode()
    assert all(sid in remote_heldout for sid in heldout_ids)
    # Held-out audio is pushed too — eval runs read it from the volume.
    sid = sorted(heldout_ids)[0]
    assert f"data/sessions/{sid}/audio_stereo.wav" in volume.pushed
