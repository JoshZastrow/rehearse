"""Live smoke test for ModalTrainer — real GPU run, skipped by default."""

from __future__ import annotations

import json

import pytest

from rehearse.train.curation.trainer import ModalTrainer


@pytest.mark.live_modal
@pytest.mark.slow
def test_modal_trainer_returns_heldout_loss(tmp_path):
    """One miniature budget-matched run end to end.

    Requires: modal auth, sessions + data/heldout_sessions.jsonl already on the
    rehearse-training volume (run the curation sweep first).
    """
    manifest = tmp_path / "curated.jsonl"
    manifest.write_text(
        json.dumps(
            {"path": "sessions/curation-demo-good/audio_stereo.wav", "duration": 56.0}
        )
        + "\n"
    )
    trainer = ModalTrainer(manifest_path=manifest)

    loss = trainer.run(excluded_ids=[], seed=0, max_steps=10, kind="baseline")

    assert isinstance(loss, float) and loss > 0
