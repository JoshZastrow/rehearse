"""Versioned selection criteria storage."""

from __future__ import annotations

from rehearse.train.curation.criteria import DEFAULT_CRITERIA, load_criteria, save_criteria


def test_load_criteria_materializes_v1_when_empty(tmp_path):
    version, text = load_criteria(tmp_path / "criteria")

    assert version == 1
    assert text == DEFAULT_CRITERIA
    assert (tmp_path / "criteria" / "criteria_v001.md").read_text() == DEFAULT_CRITERIA


def test_save_then_load_returns_latest_version(tmp_path):
    criteria_dir = tmp_path / "criteria"
    load_criteria(criteria_dir)  # materialize v1
    save_criteria(criteria_dir, 2, "# v2 criteria\nReject silence.")

    version, text = load_criteria(criteria_dir)

    assert version == 2
    assert text == "# v2 criteria\nReject silence."
    assert (criteria_dir / "criteria_v002.md").exists()
