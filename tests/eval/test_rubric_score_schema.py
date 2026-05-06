"""Schema regression tests for `RubricScore` extensions (Spec 1).

The new fields (`modality`, `confidence`, `judge_prompt_version`, `flags`)
must default safely so existing artifacts on disk continue to deserialize.
"""

from __future__ import annotations

import pytest

from rehearse.types import RubricScore


def _legacy_kwargs() -> dict:
    """Pre-Spec-1 RubricScore payload — the only fields that ever existed."""
    return dict(
        run_id="r1",
        example_id="e1",
        dimension="content_quality",
        value=0.62,
        scorer="llm_judge",
    )


def test_legacy_rubric_score_still_validates() -> None:
    """Old artifacts on disk parse with defaults for every new field."""
    score = RubricScore(**_legacy_kwargs())
    assert score.modality == "text"
    assert score.confidence is None
    assert score.judge_prompt_version is None
    assert score.flags == []


def test_legacy_rubric_score_round_trip_through_json() -> None:
    """JSON round-trip preserves both legacy and new fields."""
    original = RubricScore(**_legacy_kwargs())
    raw = original.model_dump_json()
    rehydrated = RubricScore.model_validate_json(raw)
    assert rehydrated == original


def test_rubric_score_accepts_new_modalities() -> None:
    for modality in ("text", "audio", "audio+text", "timing", "meta", "aggregate"):
        score = RubricScore(**_legacy_kwargs(), modality=modality)
        assert score.modality == modality


def test_rubric_score_rejects_unknown_modality() -> None:
    with pytest.raises(ValueError):
        RubricScore(**_legacy_kwargs(), modality="multimodal")  # type: ignore[arg-type]


def test_rubric_score_carries_judge_prompt_version() -> None:
    score = RubricScore(
        **_legacy_kwargs(),
        judge_prompt_version="content-quality-v1",
    )
    assert score.judge_prompt_version == "content-quality-v1"


def test_rubric_score_flags_default_is_empty_list() -> None:
    a = RubricScore(**_legacy_kwargs())
    b = RubricScore(**_legacy_kwargs())
    a.flags.append("audio_missing")
    # Must not share a default list across instances (mutable default trap).
    assert b.flags == []


def test_rubric_score_confidence_in_unit_interval() -> None:
    score = RubricScore(**_legacy_kwargs(), confidence=0.91)
    assert score.confidence == pytest.approx(0.91)


def test_rubric_score_emits_new_fields_in_json() -> None:
    score = RubricScore(
        **_legacy_kwargs(),
        modality="audio+text",
        confidence=0.74,
        judge_prompt_version="affect-v1",
        flags=["audio_missing"],
    )
    raw = score.model_dump()
    assert raw["modality"] == "audio+text"
    assert raw["confidence"] == pytest.approx(0.74)
    assert raw["judge_prompt_version"] == "affect-v1"
    assert raw["flags"] == ["audio_missing"]


def test_rubric_score_extra_fields_rejected() -> None:
    """Strict schema: typos surface immediately, not silently."""
    with pytest.raises(ValueError):
        RubricScore(**_legacy_kwargs(), modaliy="text")  # type: ignore[arg-type]
