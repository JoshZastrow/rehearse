"""Tests for PersonaRecord, PersonaRegistry, and PERSONA_REGISTRY seed data."""

from __future__ import annotations

import pytest

from rehearse.personas.registry import (
    PERSONA_REGISTRY,
    PersonaRecord,
    PersonaRegistry,
)


# ---------------------------------------------------------------------------
# PersonaRecord
# ---------------------------------------------------------------------------


def test_persona_record_has_required_fields() -> None:
    p = PersonaRecord(
        id="test_male",
        name="Test Male",
        description="A test persona.",
        gender="male",
        voice_name="Wise Man",
    )
    assert p.id == "test_male"
    assert p.gender == "male"
    assert p.clm_endpoint is None
    assert p.tags == []


def test_persona_record_to_tool_dict_contains_routing_fields() -> None:
    p = PersonaRecord(
        id="test_male",
        name="Test Male",
        description="A direct male counterparty.",
        gender="male",
        voice_name="Wise Man",
        tags=["work", "direct"],
    )
    d = p.to_tool_dict()
    assert d["id"] == "test_male"
    assert d["name"] == "Test Male"
    assert d["description"] == "A direct male counterparty."
    assert d["gender"] == "male"
    assert "work" in d["tags"]
    # voice_name and system_prompt_template not exposed to routing agent
    assert "voice_name" not in d
    assert "system_prompt_template" not in d


# ---------------------------------------------------------------------------
# PersonaRegistry
# ---------------------------------------------------------------------------


def _registry() -> PersonaRegistry:
    return PersonaRegistry([
        PersonaRecord(id="f_default", name="Female Default", description="Warm female.", gender="female", voice_name="Inspiring Woman", tags=["default", "female"]),
        PersonaRecord(id="m_default", name="Male Default", description="Calm male.", gender="male", voice_name="Wise Man", tags=["default", "male"]),
        PersonaRecord(id="m_boss", name="Direct Boss", description="Demanding male boss.", gender="male", voice_name="Wise Man", tags=["work", "confrontational", "male"]),
    ])


def test_registry_get_by_id() -> None:
    r = _registry()
    assert r.get("m_boss") is not None
    assert r.get("m_boss").id == "m_boss"


def test_registry_get_missing_returns_none() -> None:
    assert _registry().get("nonexistent") is None


def test_registry_list_all() -> None:
    assert len(_registry().list()) == 3


def test_registry_list_filter_by_gender() -> None:
    males = _registry().list(gender="male")
    assert len(males) == 2
    assert all(p.gender == "male" for p in males)


def test_registry_list_filter_by_tags() -> None:
    confrontational = _registry().list(tags=["confrontational"])
    assert len(confrontational) == 1
    assert confrontational[0].id == "m_boss"


def test_registry_list_filter_gender_and_tags() -> None:
    results = _registry().list(gender="male", tags=["work"])
    assert len(results) == 1
    assert results[0].id == "m_boss"


def test_registry_default_for_gender_female() -> None:
    p = _registry().default(gender="female")
    assert p.gender == "female"
    assert "default" in p.tags


def test_registry_default_for_gender_male() -> None:
    p = _registry().default(gender="male")
    assert p.gender == "male"
    assert "default" in p.tags


def test_registry_to_tool_response_is_list_of_dicts() -> None:
    items = _registry().to_tool_response()
    assert isinstance(items, list)
    assert all(isinstance(i, dict) for i in items)
    assert all("id" in i and "description" in i for i in items)


# ---------------------------------------------------------------------------
# Seed registry
# ---------------------------------------------------------------------------


def test_seed_registry_has_male_and_female_defaults() -> None:
    registry = PersonaRegistry(PERSONA_REGISTRY)
    assert registry.default(gender="male") is not None
    assert registry.default(gender="female") is not None


def test_seed_registry_all_records_have_voice_names() -> None:
    for p in PERSONA_REGISTRY:
        assert p.voice_name, f"Persona {p.id} has no voice_name"


def test_seed_registry_all_records_have_descriptions() -> None:
    for p in PERSONA_REGISTRY:
        assert p.description, f"Persona {p.id} has no description"
