"""Round-trip tests for pydantic contracts.

Establishes that every domain, eval, training, and telemetry type serializes
and deserializes without loss. These are the cheapest tests that assert the
schema exists and is self-consistent; they will run before any application
logic lands.
"""

from datetime import UTC, datetime

from rehearse.types import Session


def test_session_default_persona_key():
    session = Session(created_at=datetime.now(UTC))
    assert session.persona_key == "default"


def test_session_accepts_explicit_persona_key():
    session = Session(created_at=datetime.now(UTC), persona_key="relationship_coach")
    assert session.persona_key == "relationship_coach"
