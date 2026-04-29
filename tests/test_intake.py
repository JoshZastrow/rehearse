"""Test live intake capture and persona compilation on the runtime bus."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rehearse.bus import FrameBus
from rehearse.frames import PhaseSignal, TranscriptDelta
from rehearse.intake import IntakeProcessor
from rehearse.storage import LocalFilesystemStore
from rehearse.types import ConsentState, Phase, Session, Speaker


@pytest.fixture
def intake_store(tmp_path: Path) -> tuple[LocalFilesystemStore, str]:
    """Create a runtime store with one empty session manifest."""
    store = LocalFilesystemStore(tmp_path, "https://example.test")
    session = Session(created_at=datetime.now(UTC), consent=ConsentState.PENDING)
    (store.session_dir(session.id) / "session.json").write_text(session.model_dump_json(indent=2))
    return store, session.id


@pytest.mark.asyncio
async def test_intake_processor_persists_intake_and_compiles_persona(
    intake_store: tuple[LocalFilesystemStore, str],
) -> None:
    """User intake turns plus a practice transition should populate intake and persona."""
    store, session_id = intake_store
    bus = FrameBus(session_id)
    current_phase = Phase.INTAKE
    processor = IntakeProcessor(
        session_id,
        store,
        phase_getter=lambda: current_phase,
    )

    task = asyncio.create_task(processor.run(bus.subscribe()))
    await asyncio.sleep(0)
    await bus.publish(
        TranscriptDelta(
            session_id=session_id,
            utterance_id="u1",
            speaker=Speaker.USER,
            text="I need help negotiating a job offer with a recruiter.",
            is_final=True,
            ts_start=0.0,
            ts_end=0.1,
        )
    )
    await asyncio.sleep(0)
    await bus.publish(
        TranscriptDelta(
            session_id=session_id,
            utterance_id="u2",
            speaker=Speaker.USER,
            text="I want to ask for more salary and equity while staying calm.",
            is_final=True,
            ts_start=0.2,
            ts_end=0.3,
        )
    )
    await asyncio.sleep(0)
    current_phase = Phase.PRACTICE
    await bus.publish(
        PhaseSignal(
            session_id=session_id,
            from_phase=Phase.INTAKE,
            to_phase=Phase.PRACTICE,
            reason="cue",
            ts=0.4,
        )
    )
    await asyncio.sleep(0)
    await bus.aclose()
    await task

    manifest = json.loads((store.session_dir(session_id) / "session.json").read_text())
    assert manifest["intake"]["counterparty_relationship"] == "recruiter"
    assert "salary" in manifest["intake"]["user_goal"].lower()
    assert manifest["persona"]["relationship"] == "recruiter"
    assert "recruiter" in manifest["persona"]["personality_prompt"].lower()
