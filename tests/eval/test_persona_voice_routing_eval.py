"""3-scenario persona voice routing eval.

Verifies that the intake coach asks about gender preference exactly when needed
(first call per topic, not on repeat topics), and that IntakeAwareRouter
selects the correct character agent.

Requires:
    make deploy-judge     # deploy LLM judge
    make serve            # ngrok + Honcho + rehearse server running
    VLLM_BASE_URL, VLLM_API_KEY set in .env
    HONCHO_BASE_URL set in .env (local or cloud)

Run:
    make eval-persona-routing
    # or:
    uv run pytest tests/eval/test_persona_voice_routing_eval.py \
        -v -m "live_api and live_honcho" --timeout=180
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest

from rehearse.memory import HonchoCallerMemory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def honcho_memory() -> HonchoCallerMemory:
    base_url = os.environ.get("HONCHO_BASE_URL", "http://localhost:8001")
    return HonchoCallerMemory(
        api_key=os.environ.get("HONCHO_API_KEY", ""),
        workspace_id="rehearse-test",
        base_url=base_url,
    )


@pytest.fixture(scope="module")
def test_caller_hash() -> str:
    return f"eval-caller-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module", autouse=True)
async def clear_test_caller(honcho_memory, test_caller_hash):
    """Wipe per-topic prefs before the eval run for a clean slate."""
    await honcho_memory.clear_caller(test_caller_hash)
    yield
    # Optional: clean up after — leave data for post-run inspection.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _routing_result(session_dir: Path) -> dict:
    """Read the routing eval result artifact from a completed session."""
    result_path = session_dir / "routing_eval_result.json"
    return json.loads(result_path.read_text())


def _run_scenario(
    situation: str,
    caller_hash: str,
    caller_gender_answer: str | None = None,
) -> Path:
    """Run one synthetic call scenario and return the session directory.

    This is a placeholder for the full eval harness invocation.
    Replace with real harness call (e.g., RuntimeSandbox + LLMCustomer).
    """
    raise NotImplementedError(
        "Implement with RuntimeSandbox + LLMCustomer when harness is wired."
    )


# ---------------------------------------------------------------------------
# Scenario 1: New caller, work topic → gender question asked, male answer
# ---------------------------------------------------------------------------


@pytest.mark.live_api
@pytest.mark.live_honcho
@pytest.mark.asyncio
async def test_new_caller_is_asked_gender_question(
    honcho_memory: HonchoCallerMemory,
    test_caller_hash: str,
    tmp_path: Path,
) -> None:
    """First call ever: intake asks gender question, caller picks male."""
    # No prior preferences exist (cleared in fixture).
    pref = await honcho_memory.get_agent_preference(test_caller_hash, "work")
    assert pref is None, "Test caller should have no prior preference"

    session_dir = _run_scenario(
        situation="I need to ask my manager for a salary raise",
        caller_hash=test_caller_hash,
        caller_gender_answer="male",
    )

    result = _routing_result(session_dir)
    assert result["gender_question_asked"] is True, "Coach must ask gender question on first call"
    assert result["agent_selected"] == "male_character", f"Expected male_character, got {result['agent_selected']}"
    assert result["routing_correct"] is True, result.get("reasoning", "")

    # Memory should now have the work→male preference persisted.
    pref = await honcho_memory.get_agent_preference(test_caller_hash, "work")
    assert pref == "male", "Honcho must persist work→male preference"


# ---------------------------------------------------------------------------
# Scenario 2: Same caller, same topic (work) → NO gender question
# ---------------------------------------------------------------------------


@pytest.mark.live_api
@pytest.mark.live_honcho
@pytest.mark.asyncio
async def test_returning_caller_same_topic_skips_gender_question(
    honcho_memory: HonchoCallerMemory,
    test_caller_hash: str,
    tmp_path: Path,
) -> None:
    """Second call, same topic: preference is recalled, question is NOT asked."""
    # Precondition: scenario 1 must have run first (work→male stored).
    pref = await honcho_memory.get_agent_preference(test_caller_hash, "work")
    if pref is None:
        pytest.skip("Scenario 1 must pass first (work→male preference required)")

    session_dir = _run_scenario(
        situation="I want to discuss a promotion with my manager",
        caller_hash=test_caller_hash,
        caller_gender_answer=None,  # should not be asked
    )

    result = _routing_result(session_dir)
    assert result["gender_question_asked"] is False, (
        "Coach must NOT ask gender question when preference is already stored"
    )
    assert result["agent_selected"] == "male_character"
    assert result["routing_correct"] is True, result.get("reasoning", "")


# ---------------------------------------------------------------------------
# Scenario 3: Same caller, new topic (relationship) → gender question asked again
# ---------------------------------------------------------------------------


@pytest.mark.live_api
@pytest.mark.live_honcho
@pytest.mark.asyncio
async def test_new_topic_triggers_gender_question(
    honcho_memory: HonchoCallerMemory,
    test_caller_hash: str,
    tmp_path: Path,
) -> None:
    """Third call, new topic (relationship): question asked again, caller picks female."""
    pref = await honcho_memory.get_agent_preference(test_caller_hash, "relationship")
    assert pref is None, "No relationship preference should exist yet"

    session_dir = _run_scenario(
        situation="I want to talk to my partner about moving in together",
        caller_hash=test_caller_hash,
        caller_gender_answer="female",
    )

    result = _routing_result(session_dir)
    assert result["gender_question_asked"] is True, (
        "Coach must ask gender question for a new topic category"
    )
    assert result["agent_selected"] == "female_character"
    assert result["routing_correct"] is True, result.get("reasoning", "")

    pref = await honcho_memory.get_agent_preference(test_caller_hash, "relationship")
    assert pref == "female", "Honcho must persist relationship→female preference"


# ---------------------------------------------------------------------------
# Memory state snapshot (post-run assertion)
# ---------------------------------------------------------------------------


@pytest.mark.live_api
@pytest.mark.live_honcho
@pytest.mark.asyncio
async def test_memory_state_after_three_calls(
    honcho_memory: HonchoCallerMemory,
    test_caller_hash: str,
) -> None:
    """After all 3 scenarios, Honcho has both topic preferences persisted."""
    work_pref = await honcho_memory.get_agent_preference(test_caller_hash, "work")
    rel_pref = await honcho_memory.get_agent_preference(test_caller_hash, "relationship")
    assert work_pref == "male", f"Expected work→male, got {work_pref!r}"
    assert rel_pref == "female", f"Expected relationship→female, got {rel_pref!r}"
