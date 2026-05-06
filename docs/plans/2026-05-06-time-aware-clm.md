# Time-aware CLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Inject a per-turn "time card" (phase, remaining time, word budget) into the CLM system payload so the model paces itself within phase budgets and the call stays under Hume's 5-minute cap.

**Architecture:** A new `TimeCard` dataclass + builder in `rehearse/agents/timecard.py`. `AnthropicCLMResponder.stream_reply` builds the card on every turn and sends Anthropic's `system` parameter as a list of two blocks: the existing static role prompt (with `cache_control: ephemeral`) and the volatile rendered time card.

**Tech Stack:** Python 3.12, Anthropic SDK (async streaming), pytest, pydantic.

**Spec:** `docs/specs/v2026-05-06-time-aware-clm.md`

---

## File Structure

- **Create:** `rehearse/agents/timecard.py` — `TimeCard` dataclass, `build_time_card(session, now, hard_cap_seconds)`, `render_time_card(card) -> str`.
- **Create:** `tests/test_timecard.py` — unit tests for builder + renderer.
- **Modify:** `rehearse/agents/clm.py` — `AnthropicCLMResponder.stream_reply` builds card and passes a list of system blocks.
- **Modify:** `tests/test_clm.py` — add a test that captures the outgoing `system` payload from a stubbed Anthropic client and asserts the two-block shape.

`timecard.py` is its own file because the builder is a pure function over `Session` + clock and benefits from being unit-testable without touching CLM routing.

---

## Task 1: TimeCard dataclass and builder

**Files:**
- Create: `rehearse/agents/timecard.py`
- Test: `tests/test_timecard.py`

- [ ] **Step 1: Write the failing test for the builder**

Create `tests/test_timecard.py`:

```python
"""Verify the per-turn time card built from a live session manifest."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from rehearse.agents.timecard import (
    HARD_CAP_SECONDS,
    TimeCard,
    build_time_card,
    render_time_card,
)
from rehearse.types import ConsentState, Phase, PhaseTiming, Session

_T0 = datetime(2026, 5, 6, 12, 0, 0, tzinfo=UTC)


def _session(*timings: PhaseTiming) -> Session:
    return Session(
        created_at=_T0,
        consent=ConsentState.GRANTED,
        phase_timings=list(timings),
    )


def test_build_time_card_intake_start():
    session = _session(
        PhaseTiming(phase=Phase.INTAKE, started_at=_T0, budget_seconds=60),
    )
    card = build_time_card(session, now=_T0, hard_cap_seconds=HARD_CAP_SECONDS)
    assert card.phase == Phase.INTAKE
    assert card.seconds_elapsed_in_phase == 0
    assert card.seconds_remaining_in_phase == 60
    assert card.seconds_remaining_in_call == 300
    assert 15 <= card.word_budget_this_turn <= 80


def test_build_time_card_practice_midway():
    started = _T0 + timedelta(seconds=60)
    session = _session(
        PhaseTiming(
            phase=Phase.INTAKE, started_at=_T0, ended_at=started, budget_seconds=60
        ),
        PhaseTiming(phase=Phase.PRACTICE, started_at=started, budget_seconds=180),
    )
    now = started + timedelta(seconds=90)
    card = build_time_card(session, now=now, hard_cap_seconds=HARD_CAP_SECONDS)
    assert card.phase == Phase.PRACTICE
    assert card.seconds_elapsed_in_phase == 90
    assert card.seconds_remaining_in_phase == 90
    assert card.seconds_remaining_in_call == 150


def test_build_time_card_clamps_word_budget_floor():
    started = _T0
    session = _session(
        PhaseTiming(phase=Phase.PRACTICE, started_at=started, budget_seconds=180),
    )
    now = started + timedelta(seconds=178)
    card = build_time_card(session, now=now, hard_cap_seconds=HARD_CAP_SECONDS)
    assert card.word_budget_this_turn == 15


def test_build_time_card_after_hard_cap():
    started = _T0
    session = _session(
        PhaseTiming(phase=Phase.FEEDBACK, started_at=started, budget_seconds=60),
    )
    now = started + timedelta(seconds=400)
    card = build_time_card(session, now=now, hard_cap_seconds=HARD_CAP_SECONDS)
    assert card.seconds_remaining_in_call == 0
    assert card.seconds_remaining_in_phase == 0


def test_render_time_card_contains_phase_remaining_and_words():
    card = TimeCard(
        phase=Phase.PRACTICE,
        seconds_elapsed_in_phase=42,
        seconds_remaining_in_phase=138,
        seconds_remaining_in_call=198,
        phase_budget_seconds=180,
        word_budget_this_turn=36,
    )
    rendered = render_time_card(card)
    assert "practice" in rendered.lower()
    assert "2:18" in rendered
    assert "3:18" in rendered
    assert "36" in rendered
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_timecard.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rehearse.agents.timecard'`

- [ ] **Step 3: Implement `timecard.py`**

Create `rehearse/agents/timecard.py`:

```python
"""Build and render a per-turn time card injected into the CLM system payload.

The card lets the language model see which phase the call is in, how much time
remains in the phase and the call, and how many words it should aim to speak
on the current turn. It is recomputed for every CLM request from the session
manifest's phase timings plus the current clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from rehearse.types import Phase, Session

HARD_CAP_SECONDS: int = 300
"""Hume's per-call model-provider cap. Calls must end before this elapses."""

_AVG_TURN_SECONDS: float = 15.0
_WORDS_PER_SECOND: float = 160.0 / 60.0
_WORD_BUDGET_FLOOR: int = 15
_WORD_BUDGET_CEIL: int = 80

_MODEL_SHARE: dict[Phase, float] = {
    Phase.INTAKE: 0.50,
    Phase.PRACTICE: 0.30,
    Phase.FEEDBACK: 0.70,
}


@dataclass(frozen=True)
class TimeCard:
    """Snapshot of where the call is in time, computed once per CLM turn."""

    phase: Phase
    seconds_elapsed_in_phase: int
    seconds_remaining_in_phase: int
    seconds_remaining_in_call: int
    phase_budget_seconds: int
    word_budget_this_turn: int


def build_time_card(
    session: Session,
    *,
    now: datetime,
    hard_cap_seconds: int = HARD_CAP_SECONDS,
) -> TimeCard:
    """Compute the live time card for one CLM turn."""
    open_timing = _open_phase_timing(session)
    elapsed_in_phase = max(0, int((now - open_timing.started_at).total_seconds()))
    remaining_in_phase = max(0, open_timing.budget_seconds - elapsed_in_phase)
    call_started_at = session.phase_timings[0].started_at
    elapsed_in_call = max(0, int((now - call_started_at).total_seconds()))
    remaining_in_call = max(0, hard_cap_seconds - elapsed_in_call)
    return TimeCard(
        phase=open_timing.phase,
        seconds_elapsed_in_phase=elapsed_in_phase,
        seconds_remaining_in_phase=remaining_in_phase,
        seconds_remaining_in_call=remaining_in_call,
        phase_budget_seconds=open_timing.budget_seconds,
        word_budget_this_turn=_word_budget(open_timing.phase, remaining_in_phase),
    )


def render_time_card(card: TimeCard) -> str:
    """Return the system-block text the model sees for this turn."""
    return (
        "Live timing\n"
        f"- Phase: {card.phase.value} ({_mmss(card.phase_budget_seconds)} budget)\n"
        f"- Elapsed in phase: {_mmss(card.seconds_elapsed_in_phase)}\n"
        f"- Remaining in phase: {_mmss(card.seconds_remaining_in_phase)}\n"
        f"- Remaining in call: {_mmss(card.seconds_remaining_in_call)} (hard cap)\n"
        f"- Target length for THIS reply: ~{card.word_budget_this_turn} words\n"
        "Speak only as long as the target. When phase time is nearly up, "
        "land the current beat in one closing sentence."
    )


def _word_budget(phase: Phase, remaining_in_phase: int) -> int:
    share = _MODEL_SHARE.get(phase, 0.5)
    expected_turns_left = max(1.0, remaining_in_phase / _AVG_TURN_SECONDS)
    raw = (remaining_in_phase * share * _WORDS_PER_SECOND) / expected_turns_left
    return max(_WORD_BUDGET_FLOOR, min(_WORD_BUDGET_CEIL, int(round(raw))))


def _open_phase_timing(session: Session):
    if not session.phase_timings:
        raise ValueError("session has no phase_timings; PhaseProcessor.bootstrap not run")
    for timing in reversed(session.phase_timings):
        if timing.ended_at is None:
            return timing
    return session.phase_timings[-1]


def _mmss(seconds: int) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_timecard.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add rehearse/agents/timecard.py tests/test_timecard.py
git commit -m "add TimeCard builder and renderer for per-turn CLM context"
```

---

## Task 2: Wire TimeCard into AnthropicCLMResponder as cached system blocks

**Files:**
- Modify: `rehearse/agents/clm.py` — `AnthropicCLMResponder.__init__` and `stream_reply`
- Modify: `tests/test_clm.py` — add a test for the outgoing `system` payload shape

- [ ] **Step 1: Write the failing test**

Add to the bottom of `tests/test_clm.py` (keep existing imports; add what's missing):

```python
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from rehearse.agents.clm import AnthropicCLMResponder, CLMChatRequest, CLMMessage
from rehearse.storage import LocalFilesystemStore
from rehearse.types import (
    ConsentState,
    CounterpartyPersona,
    Phase,
    PhaseTiming,
    Session,
)


class _FakeAnthropicStream:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    @property
    def text_stream(self):
        async def _gen():
            yield "ok"
        return _gen()


class _CapturingMessages:
    def __init__(self):
        self.last_kwargs: dict | None = None

    def stream(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeAnthropicStream()


class _CapturingClient:
    def __init__(self):
        self.messages = _CapturingMessages()


@pytest.mark.asyncio
async def test_anthropic_responder_sends_two_system_blocks(tmp_path):
    started = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)
    session = Session(
        id="sess_test",
        created_at=started,
        consent=ConsentState.GRANTED,
        persona=CounterpartyPersona(
            relationship="manager",
            communication_style="direct",
            likely_reactions=["pushes back"],
            stated_goal="ship",
        ),
        phase_timings=[
            PhaseTiming(phase=Phase.INTAKE, started_at=started, budget_seconds=60),
        ],
    )
    store = LocalFilesystemStore(root=tmp_path, public_base_url="https://example.test")
    await store.write("sess_test", "session.json", session.model_dump_json())

    responder = AnthropicCLMResponder(
        api_key="test", model="claude-test", store=store
    )
    fake = _CapturingClient()
    responder._client = fake  # type: ignore[attr-defined]

    request = CLMChatRequest(messages=[CLMMessage(role="user", content="hi")])
    chunks = []
    async for chunk in responder.stream_reply(
        session_id="sess_test", role="coach", request=request
    ):
        chunks.append(chunk)

    kwargs = fake.messages.last_kwargs
    assert kwargs is not None
    system = kwargs["system"]
    assert isinstance(system, list)
    assert len(system) == 2
    assert system[0]["type"] == "text"
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert "Live timing" in system[1]["text"]
    assert system[1].get("cache_control") is None
```

The store API is just `write(session_id, name, bytes_or_str)` / `read(...)` / `update_session(...)` — see `rehearse/storage.py`. The test only needs `session.json` on disk so `_load_session` resolves it.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_clm.py::test_anthropic_responder_sends_two_system_blocks -v`
Expected: FAIL — current code passes `system` as a string, not a list of blocks.

- [ ] **Step 3: Modify `AnthropicCLMResponder`**

In `rehearse/agents/clm.py`, change the imports near the top to add:

```python
from datetime import datetime
from typing import Any, Callable, Protocol

from rehearse.agents.timecard import build_time_card, render_time_card
from rehearse.session import utcnow
```

Change `AnthropicCLMResponder.__init__` to accept an optional clock:

```python
class AnthropicCLMResponder:
    """Wrap Claude so Hume can use it as the live conversation brain."""

    def __init__(
        self,
        api_key: str,
        model: str,
        store: LocalFilesystemStore,
        *,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        """Store Anthropic credentials and create the async client lazily."""
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model
        self._store = store
        self._clock = clock
```

Replace the body of `stream_reply` with:

```python
    async def stream_reply(
        self,
        *,
        session_id: str | None,
        role: str,
        request: CLMChatRequest,
    ) -> AsyncIterator[str]:
        """Yield text chunks from Anthropic's streaming messages API."""
        session = await _load_session(session_id, self._store)
        static_prompt = _system_prompt_for_role(role, session)
        if session_id:
            static_prompt = f"{static_prompt}\n\nSession ID: {session_id}"
        messages = _anthropic_messages(request.messages)
        if not messages:
            messages = [{"role": "user", "content": "Greet the caller and start the coaching."}]

        system_blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": static_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        if session is not None and session.phase_timings:
            card = build_time_card(session, now=self._clock())
            system_blocks.append({"type": "text", "text": render_time_card(card)})

        async with self._client.messages.stream(
            model=self._model,
            max_tokens=512,
            temperature=0.4,
            system=system_blocks,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                if text:
                    yield text
```

- [ ] **Step 4: Run the new test to verify it passes**

Run: `uv run pytest tests/test_clm.py::test_anthropic_responder_sends_two_system_blocks -v`
Expected: PASS.

- [ ] **Step 5: Run the full clm + timecard test files to check for regressions**

Run: `uv run pytest tests/test_clm.py tests/test_timecard.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add rehearse/agents/clm.py tests/test_clm.py
git commit -m "inject time card as second cached system block on every CLM turn"
```

---

## Task 3: Verify against the full test suite and lint

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -q`
Expected: all green. If any test fails because it builds `Session` without `phase_timings` and ends up in the new code path, fix it by giving the fixture a single `PhaseTiming` row — never by weakening the new behavior.

- [ ] **Step 2: Run the linter**

Run: `uv run ruff check rehearse/agents/timecard.py rehearse/agents/clm.py tests/test_timecard.py tests/test_clm.py`
Expected: no errors.

- [ ] **Step 3: Commit any lint fixes**

```bash
git add -u
git commit -m "lint fixes for time-aware CLM"
```

(Skip if there were no fixes.)

---

## Task 4: Manual smoke test on a live call

This is not automated. Do it before merging.

- [ ] **Step 1: Start the runtime locally**

Follow the existing README "run a local call" instructions. Place one call.

- [ ] **Step 2: Watch the session viewer**

Confirm phase transitions land at roughly 1:00 (intake→practice) and 4:00 (practice→feedback), and the call ends at or before 5:00. The model's replies should visibly shorten as `seconds_remaining_in_phase` drops.

- [ ] **Step 3: Spot-check Anthropic billing/console**

If the project surfaces cache hits in any log, confirm cache reads on turns 2+. Otherwise skip — the unit test already proves the cache_control flag is being sent.

---

## Self-review notes

- Spec sections covered: TimeCard (Task 1), word-budget formula (Task 1, `_word_budget`), rendered string (Task 1, `render_time_card`), integration in `_handle_clm_request`/`stream_reply` (Task 2), two-block system payload with cache control (Task 2), unit + integration tests (Tasks 1–2), manual verification (Task 4).
- Spec's `ScriptedCLMResponder` note is honored implicitly: only `AnthropicCLMResponder` changes; the scripted responder is untouched and still works as the keyless fallback.
- Open questions in the spec (`avg_turn_seconds` per-phase tuning; persona/goal in card) are deliberately deferred and not in the plan.
