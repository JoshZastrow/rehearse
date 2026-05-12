"""Multi-session memory eval: consent memory persists across separate call instances.

Simulates what happens in production when the same phone number calls twice:
  Session 1: first-time caller, hears full consent prompt, says "yes"
  Session 2: same caller, memory has prior consent, hears brief reminder
             and is immediately granted (no "yes" needed)

Each session is a fresh ConsentGate + FrameBus + store, but BOTH share the
same CallerMemory instance — exactly as telephony.py wires it in production
when memory is backed by a persistent store (Honcho, MCP server, etc.).

Verification uses two layers:
  deterministic — keyword matching on CONSENT_REMINDER text
  semantic      — LLM judge via Claude Haiku (live_api, requires ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rehearse.bus import FrameBus
from rehearse.consent import ConsentGate, ConsentGateConfig
from rehearse.frames import ConsentResolved, TranscriptDelta
from rehearse.memory import InMemoryCallerMemory
from rehearse.participants import SpeakRequest
from rehearse.personas import CONSENT_PROMPT, CONSENT_REMINDER
from rehearse.storage import LocalFilesystemStore
from rehearse.types import ConsentState, Session, Speaker

CALLER_HASH = "multi-session-test-caller"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path) -> tuple[LocalFilesystemStore, str]:
    store = LocalFilesystemStore(tmp_path / f"s-{id(object())}", "https://example.test")
    session = Session(created_at=datetime.now(UTC), consent=ConsentState.PENDING)
    (store.session_dir(session.id) / "session.json").write_text(
        session.model_dump_json(indent=2)
    )
    return store, session.id


def _user_frame(session_id: str, text: str = "yes") -> TranscriptDelta:
    return TranscriptDelta(
        session_id=session_id,
        utterance_id="u-1",
        speaker=Speaker.USER,
        text=text,
        is_final=True,
        ts_start=1.0,
        ts_end=1.5,
    )


class _Speaker:
    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.said = asyncio.Event()

    async def say(self, request: SpeakRequest) -> None:
        self.spoken.append(request.text)
        self.said.set()


async def _run_gate(
    tmp_path: Path,
    caller_hash: str,
    memory: InMemoryCallerMemory,
    *,
    response: str | None = "yes",  # None = no response (auto-granted or timeout)
) -> tuple[list[str], list[ConsentResolved]]:
    """Run one ConsentGate call to completion. Returns (spoken_lines, resolved_frames)."""
    store, session_id = _make_store(tmp_path)
    bus = FrameBus(session_id)
    speaker = _Speaker()
    resolved: list[ConsentResolved] = []

    async def _collect() -> None:
        async for frame in bus.subscribe():
            if isinstance(frame, ConsentResolved):
                resolved.append(frame)

    async def _noop() -> None:
        pass

    collector = asyncio.create_task(_collect())
    gate = ConsentGate(
        session_id,
        store,
        bus,
        speaker=speaker,
        on_decline=_noop,
        config=ConsentGateConfig(prompt_timeout_seconds=5, reprompt_limit=0),
        caller_hash=caller_hash,
        memory=memory,
    )
    task = asyncio.create_task(gate.run(bus.subscribe()))
    await asyncio.wait_for(speaker.said.wait(), timeout=5.0)
    if response is not None:
        await bus.publish(_user_frame(session_id, response))
    await asyncio.wait_for(task, timeout=10.0)
    await bus.aclose()
    await collector
    return speaker.spoken, resolved


# ---------------------------------------------------------------------------
# Deterministic tests (no external services)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_session_hears_full_prompt(tmp_path: Path) -> None:
    """Session 1: new caller hears the full consent prompt."""
    memory = InMemoryCallerMemory()
    spoken, resolved = await _run_gate(tmp_path, CALLER_HASH, memory)

    assert spoken[0] == CONSENT_PROMPT
    assert resolved and resolved[0].state == ConsentState.GRANTED


@pytest.mark.asyncio
async def test_second_session_hears_brief_reminder_and_is_auto_granted(
    tmp_path: Path,
) -> None:
    """Session 2 — same caller, shared memory, separate gate instance.

    This is the core multi-session guarantee: a persistent memory backend
    carries consent from one call to the next. The second call:
      - hears CONSENT_REMINDER (not the full 55-word prompt)
      - is granted immediately (no spoken "yes" required)
    """
    memory = InMemoryCallerMemory()

    # Session 1: grant consent.
    await _run_gate(tmp_path, CALLER_HASH, memory, response="yes")
    assert await memory.has_prior_consent(CALLER_HASH)

    # Session 2: separate gate, same memory — no "yes" needed.
    spoken, resolved = await _run_gate(tmp_path, CALLER_HASH, memory, response=None)

    # Keyword check.
    assert spoken == [CONSENT_REMINDER], f"Returning caller got: {spoken}"
    assert CONSENT_PROMPT not in spoken[0]
    assert resolved and resolved[0].state == ConsentState.GRANTED


@pytest.mark.asyncio
async def test_declined_consent_not_remembered(tmp_path: Path) -> None:
    """A caller who declined must still get the full prompt on the next call."""
    memory = InMemoryCallerMemory()

    # Session 1: user says "no".
    spoken1, resolved1 = await _run_gate(tmp_path, CALLER_HASH, memory, response="no")
    assert spoken1[0] == CONSENT_PROMPT
    assert resolved1 and resolved1[0].state == ConsentState.DECLINED
    assert not await memory.has_prior_consent(CALLER_HASH)

    # Session 2: still gets full prompt, must say "yes" again.
    spoken2, resolved2 = await _run_gate(tmp_path, CALLER_HASH, memory, response="yes")
    assert spoken2[0] == CONSENT_PROMPT, "Declined caller should re-hear full prompt"
    assert resolved2 and resolved2[0].state == ConsentState.GRANTED


@pytest.mark.asyncio
async def test_different_callers_have_independent_memory(tmp_path: Path) -> None:
    """Two distinct phone hashes do not share consent state."""
    memory = InMemoryCallerMemory()
    caller_a, caller_b = "caller-aaa", "caller-bbb"

    await _run_gate(tmp_path, caller_a, memory, response="yes")

    # Caller B is still new.
    spoken_b, _ = await _run_gate(tmp_path, caller_b, memory, response="yes")
    assert spoken_b[0] == CONSENT_PROMPT

    # Caller A on a second call gets the reminder.
    spoken_a2, _ = await _run_gate(tmp_path, caller_a, memory, response=None)
    assert spoken_a2[0] == CONSENT_REMINDER


# ---------------------------------------------------------------------------
# LLM judge (live_api — requires ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.live_api
async def test_llm_judge_confirms_second_call_is_brief_reminder(tmp_path: Path) -> None:
    """Claude verifies the second-call text sounds like a brief reminder, not a consent ask.

    Two-layer check:
      1. Keyword matching (fast, free, deterministic)
      2. LLM semantic judge (confirms meaning, not just text)
    """
    import anthropic
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    memory = InMemoryCallerMemory()
    await _run_gate(tmp_path, CALLER_HASH, memory, response="yes")
    spoken, _ = await _run_gate(tmp_path, CALLER_HASH, memory, response=None)
    reminder_text = spoken[0] if spoken else ""

    # Layer 1: keyword match.
    assert "recorded" in reminder_text.lower() or CONSENT_REMINDER in reminder_text, (
        f"Expected reminder content, got: {reminder_text!r}"
    )

    # Layer 2: LLM judge.
    client = anthropic.AsyncAnthropic(api_key=api_key)
    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[
            {
                "role": "user",
                "content": (
                    f'Text spoken at the start of a phone call: "{reminder_text}"\n\n'
                    "Is this a BRIEF reminder that the call will be recorded — "
                    "NOT a full consent prompt asking the listener to say yes or no? "
                    "Answer YES or NO only."
                ),
            }
        ],
    )
    verdict = response.content[0].text.strip().upper()
    assert "YES" in verdict, (
        f"LLM judge expected YES (brief reminder) but got: {verdict!r}\n"
        f"Spoken text: {reminder_text!r}"
    )
