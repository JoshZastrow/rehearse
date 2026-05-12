# rehearse — Spec: Consent Caller Memory (v1)

**Status**: draft
**Owner**: jz
**Date**: 2026-05-12
**Depends on**: `rehearse/consent.py`, `rehearse/personas.py`, `rehearse/config.py`,
`rehearse/telephony.py`, `rehearse/session.py`
**Supersedes**: nothing

---

## 0. One-line summary

Returning callers hear a short "again, this call will be transcribed" reminder
instead of the full consent prompt. First-time callers receive the full prompt
unchanged.

---

## 1. Goal

The full consent prompt is 55 words. Reading it to every caller on every call
creates unnecessary friction for people who have already agreed. A one-sentence
reminder is enough on subsequent calls.

This spec also establishes the `CallerMemory` interface — a standard contract
that all Rehearse agents (consent, intake, coach, feedback) can use to recall
facts about a caller across sessions. v1 stores only one fact: whether this
phone number has previously granted consent.

---

## 2. Non-goals

- Long-term conversation history or session summaries for intake/feedback (v2).
- Revoking consent memory (future compliance feature).
- Caller profiles beyond the consent flag (v2).
- Running Honcho locally / self-hosted (v1 uses cloud API; fallback is
  `NullCallerMemory` when `HONCHO_API_KEY` is absent).

---

## 3. Design commitments

1. **`CallerMemory` is a `Protocol`.** Any class with the right two async
   methods satisfies it. Production uses Honcho; tests use an in-process dict.
   No inheritance required from any agent.

2. **Honcho is optional.** When `HONCHO_API_KEY` is not set, `NullCallerMemory`
   is used. The call still works; callers just always hear the full prompt. Zero
   failure modes from missing credentials.

3. **Caller identity is the phone number hash.** `Session.phone_number_hash`
   (SHA-256[:16] of the E.164 number) is the only stable cross-session
   identifier. No plaintext numbers leave the runtime.

4. **Tests run without Honcho.** `InMemoryCallerMemory` satisfies the protocol.
   The full test suite passes with no external services.

5. **`ConsentGate` is unchanged for existing callers of the class.** Both new
   params (`caller_hash`, `memory`) are optional keyword-only. Existing
   instantiation sites that omit them continue to work and always receive the
   full prompt.

---

## 4. `CallerMemory` interface

```python
# rehearse/memory.py

class CallerMemory(Protocol):
    async def has_prior_consent(self, caller_hash: str) -> bool:
        """Return True if this caller has previously granted consent."""
        ...

    async def record_consent(self, caller_hash: str) -> None:
        """Persist that this caller granted consent."""
        ...
```

### Implementations

| Class | Backend | Use |
|---|---|---|
| `NullCallerMemory` | None | Default; no-op; caller always gets full prompt |
| `InMemoryCallerMemory` | `set[str]` | Tests; shared across calls in the same process |
| `HonchoCallerMemory` | Honcho cloud API | Production when `HONCHO_API_KEY` is set |

---

## 5. Honcho mapping

| Honcho concept | Rehearse meaning |
|---|---|
| Workspace | `"rehearse"` (configurable via `HONCHO_WORKSPACE_ID`) |
| Peer | One per caller, keyed by `phone_number_hash` |
| Session | One per call (`session_id`) |
| Message | `content="consent_granted"` stored on consent |

**`has_prior_consent(caller_hash)`**:
Get-or-create the peer for `caller_hash`. List their sessions. For each
session, check messages for `content="consent_granted"`. Return True if any
exists.

**`record_consent(caller_hash)`**:
Get-or-create the peer + a session keyed by the current call. Add a message
`{"peer_id": caller_hash, "content": "consent_granted"}`.

Honcho SDK is synchronous; all calls wrapped in `asyncio.to_thread`.

---

## 6. `ConsentGate` changes

New optional kwargs on `__init__`:

```python
caller_hash: str | None = None,
memory: CallerMemory | None = None,
```

`run()` selects the prompt before speaking:

```python
prompt = CONSENT_PROMPT
if self._caller_hash and self._memory:
    if await self._memory.has_prior_consent(self._caller_hash):
        prompt = CONSENT_REMINDER
await self._ask(prompt)
```

`_grant()` records consent after persisting to the manifest:

```python
if self._caller_hash and self._memory:
    await self._memory.record_consent(self._caller_hash)
```

New constant in `rehearse/personas.py`:

```python
CONSENT_REMINDER = (
    "Again, this call will be transcribed so I can give you feedback. "
    "Just say yes to continue."
)
```

---

## 7. Configuration

Two new optional fields on `RuntimeConfig`:

```python
honcho_api_key: str | None = None      # from HONCHO_API_KEY
honcho_workspace_id: str = "rehearse"  # from HONCHO_WORKSPACE_ID
```

`telephony.py` builds the memory object once per call:

```python
memory = (
    HonchoCallerMemory(config.honcho_api_key, config.honcho_workspace_id)
    if config.honcho_api_key
    else NullCallerMemory()
)
```

---

## 8. File inventory

| File | Change |
|---|---|
| `rehearse/memory.py` | **New.** Protocol + 3 implementations |
| `rehearse/personas.py` | Add `CONSENT_REMINDER` |
| `rehearse/consent.py` | `caller_hash` + `memory` kwargs; prompt branch; record on grant |
| `rehearse/config.py` | `honcho_api_key`, `honcho_workspace_id` fields |
| `rehearse/telephony.py` | Read caller hash from session; build memory; pass to gate |
| `pyproject.toml` | Add `honcho-ai` dependency |
| `tests/test_consent_memory.py` | **New.** Two tests: first-time vs returning caller |

---

## 9. Acceptance criteria

1. `test_first_time_caller_gets_full_consent_prompt` passes: speaker receives
   `CONSENT_PROMPT` when no prior consent exists in memory.

2. `test_returning_caller_gets_brief_reminder` passes: speaker receives
   `CONSENT_REMINDER` when prior consent exists in memory.

3. `uv run pytest tests/ -q` — full suite passes with zero `HONCHO_API_KEY`
   set.

4. `isinstance(InMemoryCallerMemory(), CallerMemory)` is `True` at runtime
   (Protocol is `@runtime_checkable`).
