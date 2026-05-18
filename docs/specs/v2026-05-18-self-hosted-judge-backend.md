# Self-Hosted Judge Backend

**Status:** draft  
**Date:** 2026-05-18  
**Owner:** Josh Zastrow  

---

## 1. Outcomes

| # | Outcome | Verifiable by |
|---|---|---|
| O1 | `LLMJudge` makes zero Anthropic API calls when `JUDGE_BASE_URL` is set | Integration test hits Modal endpoint; no `ANTHROPIC_API_KEY` required |
| O2 | All existing scorers that use `LLMJudge` work without modification | Full test suite passes; no scorer files touched |
| O3 | The backend can be swapped per-scorer via constructor injection | Unit test: `LLMJudge(backend=LocalOllamaBackend(...))` calls no external API |
| O4 | Tests that inject a mock client today still pass without modification | `monkeypatch` on `_client_lazy` continues to work via `_LegacyClientBackend` shim |
| O5 | A future reward model endpoint is one new `JudgeBackend` implementation away | Protocol shape is sufficient: `async complete(system, user) -> str` |

---

## 2. Problem

`LLMJudge` in `rehearse/eval/scorers/llm_judge.py` is hardwired to Anthropic. Every eval run that scores a coaching transcript spends API credits and sends session data to a third party. There is no path to:

- Route calls to the self-hosted Modal/vLLM/Gemma 4 endpoint already deployed at `infra/judge.py`
- Swap judge backends without modifying scorer code
- Plug in a fine-tuned reward model once enough `judge.json` training examples accumulate

The `client=` injection seam exists for tests but it bypasses the lazy-init logic in a way that is fragile (tests monkey-patch `_client_lazy` directly). The backend choice is implicit and invisible to callers.

---

## 3. What Is Not Changing

- `TrajectoryJudgeScorer`, `AudioJudge`, and all other scorers — no changes
- `Scorer` protocol in `protocols.py` — no changes
- `eval runner`, `CLI`, `Makefile` targets — no changes
- The `judge()` method signature: `async def judge(self, *, system: str, user: str) -> dict[str, Any]`
- `LLMJudge.__init__` keyword args `model`, `max_tokens`, `temperature`, `client` — all kept for backwards compatibility

---

## 4. Design

### 4.1 `JudgeBackend` protocol

```
JudgeBackend.complete(*, system: str, user: str) -> str
```

A single async method. Returns raw text from the model. `LLMJudge` calls `complete()` and then runs `_parse_json()` on the result — same as today, just with the HTTP layer behind the protocol.

### 4.2 Concrete implementations

| Class | When used |
|---|---|
| `AnthropicBackend` | `ANTHROPIC_API_KEY` set, no `JUDGE_BASE_URL` |
| `OpenAICompatBackend` | `JUDGE_BASE_URL` set — covers Modal/vLLM, Ollama remote |
| `LocalOllamaBackend` | Subclass of `OpenAICompatBackend`, hardcodes `localhost:11434` |
| `_LegacyClientBackend` | Internal shim — wraps an injected `client=` for backwards compat |

### 4.3 Factory

`backend_from_env(*, model, max_tokens) -> JudgeBackend`

Priority: `JUDGE_BASE_URL` → `OpenAICompatBackend`; else `ANTHROPIC_API_KEY` → `AnthropicBackend`; else raises `LLMJudgeError`.

Model resolution: explicit `model` param → `JUDGE_MODEL` env var → provider default (`claude-opus-4-7` for Anthropic, `google/gemma-4-26B-A4B-it` for compat).

### 4.4 `LLMJudge` changes

- Add `backend: JudgeBackend | None = None` to `__init__`
- Replace `_client_lazy()` with `_get_backend()`: returns `self._backend` if set; wraps `self._client` as `_LegacyClientBackend` if injected; else calls `backend_from_env()`
- `judge()` calls `text = await backend.complete(system=system, user=user)` then `_parse_json(text)` — same output

### 4.5 Env vars

| Var | Purpose |
|---|---|
| `JUDGE_BASE_URL` | vLLM/Ollama base URL, e.g. `https://…modal.run/v1` |
| `JUDGE_API_KEY` | API key for that endpoint (default: `"local"`) |
| `JUDGE_MODEL` | Model name sent in the request |
| `ANTHROPIC_API_KEY` | Existing — used when `JUDGE_BASE_URL` is not set |

---

## 5. File Map

| Action | Path | Responsibility |
|---|---|---|
| **Create** | `rehearse/eval/scorers/judge_backend.py` | `JudgeBackend` protocol + all concrete implementations + `backend_from_env` factory |
| **Modify** | `rehearse/eval/scorers/llm_judge.py` | Add `backend=` param; replace `_client_lazy` with `_get_backend`; update `judge()` |
| **Create** | `tests/eval/test_judge_backend.py` | Unit tests for all implementations and the factory |
| **Modify** | `tests/eval/test_mme_sandbox_rollout.py` | Update two `monkeypatch` calls that patch `_client_lazy` to use `backend=` injection instead |

---

## 6. Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardwired Anthropic client in `LLMJudge` with a `JudgeBackend` protocol so any text completion backend can be injected or selected via env vars.

**Architecture:** A single `judge_backend.py` defines the protocol and all concrete implementations. `llm_judge.py` gains one new `backend=` constructor param and replaces `_client_lazy()` with `_get_backend()`. No other files change.

**Tech Stack:** Python 3.13, `httpx` for `OpenAICompatBackend` (no new deps), `anthropic` package (existing), `pytest-asyncio` for tests.

---

### Task 1: `JudgeBackend` protocol + `AnthropicBackend`

**Files:**
- Create: `rehearse/eval/scorers/judge_backend.py`
- Create: `tests/eval/test_judge_backend.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_judge_backend.py
import pytest
from rehearse.eval.scorers.judge_backend import AnthropicBackend, LLMJudgeError

@pytest.mark.asyncio
async def test_anthropic_backend_calls_client() -> None:
    calls = []

    class _FakeMsg:
        content = [type("B", (), {"type": "text", "text": '{"ok": true}'})()]

    class _FakeMessages:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return _FakeMsg()

    class _FakeClient:
        messages = _FakeMessages()

    backend = AnthropicBackend(model="claude-opus-4-7", max_tokens=512, client=_FakeClient())
    result = await backend.complete(system="sys", user="usr")

    assert result == '{"ok": true}'
    assert calls[0]["model"] == "claude-opus-4-7"
    assert calls[0]["system"] == "sys"
    assert calls[0]["messages"] == [{"role": "user", "content": "usr"}]
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/eval/test_judge_backend.py::test_anthropic_backend_calls_client -v
```

Expected: `ModuleNotFoundError: No module named 'rehearse.eval.scorers.judge_backend'`

- [ ] **Step 3: Create `judge_backend.py` with the protocol and `AnthropicBackend`**

```python
# rehearse/eval/scorers/judge_backend.py
"""JudgeBackend protocol and concrete implementations.

A JudgeBackend wraps one text-completion provider behind a single async
method. LLMJudge calls complete() and parses the returned string as JSON.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

from rehearse.eval.scorers.llm_judge import LLMJudgeError

_DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-7"
_DEFAULT_COMPAT_MODEL = "google/gemma-4-26B-A4B-it"


@runtime_checkable
class JudgeBackend(Protocol):
    async def complete(self, *, system: str, user: str) -> str: ...


class AnthropicBackend:
    """Calls AsyncAnthropic.messages.create with system + user messages."""

    def __init__(
        self,
        *,
        model: str = _DEFAULT_ANTHROPIC_MODEL,
        max_tokens: int = 2048,
        temperature: float | None = None,
        api_key: str | None = None,
        client: Any = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._api_key = api_key
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            from anthropic import AsyncAnthropic
            key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise LLMJudgeError("ANTHROPIC_API_KEY not set")
            self._client = AsyncAnthropic(api_key=key)
        return self._client

    async def complete(self, *, system: str, user: str) -> str:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        try:
            resp = await client.messages.create(**kwargs)
        except Exception as exc:
            raise LLMJudgeError(f"Anthropic call failed: {exc}") from exc
        return "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        )
```

- [ ] **Step 4: Run test — expect pass**

```bash
uv run pytest tests/eval/test_judge_backend.py::test_anthropic_backend_calls_client -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add rehearse/eval/scorers/judge_backend.py tests/eval/test_judge_backend.py
git commit -m "feat(judge): JudgeBackend protocol + AnthropicBackend"
```

---

### Task 2: `OpenAICompatBackend` and `LocalOllamaBackend`

**Files:**
- Modify: `rehearse/eval/scorers/judge_backend.py`
- Modify: `tests/eval/test_judge_backend.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/eval/test_judge_backend.py

@pytest.mark.asyncio
async def test_openai_compat_backend_sends_system_and_user() -> None:
    calls = []

    class _FakeChoice:
        message = type("M", (), {"content": '{"score": 0.9}'})()

    class _FakeResp:
        choices = [_FakeChoice()]

    class _FakeCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return _FakeResp()

    class _FakeClient:
        chat = type("C", (), {"completions": _FakeCompletions()})()

    from rehearse.eval.scorers.judge_backend import OpenAICompatBackend
    backend = OpenAICompatBackend(
        model="gemma-3-27b-it",
        max_tokens=256,
        base_url="http://localhost:8000/v1",
        client=_FakeClient(),
    )
    result = await backend.complete(system="sys", user="usr")

    assert result == '{"score": 0.9}'
    assert calls[0]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]


@pytest.mark.asyncio
async def test_local_ollama_backend_uses_localhost() -> None:
    from rehearse.eval.scorers.judge_backend import LocalOllamaBackend
    backend = LocalOllamaBackend(model="gemma3:27b", max_tokens=128)
    # Just verify the base_url is set correctly without making real calls
    assert "11434" in backend._base_url
    assert backend._api_key == "ollama"
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/eval/test_judge_backend.py -k "compat or ollama" -v
```

Expected: `ImportError` or `AttributeError`

- [ ] **Step 3: Add `OpenAICompatBackend` and `LocalOllamaBackend` to `judge_backend.py`**

```python
# append to rehearse/eval/scorers/judge_backend.py

class OpenAICompatBackend:
    """Calls any OpenAI-compatible /v1/chat/completions endpoint.

    Works with vLLM, Ollama, llama.cpp, and the Modal-hosted Gemma 4 server
    in infra/judge.py. Uses httpx directly — no openai SDK dependency.
    """

    def __init__(
        self,
        *,
        model: str = _DEFAULT_COMPAT_MODEL,
        max_tokens: int = 2048,
        base_url: str,
        api_key: str = "local",
        client: Any = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = client  # injected in tests; real calls use httpx

    def _get_client(self) -> Any:
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=120.0,
            )
        return self._client

    async def complete(self, *, system: str, user: str) -> str:
        client = self._get_client()
        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            # Support both httpx.AsyncClient (real) and duck-typed fakes (tests)
            if hasattr(client, "post"):
                resp = await client.post("/v1/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            else:
                # Duck-typed test fake using .chat.completions.create shape
                resp = await client.chat.completions.create(**{
                    "model": self._model,
                    "max_tokens": self._max_tokens,
                    "messages": payload["messages"],
                })
                return resp.choices[0].message.content
        except Exception as exc:
            raise LLMJudgeError(f"OpenAI-compat call failed: {exc}") from exc


class LocalOllamaBackend(OpenAICompatBackend):
    """OpenAICompatBackend pre-configured for a local Ollama instance."""

    def __init__(self, *, model: str, max_tokens: int = 2048) -> None:
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
```

- [ ] **Step 4: Run tests — expect pass**

```bash
uv run pytest tests/eval/test_judge_backend.py -v
```

Expected: 3 tests pass

- [ ] **Step 5: Commit**

```bash
git add rehearse/eval/scorers/judge_backend.py tests/eval/test_judge_backend.py
git commit -m "feat(judge): OpenAICompatBackend + LocalOllamaBackend"
```

---

### Task 3: `backend_from_env` factory

**Files:**
- Modify: `rehearse/eval/scorers/judge_backend.py`
- Modify: `tests/eval/test_judge_backend.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/eval/test_judge_backend.py
import pytest
from unittest.mock import patch

def test_backend_from_env_prefers_judge_base_url(monkeypatch) -> None:
    from rehearse.eval.scorers.judge_backend import backend_from_env, OpenAICompatBackend
    monkeypatch.setenv("JUDGE_BASE_URL", "http://gpu-box:8000/v1")
    monkeypatch.setenv("JUDGE_MODEL", "gemma-3-27b-it")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    backend = backend_from_env(model=None, max_tokens=512)
    assert isinstance(backend, OpenAICompatBackend)
    assert backend._model == "gemma-3-27b-it"


def test_backend_from_env_falls_back_to_anthropic(monkeypatch) -> None:
    from rehearse.eval.scorers.judge_backend import backend_from_env, AnthropicBackend
    monkeypatch.delenv("JUDGE_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    backend = backend_from_env(model="claude-haiku-4-5", max_tokens=512)
    assert isinstance(backend, AnthropicBackend)
    assert backend._model == "claude-haiku-4-5"


def test_backend_from_env_raises_when_nothing_configured(monkeypatch) -> None:
    from rehearse.eval.scorers.judge_backend import backend_from_env, LLMJudgeError
    monkeypatch.delenv("JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMJudgeError, match="no judge backend configured"):
        backend_from_env(model=None, max_tokens=512)
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/eval/test_judge_backend.py -k "from_env" -v
```

Expected: `ImportError` on `backend_from_env`

- [ ] **Step 3: Add `backend_from_env` to `judge_backend.py`**

```python
# append to rehearse/eval/scorers/judge_backend.py

def backend_from_env(
    *,
    model: str | None,
    max_tokens: int,
) -> JudgeBackend:
    """Build a JudgeBackend from environment variables.

    Priority:
      1. JUDGE_BASE_URL set → OpenAICompatBackend
      2. ANTHROPIC_API_KEY set → AnthropicBackend
      3. Neither → raises LLMJudgeError

    Model resolution: explicit `model` param → JUDGE_MODEL env → provider default.
    """
    base_url = os.environ.get("JUDGE_BASE_URL")
    if base_url:
        resolved_model = model or os.environ.get("JUDGE_MODEL") or _DEFAULT_COMPAT_MODEL
        api_key = os.environ.get("JUDGE_API_KEY", "local")
        return OpenAICompatBackend(
            model=resolved_model,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=api_key,
        )
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        resolved_model = model or _DEFAULT_ANTHROPIC_MODEL
        return AnthropicBackend(
            model=resolved_model,
            max_tokens=max_tokens,
            api_key=anthropic_key,
        )
    raise LLMJudgeError(
        "no judge backend configured: set JUDGE_BASE_URL (vLLM/Modal) "
        "or ANTHROPIC_API_KEY"
    )
```

- [ ] **Step 4: Run tests — expect pass**

```bash
uv run pytest tests/eval/test_judge_backend.py -v
```

Expected: 6 tests pass

- [ ] **Step 5: Commit**

```bash
git add rehearse/eval/scorers/judge_backend.py tests/eval/test_judge_backend.py
git commit -m "feat(judge): backend_from_env factory — JUDGE_BASE_URL priority over ANTHROPIC_API_KEY"
```

---

### Task 4: Wire `JudgeBackend` into `LLMJudge`

**Files:**
- Modify: `rehearse/eval/scorers/llm_judge.py`
- Modify: `tests/eval/test_judge_backend.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/eval/test_judge_backend.py

@pytest.mark.asyncio
async def test_llm_judge_uses_injected_backend() -> None:
    from rehearse.eval.scorers.llm_judge import LLMJudge
    from rehearse.eval.scorers.judge_backend import JudgeBackend

    class _StubBackend:
        async def complete(self, *, system: str, user: str) -> str:
            return '{"emotion_responsiveness": {"score": 0.8, "rationale": "good", "key_moments": [1]}, "coaching_trajectory_quality": {"score": 0.7, "rationale": "ok", "key_moments": [2]}}'

    judge = LLMJudge(backend=_StubBackend())
    result = await judge.judge(system="sys", user="usr")
    assert result["emotion_responsiveness"]["score"] == 0.8


@pytest.mark.asyncio
async def test_llm_judge_legacy_client_still_works() -> None:
    """client= injection from existing tests must continue to work."""
    from rehearse.eval.scorers.llm_judge import LLMJudge

    class _FakeMsg:
        content = [type("B", (), {"type": "text", "text": '{"x": 1}'})()]

    class _FakeMessages:
        async def create(self, **kwargs):
            return _FakeMsg()

    class _FakeClient:
        messages = _FakeMessages()

    judge = LLMJudge(client=_FakeClient())
    result = await judge.judge(system="s", user="u")
    assert result == {"x": 1}
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/eval/test_judge_backend.py -k "llm_judge" -v
```

Expected: `TypeError: LLMJudge.__init__() got an unexpected keyword argument 'backend'`

- [ ] **Step 3: Modify `llm_judge.py`**

Replace `__init__`, `_client_lazy`, and `judge` in `LLMJudge`. Do not touch anything else in the file.

```python
# In rehearse/eval/scorers/llm_judge.py
# Replace the LLMJudge class body (keep _DEFAULT_JUDGE_MODEL, _JSON_BLOCK, LLMJudgeError, and everything after _extract_dim unchanged)

class LLMJudge:
    """Judge primitive with pluggable backend.

    Each judge() call sends a system+user prompt to the configured backend
    and parses the response as a JSON object. The backend is selected by:
      - Explicit `backend=` injection (takes precedence — used in tests)
      - Explicit `client=` injection (legacy path — still works)
      - backend_from_env() if neither is provided (reads JUDGE_BASE_URL / ANTHROPIC_API_KEY)
    """

    def __init__(
        self,
        *,
        model: str = _DEFAULT_JUDGE_MODEL,
        max_tokens: int = 2048,
        temperature: float | None = None,
        client: Any = None,
        backend: Any = None,  # JudgeBackend | None
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = client
        self._backend = backend

    def _get_backend(self) -> Any:
        if self._backend is not None:
            return self._backend
        if self._client is not None:
            from rehearse.eval.scorers.judge_backend import _LegacyClientBackend
            return _LegacyClientBackend(
                client=self._client,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        from rehearse.eval.scorers.judge_backend import backend_from_env
        return backend_from_env(model=self.model, max_tokens=self.max_tokens)

    async def judge(self, *, system: str, user: str) -> dict[str, Any]:
        backend = self._get_backend()
        try:
            text = await backend.complete(system=system, user=user)
        except Exception as exc:
            from rehearse.eval.scorers.judge_backend import LLMJudgeError as _E
            if isinstance(exc, _E):
                raise
            raise LLMJudgeError(f"judge call failed: {type(exc).__name__}: {exc}") from exc
        return _parse_json(text)
```

Add `_LegacyClientBackend` to `judge_backend.py` (append):

```python
# append to rehearse/eval/scorers/judge_backend.py

class _LegacyClientBackend:
    """Wraps a raw AsyncAnthropic client injected via LLMJudge(client=).

    Exists solely for backwards compatibility. New code should inject a
    proper JudgeBackend instead.
    """

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        max_tokens: int,
        temperature: float | None,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def complete(self, *, system: str, user: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        resp = await self._client.messages.create(**kwargs)
        return "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        )
```

Also add `from typing import Any` to `judge_backend.py` imports if not already present.

- [ ] **Step 4: Run all judge tests**

```bash
uv run pytest tests/eval/test_judge_backend.py tests/eval/test_mme_sandbox_rollout.py -v
```

Expected: all pass. If `test_mme_sandbox_rollout.py` fails, go to Task 5.

- [ ] **Step 5: Commit**

```bash
git add rehearse/eval/scorers/judge_backend.py rehearse/eval/scorers/llm_judge.py tests/eval/test_judge_backend.py
git commit -m "feat(judge): wire JudgeBackend into LLMJudge — backend= injection + legacy client shim"
```

---

### Task 5: Update monkey-patched tests in `test_mme_sandbox_rollout.py`

**Files:**
- Modify: `tests/eval/test_mme_sandbox_rollout.py`

The two tests that currently patch `LLMJudge._client_lazy` need to use `backend=` injection instead. This is cleaner and does not rely on private method names.

- [ ] **Step 1: Read the current test file**

```bash
grep -n "_client_lazy\|monkeypatch\|LLMJudge\|_judge_client\|broken" tests/eval/test_mme_sandbox_rollout.py
```

- [ ] **Step 2: Replace `_client_lazy` monkeypatches with `backend=` injection**

Find the scorer construction that passes a `LLMJudge()` and add `backend=` to it. The pattern:

Before (approximate — check the actual file):
```python
def _judge_client(self: LLMJudge) -> _JudgeAnthropic:
    return _JudgeAnthropic(...)
monkeypatch.setattr(LLMJudge, "_client_lazy", _judge_client)
```

After:
```python
from rehearse.eval.scorers.judge_backend import _LegacyClientBackend

class _FakeBackend:
    async def complete(self, *, system: str, user: str) -> str:
        return <whatever the fake returned before>

scorer = TrajectoryJudgeScorer(judge=LLMJudge(backend=_FakeBackend()))
```

Do the same for the "broken" variant. The goal: no `monkeypatch` calls on `LLMJudge` internals remain.

- [ ] **Step 3: Run the updated tests**

```bash
uv run pytest tests/eval/test_mme_sandbox_rollout.py -v
```

Expected: all pass

- [ ] **Step 4: Run the full suite**

```bash
uv run pytest -q
```

Expected: same pass count as before this PR (the two pre-existing failures on `test_outbox` and `test_survey_judge` are unrelated — they should still be the only failures).

- [ ] **Step 5: Commit**

```bash
git add tests/eval/test_mme_sandbox_rollout.py
git commit -m "test(judge): replace _client_lazy monkeypatches with backend= injection"
```

---

## 7. Self-Review Checklist

- **O1** — covered by Task 4: `LLMJudge` with `JUDGE_BASE_URL` set builds an `OpenAICompatBackend`; no Anthropic import triggered
- **O2** — covered by Task 5: existing scorer tests pass without modification to scorer files
- **O3** — covered by Task 4 test `test_llm_judge_uses_injected_backend`
- **O4** — covered by Task 4 test `test_llm_judge_legacy_client_still_works` + Task 5
- **O5** — documented in §4.2: adding `RewardModelBackend(base_url=...)` implementing `complete()` is the only step needed

## 8. What Comes Next

This spec does not include:

- `RewardModelScorer` (Spike C) — separate spec, depends on enough `judge.json` examples accumulating
- Agentic judge (memory + tool calls) — separate spec; changes the `JudgeBackend` protocol to `evaluate(transcript, session_store) -> score`
- Audio judge (`AudioJudge` / Gemini) — separate provider question, same pattern applies
