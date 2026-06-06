# Interactive Caller Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire two Moshi model endpoints together via `ConversationBridge` so provider and caller models can hold a full-duplex audio conversation autonomously, with a live Modal integration test confirming the full loop works.

**Architecture:** `ModalInteractiveBackend` is used for both roles unchanged. `ConversationBridge` cross-wires two `(backend, bus)` pairs by routing `AudioChunk` frames from each bus into `send_caller_audio()` on the opposite backend. `infra/interactive.py` is refactored into a parameterized base class with `ProviderServer` and `CallerServer` subclasses, fixing hardcoded speaker labels and ASR routing. A `run_interactive_session` runner orchestrates both backends for eval.

**Tech Stack:** Python asyncio, Modal (GPU inference), websockets, aiohttp, `FrameBus` (in-process pub/sub), `ModalInteractiveBackend` (WebSocket proxy), pytest with `live_modal` marker.

---

## Background: what exists and what changes

**`rehearse/backends/interactive/modal_backend.py`** — WebSocket client for one Modal endpoint. Maps wire labels to `Speaker` enum. Currently maps only `"coach"` → `Speaker.COACH` and `"user"` → `Speaker.USER`. The fixed server will emit `"provider"` and `"caller"` — these need to be added to the mapping.

**`infra/interactive.py`** — Modal GPU server. Has five problems (see spec). Needs a base class extracted, speaker labels parameterized, and ASR target fixed.

**`rehearse/backends/interactive/bridge.py`** — Does not exist. New file.

**`rehearse/eval/environments/interactive_sandbox.py`** — Does not exist. New file.

**`tests/integration/test_interactive_caller.py`** — Does not exist. New file.

---

## File structure

| File | Change |
|------|--------|
| `rehearse/backends/interactive/modal_backend.py` | Add `"provider"`/`"caller"` to speaker map |
| `infra/interactive.py` | Extract `_InteractiveServerBase`, add `ProviderServer` + `CallerServer` |
| `rehearse/backends/interactive/bridge.py` | Create — `ConversationBridge` |
| `rehearse/eval/environments/interactive_sandbox.py` | Create — `run_interactive_session` + `SessionResult` |
| `tests/integration/test_interactive_caller.py` | Create — live Modal integration tests |

---

### Task 1: Fix `ModalInteractiveBackend` speaker label mapping

**Files:**
- Modify: `rehearse/backends/interactive/modal_backend.py:_handle_event`
- Test: `tests/test_modal_backend.py`

The current mapping `"coach" → Speaker.COACH` is the only wire label handled. The refactored server will emit `"provider"` and `"caller"`. Both old and new labels must map correctly.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_modal_backend.py`:

```python
@pytest.mark.asyncio
async def test_modal_backend_maps_provider_label_to_coach():
    """'provider' wire label from the refactored server must map to Speaker.COACH."""
    import asyncio
    import json
    from unittest.mock import AsyncMock, patch, MagicMock

    backend = ModalInteractiveBackend(endpoint="ws://fake")
    bus = FrameBus(session_id="test-provider-label")
    frames: list = []

    async def collect():
        async for frame in bus.subscribe():
            frames.append(frame)

    collect_task = asyncio.create_task(collect())
    await asyncio.sleep(0)

    backend._session_id = "test-provider-label"
    backend._bus = bus

    await backend._handle_event({
        "type": "transcript",
        "utterance_id": "u1",
        "speaker": "provider",
        "text": "Hello",
        "is_final": True,
    })
    await bus.aclose()
    await collect_task

    from rehearse.frames import TranscriptDelta
    from rehearse.types import Speaker
    assert any(
        isinstance(f, TranscriptDelta) and f.speaker == Speaker.COACH
        for f in frames
    ), f"Expected Speaker.COACH for 'provider' label, got {frames}"


@pytest.mark.asyncio
async def test_modal_backend_maps_caller_label_to_user():
    """'caller' wire label from the refactored server must map to Speaker.USER."""
    import asyncio

    backend = ModalInteractiveBackend(endpoint="ws://fake")
    bus = FrameBus(session_id="test-caller-label")
    frames: list = []

    async def collect():
        async for frame in bus.subscribe():
            frames.append(frame)

    collect_task = asyncio.create_task(collect())
    await asyncio.sleep(0)

    backend._session_id = "test-caller-label"
    backend._bus = bus

    await backend._handle_event({
        "type": "transcript",
        "utterance_id": "u2",
        "speaker": "caller",
        "text": "Hi",
        "is_final": True,
    })
    await bus.aclose()
    await collect_task

    from rehearse.frames import TranscriptDelta
    from rehearse.types import Speaker
    assert any(
        isinstance(f, TranscriptDelta) and f.speaker == Speaker.USER
        for f in frames
    ), f"Expected Speaker.USER for 'caller' label, got {frames}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_modal_backend.py::test_modal_backend_maps_provider_label_to_coach tests/test_modal_backend.py::test_modal_backend_maps_caller_label_to_user -v
```

Expected: FAIL — `"provider"` maps to `Speaker.USER` (falls through to `else` branch).

- [ ] **Step 3: Add `_SPEAKER_MAP` and fix `_handle_event`**

In `rehearse/backends/interactive/modal_backend.py`, add a module-level constant after the imports:

```python
_SPEAKER_MAP: dict[str, Speaker] = {
    "coach": Speaker.COACH,
    "provider": Speaker.COACH,
    "user": Speaker.USER,
    "caller": Speaker.USER,
}
```

In `_handle_event`, replace both speaker mapping lines:

```python
# OLD (two occurrences — in "transcript" and "prosody" cases):
speaker = Speaker.COACH if data["speaker"] == "coach" else Speaker.USER

# NEW (same replacement for both):
speaker = _SPEAKER_MAP.get(data.get("speaker", ""), Speaker.COACH)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_modal_backend.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add rehearse/backends/interactive/modal_backend.py tests/test_modal_backend.py
git commit -m "fix: map 'provider'/'caller' wire labels in ModalInteractiveBackend"
```

---

### Task 2: Refactor `infra/interactive.py` — base class + two named servers

**Files:**
- Modify: `infra/interactive.py`

Extract the `InteractiveServer` logic into `_InteractiveServerBase` with three class-level attributes: `checkpoint_path`, `speaker_role`. Fix five hardcoded speaker labels and wire ASR to the correct audio buffer based on `speaker_role`. Define `ProviderServer` and `CallerServer` as thin subclasses.

No unit test is possible here (requires Modal GPU). The existing `smoke_test` local entrypoint validates the provider server after deploy. The live integration test in Task 5 validates the caller server.

- [ ] **Step 1: Extract the base class and add helper methods**

Replace the current `class InteractiveServer:` block in `infra/interactive.py`. Keep all imports and module-level constants (`interactive_image`, `hf_cache_vol`, `sessions_vol`, `app`, `_SESSIONS_MOUNT`, `MINUTES`, `_AIOHTTP_PORT`) unchanged.

Add the following base class (no `@app.cls` decorator — that goes on the subclasses):

```python
class _InteractiveServerBase:
    """Shared inference logic for provider and caller Modal servers."""

    checkpoint_path: str = ""
    """Empty = load from HuggingFace. Non-empty = load from Modal Volume path."""

    speaker_role: str = "provider"
    """'provider' or 'caller'. Controls transcript labels and ASR target."""

    @property
    def _other_role(self) -> str:
        return "caller" if self.speaker_role == "provider" else "provider"

    def _asr_push_received(self, asr, raw: bytes) -> None:
        """Run ASR on received audio only when we are the provider."""
        if self.speaker_role == "provider":
            asr.push_audio(raw)

    def _asr_push_generated(self, asr, pcm16_out: bytes) -> None:
        """Run ASR on generated audio only when we are the caller."""
        if self.speaker_role == "caller":
            asr.push_audio(pcm16_out)

    @modal.enter()
    def load(self) -> None:
        sys.path.insert(0, "/app")
        from rehearse.backends.interactive.loader import load_models  # type: ignore[import]

        repo = os.environ.get("INTERACTIVE_MODEL_REPO", "kyutai/moshiko-pytorch-bf16")
        self._mimi, self._lm_gen, self._tokenizer = load_models(
            checkpoint_path=self.checkpoint_path,
            hf_repo=repo,
            device="cuda",
        )
        self._mimi.streaming_forever(1)
        self._lm_gen.streaming_forever(1)
        print("[load] warming up CUDA kernels...", file=sys.stderr, flush=True)
        self._warmup()
        print("[load] warmup done, starting aiohttp server", file=sys.stderr, flush=True)
        t = threading.Thread(
            target=lambda: asyncio.run(self._run_aiohttp()),
            daemon=True,
        )
        t.start()

    def _warmup(self) -> None:
        import torch
        frame_size = self._mimi.frame_size
        with torch.no_grad():
            for _ in range(4):
                chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device="cuda")
                codes = self._mimi.encode(chunk)
                for c in range(codes.shape[-1]):
                    tokens = self._lm_gen.step(codes[:, :, c : c + 1])
                    if tokens is None:
                        continue
                    _ = self._mimi.decode(tokens[:, 1:])
        torch.cuda.synchronize()

    async def _run_aiohttp(self) -> None:
        from aiohttp import web
        self._session_lock = asyncio.Lock()

        async def health(request: web.Request) -> web.Response:
            return web.Response(status=200)

        async def ws_handler(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await self._handle_session(ws)
            return ws

        _app = web.Application()
        _app.router.add_get("/health", health)
        _app.router.add_get("/ws", ws_handler)
        runner = web.AppRunner(_app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", _AIOHTTP_PORT)
        await site.start()
        print(f"[aiohttp] listening on 0.0.0.0:{_AIOHTTP_PORT}", file=sys.stderr, flush=True)
        await asyncio.Event().wait()

    def _log(self, msg: str) -> None:
        import time as _time
        ts = _time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", file=sys.stderr, flush=True)
```

- [ ] **Step 2: Replace `_handle_session` with the speaker-aware version**

In `_handle_session`, make three targeted changes. The method signature and all surrounding logic (handshake, session_lock, buffers, token rows, volume persistence) stays identical. Only these lines change:

**Change 1** — replace `asr.push_audio(raw)` (the line that runs ASR on all received audio):
```python
# OLD:
asr.push_audio(raw)

# NEW:
self._asr_push_received(asr, raw)
```

**Change 2** — after `pcm16_out = (audio_16k * 32767).to(torch.int16).cpu().numpy().tobytes()`, add one line:
```python
pcm16_out = (audio_16k * 32767).to(torch.int16).cpu().numpy().tobytes()
self._asr_push_generated(asr, pcm16_out)  # ← add this line
```

**Change 3** — replace all five hardcoded speaker label strings in `send_str` calls:

| Old value | New value |
|-----------|-----------|
| `"speaker": "coach"` (4 occurrences) | `"speaker": self.speaker_role` |
| `"speaker": "user"` (2 occurrences) | `"speaker": self._other_role` |

- [ ] **Step 3: Add `ProviderServer` and `CallerServer` subclasses**

After `_InteractiveServerBase` and before `_write_mask`, add:

```python
@app.cls(
    image=interactive_image,
    gpu="A10G",
    scaledown_window=3 * MINUTES,
    timeout=30 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache_vol, _SESSIONS_MOUNT: sessions_vol},
)
class ProviderServer(_InteractiveServerBase):
    """Provider (coach) model endpoint. Loads from HuggingFace by default."""
    checkpoint_path = ""
    speaker_role = "provider"

    @modal.web_server(_AIOHTTP_PORT)
    def serve(self):
        pass  # server started in load()


@app.cls(
    image=interactive_image,
    gpu="A10G",
    scaledown_window=3 * MINUTES,
    timeout=30 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache_vol, _SESSIONS_MOUNT: sessions_vol},
)
class CallerServer(_InteractiveServerBase):
    """Caller model endpoint. Set checkpoint_path to a Volume path for fine-tuned weights."""
    checkpoint_path = ""      # e.g. "/mnt/training/runs/caller-v1"
    speaker_role = "caller"

    @modal.web_server(_AIOHTTP_PORT)
    def serve(self):
        pass  # server started in load()
```

Delete the old `class InteractiveServer` and its `@app.cls` decorator entirely.

- [ ] **Step 4: Update `smoke_test` to target `ProviderServer`**

In the `@app.local_entrypoint()` `smoke_test` function, replace `InteractiveServer()` with `ProviderServer()`:

```python
# OLD:
server = InteractiveServer()
url = await server.serve.get_url.aio()

# NEW:
server = ProviderServer()
url = await server.serve.get_url.aio()
```

- [ ] **Step 5: Commit**

```bash
git add infra/interactive.py
git commit -m "refactor: extract _InteractiveServerBase, add ProviderServer + CallerServer with parameterized speaker_role"
```

---

### Task 3: `ConversationBridge`

**Files:**
- Create: `rehearse/backends/interactive/bridge.py`
- Create: `tests/test_conversation_bridge.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_conversation_bridge.py`:

```python
"""Tests for ConversationBridge.

ConversationBridge cross-wires two (backend, bus) pairs:
  - AudioChunk from provider_bus → caller_backend.send_caller_audio()
  - AudioChunk from caller_bus   → provider_backend.send_caller_audio()
  - EndOfCall on either bus      → that routing task stops
"""
from __future__ import annotations

import asyncio

import pytest

from rehearse.bus import FrameBus
from rehearse.frames import AudioChunk, EndOfCall
from rehearse.types import Speaker


def _silence(n_bytes: int = 320) -> bytes:
    return b"\x00" * n_bytes


class _MockBackend:
    def __init__(self) -> None:
        self.received: list[bytes] = []

    async def send_caller_audio(self, pcm: bytes) -> None:
        self.received.append(pcm)


@pytest.mark.asyncio
async def test_bridge_routes_provider_audio_to_caller():
    """AudioChunk on provider_bus must reach caller_backend.send_caller_audio()."""
    from rehearse.backends.interactive.bridge import ConversationBridge

    provider_bus = FrameBus(session_id="p")
    caller_bus = FrameBus(session_id="c")
    provider_backend = _MockBackend()
    caller_backend = _MockBackend()

    bridge = ConversationBridge(
        provider_backend=provider_backend,
        provider_bus=provider_bus,
        caller_backend=caller_backend,
        caller_bus=caller_bus,
    )
    await bridge.start()

    pcm = _silence(640)
    await provider_bus.publish(
        AudioChunk(session_id="p", speaker=Speaker.COACH, pcm16_16k=pcm, ts=0.0)
    )
    await asyncio.sleep(0.01)

    assert caller_backend.received == [pcm]
    assert provider_backend.received == []

    await bridge.close()
    await provider_bus.aclose()
    await caller_bus.aclose()


@pytest.mark.asyncio
async def test_bridge_routes_caller_audio_to_provider():
    """AudioChunk on caller_bus must reach provider_backend.send_caller_audio()."""
    from rehearse.backends.interactive.bridge import ConversationBridge

    provider_bus = FrameBus(session_id="p")
    caller_bus = FrameBus(session_id="c")
    provider_backend = _MockBackend()
    caller_backend = _MockBackend()

    bridge = ConversationBridge(
        provider_backend=provider_backend,
        provider_bus=provider_bus,
        caller_backend=caller_backend,
        caller_bus=caller_bus,
    )
    await bridge.start()

    pcm = _silence(320)
    await caller_bus.publish(
        AudioChunk(session_id="c", speaker=Speaker.COACH, pcm16_16k=pcm, ts=0.0)
    )
    await asyncio.sleep(0.01)

    assert provider_backend.received == [pcm]
    assert caller_backend.received == []

    await bridge.close()
    await provider_bus.aclose()
    await caller_bus.aclose()


@pytest.mark.asyncio
async def test_bridge_stops_routing_on_end_of_call():
    """EndOfCall on provider_bus must stop the provider→caller routing task."""
    from rehearse.backends.interactive.bridge import ConversationBridge

    provider_bus = FrameBus(session_id="p")
    caller_bus = FrameBus(session_id="c")
    provider_backend = _MockBackend()
    caller_backend = _MockBackend()

    bridge = ConversationBridge(
        provider_backend=provider_backend,
        provider_bus=provider_bus,
        caller_backend=caller_backend,
        caller_bus=caller_bus,
    )
    await bridge.start()

    await provider_bus.publish(
        EndOfCall(session_id="p", reason="hangup", ts=0.0)
    )
    await asyncio.sleep(0.01)

    # Publish audio after EndOfCall — must NOT reach caller
    await provider_bus.publish(
        AudioChunk(session_id="p", speaker=Speaker.COACH, pcm16_16k=_silence(), ts=1.0)
    )
    await asyncio.sleep(0.01)

    assert caller_backend.received == []

    await bridge.close()
    await provider_bus.aclose()
    await caller_bus.aclose()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_conversation_bridge.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'rehearse.backends.interactive.bridge'`

- [ ] **Step 3: Implement `ConversationBridge`**

Create `rehearse/backends/interactive/bridge.py`:

```python
"""ConversationBridge — cross-wires two (backend, bus) pairs for synthetic sessions.

Routes AudioChunk frames from each bus into send_caller_audio() on the opposite
backend, creating a full-duplex audio loop between two Moshi endpoints.
"""
from __future__ import annotations

import asyncio

from rehearse.bus import FrameBus
from rehearse.frames import AudioChunk, EndOfCall


class ConversationBridge:
    """Cross-wire two ConversationBackend instances via their FrameBuses.

    provider_bus → caller_backend.send_caller_audio()
    caller_bus   → provider_backend.send_caller_audio()
    """

    def __init__(
        self,
        *,
        provider_backend,
        provider_bus: FrameBus,
        caller_backend,
        caller_bus: FrameBus,
    ) -> None:
        self._provider_backend = provider_backend
        self._provider_bus = provider_bus
        self._caller_backend = caller_backend
        self._caller_bus = caller_bus
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Start routing tasks. Returns immediately; routing runs in background."""
        self._tasks = [
            asyncio.create_task(
                self._route(self._provider_bus, self._caller_backend),
                name="bridge-provider-to-caller",
            ),
            asyncio.create_task(
                self._route(self._caller_bus, self._provider_backend),
                name="bridge-caller-to-provider",
            ),
        ]

    async def _route(self, source_bus: FrameBus, target_backend) -> None:
        async for frame in source_bus.subscribe():
            if isinstance(frame, AudioChunk):
                await target_backend.send_caller_audio(frame.pcm16_16k)
            elif isinstance(frame, EndOfCall):
                return

    async def close(self) -> None:
        """Cancel routing tasks and wait for them to finish."""
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_conversation_bridge.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add rehearse/backends/interactive/bridge.py tests/test_conversation_bridge.py
git commit -m "feat: add ConversationBridge to cross-wire two Moshi endpoints"
```

---

### Task 4: Two-backend session runner

**Files:**
- Create: `rehearse/eval/environments/interactive_sandbox.py`
- Create: `tests/eval/test_interactive_sandbox.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/eval/test_interactive_sandbox.py`:

```python
"""Tests for run_interactive_session.

Uses mock backends — no Modal or GPU required.
"""
from __future__ import annotations

import asyncio

import pytest

from rehearse.bus import FrameBus
from rehearse.frames import AudioChunk, EndOfCall
from rehearse.types import Speaker


class _FakeBackend:
    """Backend that publishes one AudioChunk then EndOfCall when send_caller_audio is called."""

    def __init__(self, bus: FrameBus, session_id: str) -> None:
        self._bus = bus
        self._session_id = session_id
        self._started = False

    async def start(self, session_id: str, bus: FrameBus) -> None:
        self._bus = bus
        self._session_id = session_id
        self._started = True

    async def send_caller_audio(self, pcm: bytes) -> None:
        await self._bus.publish(
            AudioChunk(session_id=self._session_id, speaker=Speaker.COACH, pcm16_16k=pcm, ts=0.0)
        )
        await self._bus.publish(
            EndOfCall(session_id=self._session_id, reason="hangup", ts=0.0)
        )

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_run_interactive_session_returns_session_result(monkeypatch):
    """run_interactive_session must return a SessionResult when EndOfCall is received."""
    from rehearse.eval.environments.interactive_sandbox import (
        SessionResult,
        run_interactive_session,
    )
    from rehearse.backends.interactive import modal_backend as mb

    provider_bus_holder: list[FrameBus] = []
    caller_bus_holder: list[FrameBus] = []

    def _make_backend(endpoint: str):
        if "provider" in endpoint:
            b = _FakeBackend(None, "")  # type: ignore[arg-type]
            provider_bus_holder.append(b)
            return b
        else:
            b = _FakeBackend(None, "")  # type: ignore[arg-type]
            caller_bus_holder.append(b)
            return b

    monkeypatch.setattr(
        "rehearse.eval.environments.interactive_sandbox.ModalInteractiveBackend",
        _make_backend,
    )

    result = await run_interactive_session(
        session_id="test-session",
        provider_endpoint="ws://provider",
        caller_endpoint="ws://caller",
        max_duration_sec=5.0,
    )

    assert isinstance(result, SessionResult)
    assert result.session_id == "test-session"
    assert result.end_reason == "hangup"
    assert result.duration_sec >= 0.0


@pytest.mark.asyncio
async def test_run_interactive_session_times_out(monkeypatch):
    """run_interactive_session must return after max_duration_sec if EndOfCall never arrives."""
    from rehearse.eval.environments.interactive_sandbox import (
        SessionResult,
        run_interactive_session,
    )

    class _SilentBackend:
        async def start(self, session_id: str, bus: FrameBus) -> None:
            pass
        async def send_caller_audio(self, pcm: bytes) -> None:
            pass
        async def close(self) -> None:
            pass

    monkeypatch.setattr(
        "rehearse.eval.environments.interactive_sandbox.ModalInteractiveBackend",
        lambda endpoint: _SilentBackend(),
    )

    result = await run_interactive_session(
        session_id="timeout-session",
        provider_endpoint="ws://provider",
        caller_endpoint="ws://caller",
        max_duration_sec=0.1,
    )

    assert result.end_reason == "timeout"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/eval/test_interactive_sandbox.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'rehearse.eval.environments.interactive_sandbox'`

- [ ] **Step 3: Implement `run_interactive_session`**

Create `rehearse/eval/environments/interactive_sandbox.py`:

```python
"""Two-backend session runner for synthetic caller/provider conversations.

Wires two ModalInteractiveBackend instances together via ConversationBridge
and runs until EndOfCall or max_duration_sec, whichever comes first.
"""
from __future__ import annotations

import asyncio
import dataclasses
import time
from pathlib import Path

from rehearse.backends.interactive.bridge import ConversationBridge
from rehearse.backends.interactive.modal_backend import ModalInteractiveBackend
from rehearse.bus import FrameBus
from rehearse.frames import EndOfCall


@dataclasses.dataclass
class SessionResult:
    session_id: str
    duration_sec: float
    end_reason: str
    run_dir: Path | None = None


async def run_interactive_session(
    *,
    session_id: str,
    provider_endpoint: str,
    caller_endpoint: str,
    max_duration_sec: float = 120.0,
    run_dir: Path | None = None,
) -> SessionResult:
    """Run one synthetic session between a provider and caller Moshi endpoint.

    Seeds the caller with 100ms of silence to start the loop, then waits for
    EndOfCall on the provider bus or max_duration_sec timeout.
    """
    provider_bus = FrameBus(session_id=session_id)
    caller_bus = FrameBus(session_id=session_id + "-caller")

    provider_backend = ModalInteractiveBackend(endpoint=provider_endpoint)
    caller_backend = ModalInteractiveBackend(endpoint=caller_endpoint)

    bridge = ConversationBridge(
        provider_backend=provider_backend,
        provider_bus=provider_bus,
        caller_backend=caller_backend,
        caller_bus=caller_bus,
    )

    t_start = time.monotonic()
    end_reason = "timeout"

    await provider_backend.start(session_id, provider_bus)
    await caller_backend.start(session_id + "-caller", caller_bus)
    await bridge.start()

    # Seed caller with silence so Moshi starts generating audio
    silence_100ms = b"\x00" * 3200  # 100ms at 16kHz PCM16 (3200 bytes)
    await caller_backend.send_caller_audio(silence_100ms)

    try:
        async with asyncio.timeout(max_duration_sec):
            async for frame in provider_bus.subscribe():
                if isinstance(frame, EndOfCall):
                    end_reason = frame.reason
                    break
    except asyncio.TimeoutError:
        pass
    finally:
        await bridge.close()
        await caller_backend.close()
        await provider_backend.close()
        await provider_bus.aclose()
        await caller_bus.aclose()

    return SessionResult(
        session_id=session_id,
        duration_sec=time.monotonic() - t_start,
        end_reason=end_reason,
        run_dir=run_dir,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/eval/test_interactive_sandbox.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Run the full test suite to check for regressions**

```bash
uv run pytest tests/ -v --ignore=tests/integration -q
```

Expected: all existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add rehearse/eval/environments/interactive_sandbox.py tests/eval/test_interactive_sandbox.py
git commit -m "feat: add run_interactive_session and SessionResult for two-backend eval"
```

---

### Task 5: Live Modal integration test

**Files:**
- Create: `tests/integration/test_interactive_caller.py`

This task requires both `ProviderServer` and `CallerServer` deployed to Modal. The test reads endpoints from environment variables. It is marked `live_modal` and excluded from the default pytest run.

Before running: deploy with `modal deploy infra/interactive.py`, then set:
```
INTERACTIVE_PROVIDER_ENDPOINT=wss://<workspace>--rehearse-interactive-providerserver-serve.modal.run/ws
INTERACTIVE_CALLER_ENDPOINT=wss://<workspace>--rehearse-interactive-callerserver-serve.modal.run/ws
```

- [ ] **Step 1: Create the integration test file**

Create `tests/integration/test_interactive_caller.py`:

```python
"""Live Modal integration test: ConversationBridge with two real Moshi endpoints.

Requires both ProviderServer and CallerServer deployed and reachable.
Run with:
    INTERACTIVE_PROVIDER_ENDPOINT=wss://... \
    INTERACTIVE_CALLER_ENDPOINT=wss://... \
    pytest tests/integration/test_interactive_caller.py -m live_modal -v
"""
from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from rehearse.backends.interactive.bridge import ConversationBridge
from rehearse.backends.interactive.modal_backend import ModalInteractiveBackend
from rehearse.bus import FrameBus
from rehearse.frames import AudioChunk, EndOfCall, TranscriptDelta


def _endpoints() -> tuple[str, str]:
    provider = os.environ.get("INTERACTIVE_PROVIDER_ENDPOINT", "")
    caller = os.environ.get("INTERACTIVE_CALLER_ENDPOINT", "")
    if not provider or not caller:
        pytest.skip(
            "INTERACTIVE_PROVIDER_ENDPOINT and INTERACTIVE_CALLER_ENDPOINT must be set"
        )
    return provider, caller


@pytest.mark.live_modal
@pytest.mark.asyncio
async def test_bridge_audio_flows_both_ways():
    """Both endpoints must generate audio within 10 seconds of seeding."""
    provider_url, caller_url = _endpoints()

    session_id = str(uuid.uuid4())
    provider_bus = FrameBus(session_id=session_id)
    caller_bus = FrameBus(session_id=session_id + "-caller")

    provider_chunks: list[AudioChunk] = []
    caller_chunks: list[AudioChunk] = []

    async def _collect_provider() -> None:
        async for frame in provider_bus.subscribe():
            if isinstance(frame, AudioChunk):
                provider_chunks.append(frame)
            elif isinstance(frame, EndOfCall):
                return

    async def _collect_caller() -> None:
        async for frame in caller_bus.subscribe():
            if isinstance(frame, AudioChunk):
                caller_chunks.append(frame)
            elif isinstance(frame, EndOfCall):
                return

    provider_backend = ModalInteractiveBackend(endpoint=provider_url)
    caller_backend = ModalInteractiveBackend(endpoint=caller_url)
    bridge = ConversationBridge(
        provider_backend=provider_backend,
        provider_bus=provider_bus,
        caller_backend=caller_backend,
        caller_bus=caller_bus,
    )

    collect_p = asyncio.create_task(_collect_provider())
    collect_c = asyncio.create_task(_collect_caller())

    await provider_backend.start(session_id, provider_bus)
    await caller_backend.start(session_id + "-caller", caller_bus)
    await bridge.start()

    # Seed caller to start the loop
    await caller_backend.send_caller_audio(b"\x00" * 3200)

    await asyncio.sleep(10.0)

    await bridge.close()
    await caller_backend.close()
    await provider_backend.close()
    await provider_bus.aclose()
    await caller_bus.aclose()
    collect_p.cancel()
    collect_c.cancel()
    await asyncio.gather(collect_p, collect_c, return_exceptions=True)

    assert len(provider_chunks) > 0, "Provider endpoint generated no audio in 10s"
    assert len(caller_chunks) > 0, "Caller endpoint generated no audio in 10s"


@pytest.mark.live_modal
@pytest.mark.asyncio
async def test_bridge_transcript_appears_within_30s():
    """A final TranscriptDelta must appear on the provider bus within 30 seconds."""
    provider_url, caller_url = _endpoints()

    session_id = str(uuid.uuid4())
    provider_bus = FrameBus(session_id=session_id)
    caller_bus = FrameBus(session_id=session_id + "-caller")

    transcripts: list[TranscriptDelta] = []

    async def _collect() -> None:
        async for frame in provider_bus.subscribe():
            if isinstance(frame, TranscriptDelta) and frame.is_final:
                transcripts.append(frame)
                return
            elif isinstance(frame, EndOfCall):
                return

    provider_backend = ModalInteractiveBackend(endpoint=provider_url)
    caller_backend = ModalInteractiveBackend(endpoint=caller_url)
    bridge = ConversationBridge(
        provider_backend=provider_backend,
        provider_bus=provider_bus,
        caller_backend=caller_backend,
        caller_bus=caller_bus,
    )

    collect_task = asyncio.create_task(_collect())

    await provider_backend.start(session_id, provider_bus)
    await caller_backend.start(session_id + "-caller", caller_bus)
    await bridge.start()

    await caller_backend.send_caller_audio(b"\x00" * 3200)

    try:
        await asyncio.wait_for(collect_task, timeout=30.0)
    except asyncio.TimeoutError:
        collect_task.cancel()

    await bridge.close()
    await caller_backend.close()
    await provider_backend.close()
    await provider_bus.aclose()
    await caller_bus.aclose()

    assert len(transcripts) >= 1, "No final transcript from provider within 30s"
```

- [ ] **Step 2: Verify the test is excluded from the default run**

```bash
uv run pytest tests/ -q --ignore=tests/integration 2>&1 | tail -5
```

Expected: all pass, no `test_interactive_caller` tests appear.

- [ ] **Step 3: Deploy to Modal**

```bash
modal deploy infra/interactive.py
```

Expected output includes two URLs — one for `ProviderServer` and one for `CallerServer`. Copy both.

- [ ] **Step 4: Set environment variables and run the live test**

```bash
export INTERACTIVE_PROVIDER_ENDPOINT=wss://<workspace>--rehearse-interactive-providerserver-serve.modal.run/ws
export INTERACTIVE_CALLER_ENDPOINT=wss://<workspace>--rehearse-interactive-callerserver-serve.modal.run/ws

uv run pytest tests/integration/test_interactive_caller.py -m live_modal -v
```

Expected:
- Both tests PASS
- `test_bridge_audio_flows_both_ways`: `provider_chunks > 0` and `caller_chunks > 0`
- `test_bridge_transcript_appears_within_30s`: at least one final `TranscriptDelta`

Note: first run may take up to 90 seconds for Modal GPU cold start. The `ModalInteractiveBackend` retries 18 × 5s to cover this.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_interactive_caller.py
git commit -m "test: add live Modal integration test for ConversationBridge"
```

---

## Model swap path

Once a fine-tuned caller model checkpoint is ready on the `rehearse-training` Modal Volume:

1. In `infra/interactive.py`, set `CallerServer.checkpoint_path = "/mnt/training/runs/caller-v1"` (the path on the Volume where the checkpoint lives).
2. If the model requires more VRAM, change `gpu="A10G"` to `gpu=modal.gpu.A10G(count=2)` in `CallerServer`'s `@app.cls` decorator.
3. `modal deploy infra/interactive.py` — Modal reuses the same URL on redeploy.
4. Re-run `tests/integration/test_interactive_caller.py -m live_modal` to confirm the fine-tuned model responds.

No other code changes required.
