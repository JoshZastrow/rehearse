# Interactive Caller Client Design

**Date:** 2026-06-01
**Status:** Approved

## Goal

Wire two Moshi model endpoints together so a caller model and a provider model can hold a full-duplex audio conversation autonomously. Enables synthetic training data generation and evaluation without a human caller. When a fine-tuned caller model is ready, swapping it in means changing a checkpoint path and redeploying — nothing else changes.

## Architecture

Two `ModalInteractiveBackend` instances run against two separate Modal endpoints. A `ConversationBridge` cross-wires them: it reads `AudioChunk` frames from each backend's FrameBus and calls `send_caller_audio()` on the other. Neither backend knows it is talking to a model rather than a human.

```
FrameBus(provider)                        FrameBus(caller)
  AudioChunk(COACH) ──→ ConversationBridge ──→ send_caller_audio()
  send_caller_audio() ←── ConversationBridge ←── AudioChunk(COACH)

Recording writers attach to FrameBus(provider) only.
```

Each endpoint is a separate Modal container with its own GPU allocation and model checkpoint. Both endpoints speak the same wire protocol (binary PCM16 + JSON events) so `ModalInteractiveBackend` works against either without modification.

---

## Current infrastructure problems

Five issues in `infra/interactive.py` must be fixed before two deployments are possible:

1. **Checkpoint hardcoded to HuggingFace** — `load_models(checkpoint_path="")` always downloads from HF. Fine-tuned models on the Modal Volume cannot be loaded.

2. **GPU count hardcoded** — `gpu="A10G"` is in the `@app.cls` decorator. Multi-GPU for larger fine-tuned models requires duplicating the class.

3. **Single class, single endpoint** — `InteractiveServer` cannot be deployed twice with different configs. Two roles need two `@app.cls` definitions.

4. **Speaker labels hardcoded to `"coach"`/`"user"`** — transcript events always emit `"speaker": "coach"`. The caller endpoint generates caller audio; its events must say `"caller"` so `ModalInteractiveBackend` maps them correctly.

5. **ASR wired to received audio** — faster-whisper runs on `caller_buf` (audio the server receives). For the caller endpoint, ASR should run on generated audio (what the caller model produces), not on what it hears.

---

## File structure

| File | Change |
|------|--------|
| `infra/interactive.py` | Refactor into `_InteractiveServerBase` + `ProviderServer` + `CallerServer` |
| `rehearse/backends/interactive/bridge.py` | New — `ConversationBridge` |
| `rehearse/eval/environments/interactive_sandbox.py` | New — two-backend session runner |
| `tests/integration/test_interactive_caller.py` | New — live Modal integration test |

---

## Section 1: Parameterized Modal server (`infra/interactive.py`)

Extract all inference logic into `_InteractiveServerBase`. Add three class-level attributes:

```python
class _InteractiveServerBase:
    checkpoint_path: str = ""
    """Empty string = load from HuggingFace. Non-empty = load from Modal Volume path."""

    gpu_count: int = 1
    """Number of GPUs per container. Set > 1 for large fine-tuned models."""

    speaker_role: Literal["provider", "caller"] = "provider"
    """Controls transcript event speaker labels and which audio buffer ASR runs on."""
```

`load()` uses `checkpoint_path` when non-empty:

```python
@modal.enter()
def load(self) -> None:
    repo = os.environ.get("INTERACTIVE_MODEL_REPO", "kyutai/moshiko-pytorch-bf16")
    self._mimi, self._lm_gen, self._tokenizer = load_models(
        checkpoint_path=self.checkpoint_path,  # non-empty = load fine-tuned weights
        hf_repo=repo,
        device="cuda",
    )
```

`_handle_session()` uses `self.speaker_role` in two places:

- Transcript events: `"speaker": self.speaker_role` (replaces hardcoded `"coach"`)
- ASR target: run on `provider_buf` (generated audio) when `speaker_role == "caller"`, on `caller_buf` (received audio) when `"provider"`

Two concrete subclasses in the same file:

```python
@app.cls(image=interactive_image, gpu="A10G", ...)
class ProviderServer(_InteractiveServerBase):
    checkpoint_path = ""
    gpu_count = 1
    speaker_role = "provider"

    @modal.web_server(_AIOHTTP_PORT)
    def serve(self): pass


@app.cls(image=interactive_image, gpu=modal.gpu.A10G(count=1), ...)
class CallerServer(_InteractiveServerBase):
    checkpoint_path = ""          # set to Volume path once fine-tuned model is ready
    gpu_count = 1                 # increase when fine-tuned model requires more VRAM
    speaker_role = "caller"

    @modal.web_server(_AIOHTTP_PORT)
    def serve(self): pass
```

Each class gets its own URL from Modal. Both are deployed with a single `modal deploy infra/interactive.py`.

Environment variables:
```
INTERACTIVE_PROVIDER_ENDPOINT=wss://...providerserver-serve.modal.run/ws
INTERACTIVE_CALLER_ENDPOINT=wss://...callerserver-serve.modal.run/ws
```

---

## Section 2: `ConversationBridge` (`rehearse/backends/interactive/bridge.py`)

Cross-wires two `(backend, bus)` pairs. Subscribes to `AudioChunk` on each bus and calls `send_caller_audio()` on the opposite backend.

```python
class ConversationBridge:
    def __init__(
        self,
        *,
        provider_backend: ConversationBackend,
        provider_bus: FrameBus,
        caller_backend: ConversationBackend,
        caller_bus: FrameBus,
    ) -> None: ...

    async def start(self) -> None:
        """Start two routing tasks. Returns immediately."""

    async def close(self) -> None:
        """Cancel routing tasks and drain queues."""
```

Two async tasks started in `start()`:

- **Provider → caller:** subscribes to `AudioChunk` on `provider_bus`, calls `caller_backend.send_caller_audio(chunk.pcm16_16k)`.
- **Caller → provider:** subscribes to `AudioChunk` on `caller_bus`, calls `provider_backend.send_caller_audio(chunk.pcm16_16k)`.

Both tasks stop on `EndOfCall` on either bus, or when `close()` is called.

No synchronization between the two tasks — Moshi is full-duplex; the natural backpressure of the WebSocket queues handles timing.

---

## Section 3: Two-backend session runner (`rehearse/eval/environments/interactive_sandbox.py`)

Wires everything together for a single synthetic session:

```python
async def run_interactive_session(
    *,
    session_id: str,
    provider_endpoint: str,
    caller_endpoint: str,
    max_duration_sec: float = 120.0,
    run_dir: Path | None = None,
) -> SessionResult:
```

Sequence:
1. Create `provider_bus` and `caller_bus` (two separate `FrameBus` instances).
2. Create `ModalInteractiveBackend(endpoint=provider_endpoint)` and `ModalInteractiveBackend(endpoint=caller_endpoint)`.
3. Attach recording writers (transcript, prosody, timing) to `provider_bus`.
4. `await provider_backend.start(session_id, provider_bus)`.
5. `await caller_backend.start(session_id + "-caller", caller_bus)`.
6. Create and `await bridge.start()`.
7. Seed the caller with 100ms of silence to trigger the first provider response.
8. Wait for `EndOfCall` on `provider_bus` or `max_duration_sec`, whichever comes first.
9. `await bridge.close()`, `await caller_backend.close()`, `await provider_backend.close()`.
10. Return `SessionResult` with transcript, duration, and artifact paths.

The seed silence at step 7 is necessary because Moshi generates in response to received audio — without any input, neither model starts speaking.

---

## Section 4: Live Modal integration test (`tests/integration/test_interactive_caller.py`)

Marked `@pytest.mark.live_modal`. Reads endpoints from environment.

```python
@pytest.mark.live_modal
async def test_bridge_audio_flows_both_ways():
    provider_url = os.environ["INTERACTIVE_PROVIDER_ENDPOINT"]
    caller_url = os.environ["INTERACTIVE_CALLER_ENDPOINT"]

    provider_bus = FrameBus()
    caller_bus = FrameBus()

    provider_chunks: list[AudioChunk] = []
    caller_chunks: list[AudioChunk] = []

    provider_bus.subscribe(AudioChunk, provider_chunks.append)
    caller_bus.subscribe(AudioChunk, caller_chunks.append)

    provider_backend = ModalInteractiveBackend(endpoint=provider_url)
    caller_backend = ModalInteractiveBackend(endpoint=caller_url)
    bridge = ConversationBridge(
        provider_backend=provider_backend, provider_bus=provider_bus,
        caller_backend=caller_backend,   caller_bus=caller_bus,
    )

    session_id = str(uuid.uuid4())
    await provider_backend.start(session_id, provider_bus)
    await caller_backend.start(session_id + "-caller", caller_bus)
    await bridge.start()

    # Seed caller to start the loop
    silence_100ms = b"\x00" * 3200
    await caller_backend.send_caller_audio(silence_100ms)

    await asyncio.sleep(10.0)

    await bridge.close()
    await caller_backend.close()
    await provider_backend.close()

    assert len(provider_chunks) > 0, "No audio generated by provider"
    assert len(caller_chunks) > 0, "No audio generated by caller"
```

A second test verifies transcript events appear on `provider_bus` within 30 seconds.

---

## Model swap path

When a fine-tuned caller model checkpoint is ready on the `rehearse-training` Modal Volume:

1. Set `CallerServer.checkpoint_path = "/mnt/training/runs/caller-v1/model.safetensors"` in `infra/interactive.py`.
2. Increase `CallerServer.gpu_count` if needed.
3. `modal deploy infra/interactive.py`.
4. `INTERACTIVE_CALLER_ENDPOINT` stays the same — Modal reuses the URL on redeploy.

The bridge, session runner, and test are unchanged.

---

## Out of scope

- Prosody scoring on caller-generated audio (no Hume integration for synthetic caller)
- Multi-turn session termination policy (max duration only for now)
- Concurrent synthetic sessions (one session per container pair; scale by deploying more containers)
