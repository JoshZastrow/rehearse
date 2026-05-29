# Interactive Model-Agnostic Naming + Modal Inference Server

**Status:** `acknowledged`
**Policy:** `implementation`
**Applies to:** `rehearse/backends/`, `rehearse/config.py`, `.env`, Modal inference server

---

## Problem

Three things are blocked:

1. **`MOSHI_*` names are model-specific.** `MOSHI_HF_REPO` makes no sense when the underlying model might be PersonaPlex. Anyone reading the env file has to know which interactive model is in use to interpret the variable.

2. **Moshi can't run in production.** 14 GB weights + 20s/frame CPU inference = unusable locally. A hosted inference server on GPU is required. There is no spec or code for that path today.

3. **PersonaPlex can't be selected.** PersonaPlex (`nvidia/personaplex-7b-v1`) is a drop-in Moshi fine-tune with persona conditioning (voice prompt + text prompt). Its wire protocol differs slightly from stock Moshi: dual Mimi instances, `reset_streaming()` per-session, Opus transport. There is no path to select it today.

---

## Goals

- **G1.** Replace all `MOSHI_*` env vars and config fields with model-agnostic `INTERACTIVE_*` names.
- **G2.** Add `BACKEND_TYPE=interactive-modal` that proxies audio to a Modal-hosted WebSocket server rather than running inference in-process.
- **G3.** The Modal server supports both Moshi and PersonaPlex, selected by `INTERACTIVE_MODEL_TYPE`.
- **G4.** PersonaPlex persona (voice prompt + text prompt) integrates with the existing `swap_persona()` API and `PersonaSpec`.
- **G5.** No changes to `ManagedBackend`, `PipelineBackend`, or any Hume/pipeline code paths.

---

## Non-Goals

- vLLM is not part of this spec. Moshi uses its own streaming context managers; vLLM serves text, not full-interactive audio.
- This spec does not redesign the `ConversationBackend` protocol.
- PersonaPlex voice training (generating voice prompts) is out of scope.

---

## Design

### 1. Naming Rename

| Old env var | New env var | Description |
|---|---|---|
| `BACKEND_TYPE=moshi` | `BACKEND_TYPE=interactive` | Local in-process inference |
| _(new)_ | `BACKEND_TYPE=interactive-modal` | Remote Modal WebSocket inference |
| `MOSHI_CHECKPOINT_PATH` | `INTERACTIVE_CHECKPOINT_PATH` | Local weights dir; empty = HF Hub download |
| `MOSHI_HF_REPO` | `INTERACTIVE_MODEL_REPO` | HF repo to pull weights from |
| `MOSHI_DEVICE` | `INTERACTIVE_DEVICE` | `cuda` or `cpu` |
| `MOSHI_ASR_MODEL` | `INTERACTIVE_ASR_MODEL` | faster-whisper model size for user ASR |
| _(new)_ | `INTERACTIVE_MODEL_TYPE` | `moshi` (default) or `personaplex` |
| _(new)_ | `INTERACTIVE_MODAL_ENDPOINT` | WebSocket URL when `BACKEND_TYPE=interactive-modal` |

Config field rename (1-to-1 with env):

| Old field | New field |
|---|---|
| `moshi_checkpoint_path` | `interactive_checkpoint_path` |
| `moshi_hf_repo` | `interactive_model_repo` |
| `moshi_device` | `interactive_device` |
| `moshi_asr_model` | `interactive_asr_model` |
| _(new)_ | `interactive_model_type` |
| _(new)_ | `interactive_modal_endpoint` |

### 2. `BACKEND_TYPE` Values After This Spec

| Value | Class | Where inference runs |
|---|---|---|
| `pipeline` | `PipelineBackend` | Local (unchanged) |
| `managed` | `ManagedBackend` | Hume EVI cloud (unchanged) |
| `interactive` | `InteractiveBackend` | Local GPU/CPU (was `moshi`) |
| `interactive-modal` | `ModalInteractiveBackend` | Modal GPU cloud (new) |

`MoshiBackend` is renamed to `InteractiveBackend`. The class name `MoshiBackend` is not exposed in any public protocol; the rename is internal.

### 3. `InteractiveBackend` (renamed from `MoshiBackend`)

No behavior change. Only:
- Class renamed from `MoshiBackend` → `InteractiveBackend` in `rehearse/backends/interactive.py`
- Old `rehearse/backends/moshi.py` deleted (or kept as a 1-line re-export for test compatibility during transition)
- `load_models()` receives `model_type` parameter, routed to stock Moshi or PersonaPlex loader

### 4. `ModalInteractiveBackend` (`BACKEND_TYPE=interactive-modal`)

A thin async client that satisfies `ConversationBackend` by proxying audio over a WebSocket to a Modal server.

**Wire protocol (binary WebSocket):**

- Client → server: raw PCM16 16 kHz audio chunks (same bytes as `send_caller_audio`)
- Server → client: newline-delimited JSON control frames OR raw PCM16 16 kHz audio

Control frames emitted by server:

```json
{"type": "transcript", "speaker": "coach", "text": "...", "is_final": true}
{"type": "transcript", "speaker": "user", "text": "...", "is_final": true}
{"type": "prosody", "speaker": "user", "arousal": 0.0, "valence": 0.0}
{"type": "end_of_call", "reason": "hangup"}
```

Audio chunks have no framing header — they are raw bytes. The client distinguishes them by checking `isinstance(msg, bytes)`.

**`ModalInteractiveBackend` responsibilities:**

1. On `start()`: open WebSocket to `INTERACTIVE_MODAL_ENDPOINT`, send a JSON handshake with session metadata and persona fields.
2. On `send_caller_audio(pcm)`: forward raw bytes to server.
3. Background receive loop: deserialize JSON → emit `TranscriptDelta`, `ProsodyEvent`, `EndOfCall` onto `FrameBus`; binary → emit `AudioChunk(COACH)`.
4. On `inject_speech(text)`: send `{"type": "inject", "text": "..."}` to server.
5. On `swap_persona(spec)`: send `{"type": "swap_persona", "name": ..., "voice_prompt_b64": ..., "text_prompt": ...}`.
6. On `close()`: send close frame, await server acknowledgment or timeout 5s.

**Reconnect:** single reconnect attempt on WebSocket disconnect mid-call, then emit `EndOfCall(reason="error")` if second attempt fails.

### 5. Modal Inference Server

Implemented as a Modal app at `infra/modal/interactive_server.py`. Exposed as a Modal WebSocket endpoint.

**Startup (once per container):**
- Load model weights from HF Hub or Modal Volume (controlled by `INTERACTIVE_MODEL_REPO` / `INTERACTIVE_CHECKPOINT_PATH` secrets).
- Cache `(mimi, lm_gen, tokenizer)` in module-level `_state`. Same cache key pattern as `moshi_loader.py`.

**Per-session handling:**
- Each WebSocket connection = one session.
- An `asyncio.Lock` serializes concurrent sessions on the same container (Moshi streaming context is not re-entrant).
- On connect: receive handshake JSON → extract `session_id`, persona fields.
- If `model_type=personaplex`: call `lm_gen.step_system_prompts_async(mimi)` using `voice_prompt` + `text_prompt` from handshake to prime persona before first audio frame.
- If `model_type=moshi`: no priming step.
- Run inference loop: same algorithm as `InteractiveBackend._sync_inference_loop` but runs in `asyncio` with `await asyncio.to_thread(...)` for GPU calls.
- On `swap_persona` message: call `lm_gen.reset_streaming()` then re-prime with new prompts (PersonaPlex only; Moshi ignores).
- On session end (client closes or `inject` with EOS): flush, send `end_of_call`, close.

**Modal configuration:**

```python
# infra/modal/interactive_server.py
import modal

app = modal.App("rehearse-interactive")
image = modal.Image.debian_slim().pip_install(
    "torch", "torchaudio", "faster-whisper", "huggingface_hub"
).run_commands("pip install git+https://github.com/kyutai-labs/moshi.git")

model_volume = modal.Volume.from_name("rehearse-interactive-weights", create_if_missing=True)

@app.cls(
    gpu="A10G",
    container_idle_timeout=300,
    volumes={"/models": model_volume},
)
class InteractiveServer:
    @modal.enter()
    def load(self): ...

    @modal.web_endpoint(method="GET")
    async def ws(self, ...): ...
```

**Choosing GPU size:** A10G (24 GB VRAM) fits Moshi bf16 (~14 GB) with headroom. PersonaPlex at the same parameter count fits identically.

### 6. PersonaPlex Support

PersonaPlex differences from stock Moshi:

| Aspect | Moshi | PersonaPlex |
|---|---|---|
| HF repo | `kyutai/moshiko-pytorch-bf16` | `nvidia/personaplex-7b-v1` |
| Mimi instances | 1 | 2 (one for caller, one for persona encoding) |
| Persona priming | none | `lm_gen.step_system_prompts_async(other_mimi)` |
| Session reset | close + reopen streaming ctx | `lm_gen.reset_streaming()` + re-prime |
| Audio codec (PersonaPlex server) | PCM | Opus (we decode to PCM before sending to server) |

**`PersonaSpec` extensions** (additive, no breaking changes):

```python
class PersonaSpec(TypedDict, total=False):
    name: str
    text_prompt: str          # new: system-level persona description
    voice_prompt_b64: str     # new: base64-encoded WAV reference voice clip
```

`swap_persona()` in `InteractiveBackend` (local) and `ModalInteractiveBackend` (remote):
- Local: call `lm_gen.reset_streaming()`, then re-prime with new prompts before next frame.
- Remote: send `swap_persona` JSON message; server handles reset + re-prime.

**`INTERACTIVE_MODEL_TYPE=personaplex` default repo:** `nvidia/personaplex-7b-v1`.

**`INTERACTIVE_MODEL_TYPE=moshi` default repo:** `kyutai/moshiko-pytorch-bf16` (unchanged from current).

---

## File Map

| File | Action | Change |
|---|---|---|
| `rehearse/config.py` | Modify | Rename 4 fields; add `interactive_model_type`, `interactive_modal_endpoint` |
| `rehearse/backends/moshi.py` | Delete / rename | Move to `rehearse/backends/interactive.py`; rename class |
| `rehearse/backends/moshi_loader.py` | Modify | Add `model_type` param; route to PersonaPlex loader |
| `rehearse/backends/moshi_asr.py` | Rename | `rehearse/backends/interactive_asr.py`; no logic change |
| `rehearse/backends/interactive_modal.py` | Create | `ModalInteractiveBackend` |
| `rehearse/backends/factory.py` | Modify | `"interactive"` and `"interactive-modal"` cases; remove `"moshi"` |
| `rehearse/api/app.py` | Modify | Update `backend_type == "interactive"` check for preloading |
| `scripts/serve.sh` | Modify | Rename `BACKEND_TYPE=moshi` → `BACKEND_TYPE=interactive` |
| `infra/modal/interactive_server.py` | Create | Modal WebSocket inference server |
| `infra/modal/deploy.sh` | Create | `modal deploy infra/modal/interactive_server.py` |
| `.env` | Modify | Rename `MOSHI_*` → `INTERACTIVE_*`; add new vars |
| `tests/test_moshi_backend.py` | Rename + update | `tests/test_interactive_backend.py`; update imports |
| `tests/integration/test_moshi_e2e.py` | Rename + update | `tests/integration/test_interactive_e2e.py` |

---

## Acceptance Criteria

- [ ] All `MOSHI_*` env vars removed; `INTERACTIVE_*` equivalents work identically for the Moshi case.
- [ ] `BACKEND_TYPE=interactive` boots and passes existing unit + E2E tests (renamed from `moshi`).
- [ ] `BACKEND_TYPE=interactive-modal` instantiates `ModalInteractiveBackend`; sends audio to the configured WebSocket endpoint; publishes `AudioChunk`, `TranscriptDelta`, `EndOfCall` onto the bus.
- [ ] `INTERACTIVE_MODEL_TYPE=personaplex` loads PersonaPlex weights; `swap_persona()` triggers `reset_streaming()` + re-prime.
- [ ] Modal server deploys with `make deploy-interactive`; smoke test: send 5s of silence, receive `AudioChunk(COACH)` frames within 10s (GPU).
- [ ] `.env.example` updated; existing `.env` migration documented in a one-line comment block.

---

## Migration Notes

Operators updating `.env`:

```bash
# Rename these:
BACKEND_TYPE=moshi         →  BACKEND_TYPE=interactive
MOSHI_CHECKPOINT_PATH=...  →  INTERACTIVE_CHECKPOINT_PATH=...
MOSHI_HF_REPO=...          →  INTERACTIVE_MODEL_REPO=...
MOSHI_DEVICE=...           →  INTERACTIVE_DEVICE=...
MOSHI_ASR_MODEL=...        →  INTERACTIVE_ASR_MODEL=...

# Add these:
INTERACTIVE_MODEL_TYPE=moshi                        # or personaplex
INTERACTIVE_MODAL_ENDPOINT=                         # wss://... when using interactive-modal
```

Old `MOSHI_*` vars are silently ignored after this change. A startup warning should be emitted if any `MOSHI_*` var is detected in the environment (one-time deprecation log, not a hard error).
