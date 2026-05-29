# Backend-Aware Startup Handlers

**Date:** 2026-05-29
**Status:** Approved

## Problem

`make serve` runs every startup step unconditionally regardless of `BACKEND_TYPE`. Two concrete failures today:

1. **Hume EVI config sync** runs for all backends — `interactive` never uses Hume EVI, so this is wasted work and noise.
2. **Anthropic credential validation** fires whenever `ANTHROPIC_API_KEY` is set — `interactive` uses Moshi end-to-end and never calls the CLM webhook, so an out-of-credits key crashes startup unnecessarily.
3. **Interactive model loading** runs even when `INTERACTIVE_MODAL_ENDPOINT` is set — the Modal backend connects to a remote server and has no need for local model files, causing a CUDA error on non-GPU machines.

## Design

### Shell layer — `scripts/serve.sh`

The Hume EVI sync step is wrapped in a `managed`-only guard. Default (`BACKEND_TYPE` unset) preserves existing behavior since `managed` is the default.

```bash
if [ "$BACKEND_TYPE" = "managed" ] || [ -z "$BACKEND_TYPE" ]; then
  echo "Hume: syncing local persona configs to provider (BACKEND_TYPE=$BACKEND_TYPE)..."
  uv run rehearse-hume sync 2>&1 | tail -2
else
  echo "Hume: skipping persona config sync — BACKEND_TYPE=$BACKEND_TYPE does not use Hume EVI"
fi
```

All other shell steps (Honcho, LiteLLM, ngrok) are unchanged — LiteLLM is already gated on backend type.

### Python layer — `rehearse/api/app.py`

The ad-hoc lifespan body is replaced with a named handler list. Each handler declares which backends it applies to. The loop logs `startup.handler.running` or `startup.handler.skipped` for every handler, giving a complete picture of what ran.

**Handler registry:**

| Handler | Applies to | Reason |
|---|---|---|
| `anthropic_credentials` | `managed`, `pipeline` | Only these backends use the CLM webhook that calls Anthropic |
| `interactive_models` | `interactive`, `moshi` | Only these backends run Moshi locally; skipped when `INTERACTIVE_MODAL_ENDPOINT` is set |

**`_CLM_BACKENDS = {"managed", "pipeline"}`** — `interactive` is excluded because Moshi handles the full conversation loop (speech in → speech out) without going through the CLM webhook.

**`_load_interactive_models`** checks `config.interactive_modal_endpoint` first: if set, it logs a skip reason (`using_modal_endpoint`) and returns without touching local model files.

### Log output shape

With `BACKEND_TYPE=interactive`:
```
Hume: skipping persona config sync — BACKEND_TYPE=interactive does not use Hume EVI
{"event": "startup.handler.skipped", "handler": "anthropic_credentials", "backend": "interactive"}
{"event": "startup.handler.running", "handler": "interactive_models", "backend": "interactive"}
```

With `BACKEND_TYPE=managed`:
```
Hume: syncing local persona configs to provider (BACKEND_TYPE=managed)...
{"event": "startup.handler.running", "handler": "anthropic_credentials", "backend": "managed"}
{"event": "startup.handler.skipped", "handler": "interactive_models", "backend": "managed"}
```

## What does not change

- Honcho startup (all backends need memory)
- LiteLLM proxy gating (already correct in serve.sh)
- ngrok (all backends need the tunnel)
- `build_clm_responder` and `mount_clm_routes` (CLM routes remain mounted for all backends; the endpoint simply won't be called for `interactive`)
- `FinalizeSweeper` startup (unchanged)

## Adding future handlers

Add one entry to `_STARTUP_HANDLERS` in `app.py`:

```python
("my_new_step", {"managed"}, _my_new_handler),
```

No control flow changes needed — logging is automatic.
