# Inference Infrastructure

## Overview

All model calls in Rehearse route through a single LiteLLM proxy. One base URL,
models selected by alias. To swap a backend, edit `litellm_config.yaml` and
restart the proxy — no code changes.

```
Your code  ──►  http://localhost:4000  ──►  Modal/vLLM (Gemma 4)
                  (LiteLLM proxy)       ──►  Anthropic (Claude)
                                        ──►  Ollama (local dev)
```

## Quick start

**1. Add to `.env`:**

```bash
MODAL_GEMMA_API_KEY=<your-modal-token>   # modal token new
ANTHROPIC_API_KEY=<your-anthropic-key>
LITELLM_MASTER_KEY=sk-rehearse-local     # any secret; required by proxy
```

**2. Install LiteLLM (once):**

```bash
uv pip install litellm[proxy]
```

**3. Start the proxy:**

```bash
litellm --config infra/litellm_config.yaml
# Listening on http://0.0.0.0:4000
```

**4. Point your code at the proxy:**

```bash
# In .env — used by PipelineBackend, LLMJudge, and any other callers
LITELLM_BASE_URL=http://localhost:4000
LITELLM_API_KEY=sk-rehearse-local        # must match LITELLM_MASTER_KEY
```

---

## Making chat completion calls

The proxy speaks standard OpenAI `/chat/completions`. Any HTTP client works.

**curl:**

```bash
curl http://localhost:4000/chat/completions \
  -H "Authorization: Bearer sk-rehearse-local" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "judge",
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

**Python (httpx — what PipelineBackend uses):**

```python
import httpx, os

async def chat(model: str, messages: list[dict]) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{os.environ['LITELLM_BASE_URL']}/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['LITELLM_API_KEY']}"},
            json={"model": model, "messages": messages},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

# Use the model alias, not the full model name:
reply = await chat("coach", [{"role": "user", "content": "Hello"}])
reply = await chat("judge", [{"role": "user", "content": "Score this..."}])
```

**Python (openai SDK — if preferred):**

```python
from openai import AsyncOpenAI
import os

client = AsyncOpenAI(
    base_url=os.environ["LITELLM_BASE_URL"],
    api_key=os.environ["LITELLM_API_KEY"],
)

resp = await client.chat.completions.create(
    model="coach",
    messages=[{"role": "user", "content": "Hello"}],
)
```

---

## Model aliases

| Alias | Backend | Used by |
|---|---|---|
| `judge` | Gemma 4 on Modal (H200) | `LLMJudge`, eval scoring |
| `coach` | Anthropic Claude Sonnet | Hume EVI webhook, `PipelineBackend` |
| `local` | Ollama `gemma3:27b` | Local dev, no API keys needed |

To add a model: add an entry to `infra/litellm_config.yaml` and restart.

---

## Wiring PipelineBackend

`PipelineBackend` already has `self._clm_url` for chat completions. Point it at
the proxy:

```bash
# .env
PIPELINE_CLM_URL=http://localhost:4000/chat/completions
```

The HTTP call shape stays identical — it's the same OpenAI-compat endpoint.
The only difference is the model name in the request body should be `"coach"`
(or whichever alias you want) rather than a raw Anthropic model string.

---

## Deployed infrastructure

| Service | URL | Command |
|---|---|---|
| Gemma 4 vLLM | `https://joshzastrow--rehearse-gemma-judge-serve.modal.run` | `modal deploy infra/judge.py` |
| LiteLLM proxy | `http://localhost:4000` (local) | `litellm --config infra/litellm_config.yaml` |
