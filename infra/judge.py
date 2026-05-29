"""LLM inference server for Rehearse eval judges.

Serves an OpenAI-compatible /v1/chat/completions endpoint that accepts both
text and audio inputs. Used as the audio-judge alias in litellm_config.yaml.

Deploy:
    modal deploy infra/judge.py

The deployed URL is deterministic:
    https://<workspace>--rehearse-gemma-judge-serve.modal.run

Set in .env:
    VLLM_BASE_URL=https://<workspace>--rehearse-gemma-judge-serve.modal.run/v1
    VLLM_API_KEY=<modal-token>  # or any non-empty string if auth is disabled

Cost: ~$0 when idle (scaledown_window=30s). Spins up on first eval request.
"""

import json

import modal

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.19.0")
    .uv_pip_install("transformers==5.5.0")  # required for Gemma 4 as of vllm 0.19.0
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})   # faster model transfers from HuggingFace
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# Gemma 4 26B MoE with 4B active parameters. Supports text + audio + image + video.
# Pinned to a specific revision to avoid upstream changes breaking evals.
MODEL_NAME = "google/gemma-4-26B-A4B-it"
MODEL_REVISION = "47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301"

# Model weights cached in a Modal Volume so we don't re-download on every cold start.
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# Performance knob
# ---------------------------------------------------------------------------

# True  = faster cold start, lower throughput (good for dev iteration)
# False = slower cold start, higher throughput (good for CI/eval batches)
FAST_BOOT = True

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = modal.App("rehearse-gemma-judge")

MINUTES = 60
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu="H100:1",
    # Release GPU 30 seconds after the last request.
    # Enough buffer for back-to-back calls in a single eval run.
    # Cold start is ~25s from cache — cheaper to restart than idle.
    scaledown_window=30,
    # Allow up to 10 minutes for the container to start (model download + compile).
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--revision", MODEL_REVISION,
        "--served-model-name", MODEL_NAME,
        "llm",
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--uvicorn-log-level=info",
        "--async-scheduling",
        "--tensor-parallel-size", "1",
    ]

    # Compilation tradeoff: set FAST_BOOT=True to skip for faster iteration
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    # Audio input enabled (up to 1 audio clip per prompt).
    # This is the key difference from the standard vLLM tutorial, which disables
    # multimedia. Rehearse evals send one WAV clip per judge call.
    cmd += [
        "--limit-mm-per-prompt",
        f"'{json.dumps({'image': 0, 'video': 0, 'audio': 1})}'",
    ]

    # Reasoning + tool use — improves judge output quality
    cmd += [
        "--enable-auto-tool-choice",
        "--reasoning-parser", "gemma4",
        "--tool-call-parser", "gemma4",
    ]

    print("Starting vLLM:", " ".join(cmd))
    subprocess.Popen(" ".join(cmd), shell=True)


# ---------------------------------------------------------------------------
# Smoke test (run with: modal run modal_app/gemma_judge.py)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
async def smoke_test():
    """Quick sanity check: deploy → healthcheck → one text completion."""
    import aiohttp

    url = await serve.get_web_url.aio()
    print(f"Server URL: {url}")

    async with aiohttp.ClientSession(base_url=url) as session:
        # Health check
        async with session.get("/health", timeout=9 * MINUTES) as resp:
            assert resp.status == 200, f"Health check failed: {resp.status}"
        print("Health check passed.")

        # One text completion
        payload = {
            "model": "llm",
            "stream": False,
            "messages": [
                {"role": "user", "content": "Reply with exactly one word: hello"}
            ],
        }
        async with session.post("/v1/chat/completions", json=payload) as resp:
            result = await resp.json()
        content = result["choices"][0]["message"]["content"]
        print(f"Smoke test response: {content!r}")
        assert content.strip().lower().startswith("hello"), f"Unexpected response: {content!r}"
        print("Smoke test passed.")
