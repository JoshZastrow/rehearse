"""Integration tests for the Modal-hosted Gemma 4 vLLM judge server.

All tests hit a live Modal deployment and are gated behind the ``live_modal``
marker.  They are excluded from the default test run and require:

    export VLLM_BASE_URL=https://<workspace>--rehearse-gemma-judge-serve.modal.run/v1

Run with::

    pytest -m live_modal tests/integration/test_modal_judge.py -v

Tests use ``httpx.AsyncClient`` directly (no openai SDK) so failures surface as
raw HTTP/JSON errors rather than SDK-level exceptions.
"""

from __future__ import annotations

import json

import httpx
import pytest

# Import directly from the module file to avoid triggering rehearse/eval/scorers/__init__.py
# which pulls in deepeval (an optional dependency not always installed in this env).
import importlib.util as _ilu
import pathlib as _pl

_llm_judge_path = _pl.Path(__file__).parents[2] / "rehearse" / "eval" / "scorers" / "llm_judge.py"
_llm_judge_mod = _ilu.module_from_spec(
    spec := _ilu.spec_from_file_location("rehearse.eval.scorers.llm_judge", _llm_judge_path)
)
spec.loader.exec_module(_llm_judge_mod)
_TRAJECTORY_JUDGE_SYSTEM: str = _llm_judge_mod._TRAJECTORY_JUDGE_SYSTEM  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FULL_MODEL_NAME = "google/gemma-4-26B-A4B-it"
_MODEL_ALIAS = "llm"

# Short synthetic coaching transcript — 3 turns, enough to exercise the rubric.
_SYNTHETIC_TRANSCRIPT = (
    "Scenario:\n"
    "  Situation: Annual performance review with a critical manager\n"
    "  User goal: Ask for a raise without damaging the relationship\n"
    "  Counterparty: direct manager — blunt, data-driven\n"
    "  Stakes: medium — job security not at risk, but raise matters\n"
    "  User's opening emotional state: anxious, slightly defensive\n\n"
    "Transcript:\n"
    "[0] USER: I'm really nervous. My manager always cuts me off.\n"
    "[1] COACH: That sounds stressful. Let's practice a calm opener. "
    "Try: 'I'd like to walk through some results I'm proud of before we discuss compensation.'\n"
    "[2] USER: Okay, that feels less confrontational. What if he still interrupts?\n"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base(vllm_base_url: str) -> str:
    """Strip a trailing /v1 to get the server root URL."""
    return vllm_base_url.rstrip("/").removesuffix("/v1")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.live_modal
@pytest.mark.asyncio
async def test_health_endpoint(vllm_base_url: str) -> None:
    """GET /health → 200.

    Surfaces: server not deployed, wrong VLLM_BASE_URL.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{_base(vllm_base_url)}/health")
    assert resp.status_code == 200, (
        f"Health check failed ({resp.status_code}): {resp.text[:200]}"
    )


@pytest.mark.live_modal
@pytest.mark.asyncio
async def test_models_endpoint_lists_gemma4(vllm_base_url: str) -> None:
    """GET /v1/models → both model names appear in data[].id.

    This is the key gap test.  The smoke test in infra/judge.py uses ``"llm"``
    but LLMJudge will send ``JUDGE_MODEL=google/gemma-4-26B-A4B-it``.
    Both aliases must be registered for the end-to-end flow to work.

    vLLM command in infra/judge.py passes ``--served-model-name MODEL_NAME llm``
    as separate argv elements joined with shell=True — ``"llm"`` becomes the
    second value.  If that shell expansion is broken, only one name will appear.

    GAP note: if ``"llm"`` is NOT in the returned ids, the smoke test in
    infra/judge.py will fail; either fix the vLLM command or update the smoke
    test to use the full model name.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{vllm_base_url}/models")
    assert resp.status_code == 200, f"GET /v1/models failed: {resp.text[:200]}"

    body = resp.json()
    ids = {m["id"] for m in body.get("data", [])}

    assert _FULL_MODEL_NAME in ids, (
        f"Full model name {_FULL_MODEL_NAME!r} not found in /v1/models. "
        f"Got: {ids}. "
        "Check that --served-model-name includes the full model name."
    )
    assert _MODEL_ALIAS in ids, (
        f"Alias {_MODEL_ALIAS!r} not found in /v1/models. Got: {ids}. "
        "GAP: llm alias not registered — smoke test in infra/judge.py will fail; "
        "either fix the vLLM command (--served-model-name MODEL_NAME llm) or "
        "update the smoke test to use the full model name."
    )


@pytest.mark.live_modal
@pytest.mark.asyncio
async def test_simple_completion_with_full_model_name(vllm_base_url: str) -> None:
    """POST /v1/chat/completions with the full model name → non-empty response."""
    payload = {
        "model": _FULL_MODEL_NAME,
        "stream": False,
        "messages": [{"role": "user", "content": "Reply with exactly: yes"}],
        "max_tokens": 16,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{vllm_base_url}/chat/completions", json=payload)
    assert resp.status_code == 200, (
        f"Completion with full model name failed ({resp.status_code}): {resp.text[:400]}"
    )
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    assert content, "Response content is empty"


@pytest.mark.live_modal
@pytest.mark.asyncio
async def test_simple_completion_with_llm_alias(vllm_base_url: str) -> None:
    """POST /v1/chat/completions with the ``llm`` alias → non-empty response.

    Surfaces: alias routing broken (second --served-model-name value not wired up).
    """
    payload = {
        "model": _MODEL_ALIAS,
        "stream": False,
        "messages": [{"role": "user", "content": "Reply with exactly: yes"}],
        "max_tokens": 16,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{vllm_base_url}/chat/completions", json=payload)
    assert resp.status_code == 200, (
        f"Completion with alias 'llm' failed ({resp.status_code}): {resp.text[:400]}"
    )
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    assert content, "Response content is empty when using 'llm' alias"


@pytest.mark.live_modal
@pytest.mark.asyncio
async def test_judge_shaped_prompt_returns_parseable_json(vllm_base_url: str) -> None:
    """Critical path: send the real LLMJudge prompt shape, parse JSON response.

    Uses the actual ``_TRAJECTORY_JUDGE_SYSTEM`` prompt from llm_judge.py so
    any future prompt edits are automatically exercised here.

    Asserts:
    - Response parses as a JSON object.
    - Top-level keys ``emotion_responsiveness`` and ``coaching_trajectory_quality``
      are present.
    - Each dimension has a ``score`` that can be cast to float in [0, 1].
    """
    payload = {
        "model": _FULL_MODEL_NAME,
        "stream": False,
        "messages": [
            {"role": "system", "content": _TRAJECTORY_JUDGE_SYSTEM},
            {"role": "user", "content": _SYNTHETIC_TRANSCRIPT},
        ],
        "max_tokens": 512,
    }
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(f"{vllm_base_url}/chat/completions", json=payload)
    assert resp.status_code == 200, (
        f"Judge-shaped completion failed ({resp.status_code}): {resp.text[:400]}"
    )

    body = resp.json()
    raw_content = body["choices"][0]["message"]["content"]
    assert raw_content, "Judge response content is empty"

    # Parse JSON — allow the model to wrap it in a code fence.
    text = raw_content.strip()
    if text.startswith("```"):
        # Strip ```json ... ``` fences.
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()

    try:
        judge_output = json.loads(text)
    except json.JSONDecodeError:
        # Fall back to first {...} block (matches LLMJudge._parse_json behaviour).
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        assert match, (
            f"Could not find a JSON object in judge output: {raw_content[:400]!r}"
        )
        judge_output = json.loads(match.group(0))

    assert isinstance(judge_output, dict), (
        f"Judge output is not a dict: {type(judge_output)}"
    )

    for dim in ("emotion_responsiveness", "coaching_trajectory_quality"):
        assert dim in judge_output, (
            f"Missing dimension {dim!r} in judge output. Got keys: {list(judge_output)}"
        )
        block = judge_output[dim]
        assert isinstance(block, dict), f"{dim} value is not a dict: {block!r}"
        assert "score" in block, f"{dim} missing 'score' key. Block: {block!r}"
        score = float(block["score"])
        assert 0.0 <= score <= 1.0, f"{dim}.score={score} is out of [0, 1] range"


@pytest.mark.live_modal
@pytest.mark.xfail(strict=False, reason="tool-call parser may not be configured in this vLLM build")
@pytest.mark.asyncio
async def test_tool_call_capability(vllm_base_url: str) -> None:
    """POST with a tool definition → tool call response OR graceful text fallback.

    ``--tool-call-parser gemma4`` is in the vLLM config but not verified at
    deploy time.  This test documents whether Gemma 4's tool-call path is
    working.  It is marked xfail(strict=False) so a failure is recorded but
    does not block CI.

    Acceptance criteria (either is a pass):
    - ``choices[0].finish_reason == "tool_calls"`` and
      ``choices[0].message.tool_calls`` is non-empty.
    - OR ``choices[0].message.content`` is truthy (graceful text fallback).

    A 5xx response is a failure regardless.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_score",
                "description": "Return a numeric quality score.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "score": {
                            "type": "number",
                            "description": "Quality score between 0 and 1.",
                        }
                    },
                    "required": ["score"],
                },
            },
        }
    ]
    payload = {
        "model": _FULL_MODEL_NAME,
        "stream": False,
        "messages": [
            {"role": "user", "content": "Rate the quality of 'hello world' on a 0-1 scale."}
        ],
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 128,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{vllm_base_url}/chat/completions", json=payload)

    # 5xx is always a hard failure.
    assert resp.status_code < 500, (
        f"Server error on tool-call request ({resp.status_code}): {resp.text[:400]}"
    )

    body = resp.json()
    choice = body["choices"][0]
    message = choice.get("message", {})

    has_tool_call = bool(message.get("tool_calls"))
    has_content = bool(message.get("content"))

    assert has_tool_call or has_content, (
        f"Neither tool_calls nor content in response. message={message!r}"
    )


@pytest.mark.live_modal
@pytest.mark.asyncio
async def test_audio_input_not_supported_for_text_only_calls(vllm_base_url: str) -> None:
    """Text-only completion is not broken by the audio multimodal config.

    ``--limit-mm-per-prompt`` is set in infra/judge.py to allow 1 audio clip
    per prompt.  The value has extra single-quotes in the shell command::

        f"'{json.dumps({'image': 0, 'video': 0, 'audio': 1})}'"

    When expanded via ``shell=True`` + ``" ".join(cmd)``, this produces::

        --limit-mm-per-prompt '{"image": 0, "video": 0, "audio": 1}'

    with literal single-quotes that the shell strips at runtime.  If the
    single-quotes are NOT stripped (e.g. wrong quoting in the subprocess call),
    vLLM may reject the flag or default to no multimodal support.

    GAP note: if audio judge calls fail with a 4xx/5xx, check that
    ``--limit-mm-per-prompt`` was parsed correctly (vLLM logs will show the
    effective mm-per-prompt config at startup).

    This test just confirms text-only calls return 200 — it does not send
    audio, so it will pass even if the audio config is broken.
    """
    payload = {
        "model": _FULL_MODEL_NAME,
        "stream": False,
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 16,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{vllm_base_url}/chat/completions", json=payload)
    assert resp.status_code == 200, (
        f"Text-only call failed ({resp.status_code}) — audio config may have "
        f"broken the server: {resp.text[:400]}"
    )
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    assert content, "Text-only response content is empty"
