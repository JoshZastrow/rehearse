"""raw-llm target — single Anthropic SDK call with the example's prompt.

Used by EQ-Bench-style benchmarks. The model slot key is `raw_llm`; default
is Claude Sonnet 4.6, matching the synthesis slot per SPEC §9.

Reads the prompt and generation params from `example.payload`:
  - prompt: str (required)
  - max_tokens: int (default 1024)
  - temperature: float (default 0.0)
  - system: str (optional)
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from anthropic import AsyncAnthropic

from rehearse.eval.protocols import BenchmarkExample, RolloutResult

_DEFAULT_MODEL = "claude-sonnet-4-6"


class RawLLMTarget:
    name = "raw-llm"
    version = "v0"

    def __init__(self, model_slots: dict[str, str] | None = None) -> None:
        self.model_slots = model_slots or {}
        self.model = self.model_slots.get("raw_llm", _DEFAULT_MODEL)
        self._client: AsyncAnthropic | None = None

    def _client_lazy(self) -> AsyncAnthropic:
        if self._client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            self._client = AsyncAnthropic(api_key=api_key)
        return self._client

    async def rollout(
        self,
        example: BenchmarkExample,
        run_dir: Path,
        rng_seed: int,
    ) -> RolloutResult:
        started = datetime.now()
        prompt = example.payload.get("prompt")
        if not prompt:
            completed = datetime.now()
            return RolloutResult(
                example_id=example.id,
                target_name=self.name,
                target_version=self.version,
                status="error",
                started_at=started,
                completed_at=completed,
                duration_ms=int((completed - started).total_seconds() * 1000),
                error="example.payload missing 'prompt'",
            )

        kwargs = {
            "model": self.model,
            "max_tokens": int(example.payload.get("max_tokens", 1024)),
            "temperature": float(example.payload.get("temperature", 0.0)),
            "messages": [{"role": "user", "content": prompt}],
        }
        if "system" in example.payload:
            kwargs["system"] = example.payload["system"]

        client = self._client_lazy()
        try:
            resp = await client.messages.create(**kwargs)
        except Exception as exc:
            completed = datetime.now()
            return RolloutResult(
                example_id=example.id,
                target_name=self.name,
                target_version=self.version,
                status="error",
                started_at=started,
                completed_at=completed,
                duration_ms=int((completed - started).total_seconds() * 1000),
                error=f"anthropic call failed: {type(exc).__name__}: {exc}",
            )

        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        completed = datetime.now()
        return RolloutResult(
            example_id=example.id,
            target_name=self.name,
            target_version=self.version,
            status="ok",
            started_at=started,
            completed_at=completed,
            duration_ms=int((completed - started).total_seconds() * 1000),
            payload={
                "output": text,
                "model": self.model,
                "stop_reason": resp.stop_reason,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
        )
