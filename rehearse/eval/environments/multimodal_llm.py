"""Audio-native multimodal LLM environment."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rehearse.eval.protocols import BenchmarkExample, RolloutResult
from rehearse.eval.providers import get_provider
from rehearse.eval.providers.base import AudioInput, ProviderError


class MultimodalLLMEnvironment:
    name = "multimodal-llm"
    version = "v0"

    def __init__(self, model_slots: dict[str, str] | None = None) -> None:
        self.model_slots = model_slots or {}
        self.provider_name = self.model_slots.get("provider")

    async def rollout(
        self,
        example: BenchmarkExample,
        run_dir: Path,
        rng_seed: int,
    ) -> RolloutResult:
        started = datetime.now()
        prompt = example.payload.get("prompt")
        media = example.payload.get("audio_path") or example.payload.get("video_path")
        if not prompt:
            return _error(
                example,
                self.name,
                self.version,
                started,
                "example.payload missing 'prompt'",
            )
        if not media:
            return _error(
                example,
                self.name,
                self.version,
                started,
                "example.payload missing 'audio_path' or 'video_path'",
            )
        if not self.provider_name:
            return _error(
                example,
                self.name,
                self.version,
                started,
                "model slot 'provider' must be set to gemini or vllm",
            )

        media_path = Path(str(media))
        duration_s = float(example.metadata.get("duration_s") or 0.0)
        audio_max_s = example.payload.get("audio_max_s")
        if audio_max_s is not None and duration_s > float(audio_max_s):
            return _error(
                example,
                self.name,
                self.version,
                started,
                "audio_exceeds_provider_limit",
            )
        if not media_path.exists():
            return _error(
                example,
                self.name,
                self.version,
                started,
                f"media file not found: {media_path}",
            )

        provider = get_provider(self.provider_name, self.model_slots)
        try:
            response = await provider.complete(
                AudioInput(path=media_path, duration_s=duration_s),
                str(prompt),
                max_tokens=int(example.payload.get("max_tokens", 512)),
                temperature=float(example.payload.get("temperature", 0.0)),
                timeout_s=int(example.payload.get("timeout_s", 60)),
            )
        except ProviderError as exc:
            return _error(example, self.name, self.version, started, str(exc))

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
                "output": response.text,
                "model": response.provider_metadata.get("model"),
                "provider": provider.name,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "audio_duration_s": response.audio_duration_s,
                "finish_reason": response.raw_finish_reason,
            },
        )


def _error(
    example: BenchmarkExample,
    target_name: str,
    target_version: str,
    started: datetime,
    error: str,
) -> RolloutResult:
    completed = datetime.now()
    return RolloutResult(
        example_id=example.id,
        target_name=target_name,
        target_version=target_version,
        status="error",
        started_at=started,
        completed_at=completed,
        duration_ms=int((completed - started).total_seconds() * 1000),
        error=error,
    )
