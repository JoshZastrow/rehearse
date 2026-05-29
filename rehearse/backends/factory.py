"""create_backend — select a ConversationBackend implementation from config.

BACKEND_TYPE=managed      → ManagedBackend (default, uses Hume EVI)
BACKEND_TYPE=pipeline     → PipelineBackend (Pipecat, open-source stack)
BACKEND_TYPE=interactive  → InteractiveBackend (Moshi full-duplex, local GPU)
"""

from __future__ import annotations

from rehearse.backends.base import ConversationBackend
from rehearse.config import RuntimeConfig


def create_backend(config: RuntimeConfig) -> ConversationBackend:
    """Return the ConversationBackend selected by config.backend_type."""
    match config.backend_type:
        case "managed":
            from rehearse.backends.managed import ManagedBackend
            return ManagedBackend(
                api_key=config.managed_api_key,
                config_id=config.managed_config_id,
            )
        case "pipeline":
            from rehearse.backends.pipeline import PipelineBackend
            return PipelineBackend(
                speech_mode=config.pipeline_speech_mode,
                stt_model=config.pipeline_stt_model,
                tts_model=config.pipeline_tts_model,
                clm_url=config.pipeline_clm_url,
                clm_model=config.pipeline_clm_model,
            )
        case "interactive" | "moshi":  # "moshi" kept as deprecated alias
            from rehearse.backends.interactive import InteractiveBackend
            return InteractiveBackend(
                checkpoint_path=config.interactive_checkpoint_path,
                hf_repo=config.interactive_model_repo,
                device=config.interactive_device,
                asr_model=config.interactive_asr_model,
            )
        case _:
            raise ValueError(f"Unknown backend_type: {config.backend_type!r}")
