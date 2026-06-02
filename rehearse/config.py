"""Load runtime settings from environment variables.

This file defines the configuration object used by the live runtime. It reads
Twilio, Hume, and local filesystem settings in one place so the rest of the
runtime does not touch `os.environ` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class RuntimeConfig:
    """All settings needed to run the live Twilio and Hume call path."""

    twilio_account_sid: str
    twilio_auth_token: str
    twilio_from_number: str
    public_base_url: str
    hume_api_key: str
    hume_config_id: str
    session_root: Path
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"
    litellm_base_url: str | None = None   # LITELLM_BASE_URL — proxy URL (overrides direct Anthropic)
    litellm_api_key: str | None = None    # LITELLM_API_KEY — proxy bearer token
    litellm_model: str = "coach"          # LITELLM_MODEL — model alias on the proxy
    hume_clm_secret: str | None = None
    log_level: str = "info"
    validate_twilio_signature: bool = True
    disable_sms: bool = False
    consent_prompt_timeout_seconds: int = 45
    consent_reprompt_limit: int = 1
    outcome_prompt_lead_seconds: int = 15
    outcome_response_timeout_seconds: int = 15
    outcome_reprompt_limit: int = 1
    finalize_sweep_max_call_seconds: int = 360
    finalize_sweep_grace_seconds: int = 60
    finalize_sweep_interval_seconds: int = 60
    finalize_sweep_enabled: bool = True
    honcho_api_key: str | None = None
    honcho_workspace_id: str = "rehearse"
    honcho_base_url: str | None = None  # set to http://localhost:8001 for self-hosted
    memory_mcp_url: str | None = None  # MEMORY_MCP_URL — MCP memory server endpoint
    backend_type: str = "managed"
    managed_api_key: str = ""
    managed_config_id: str = ""
    pipeline_speech_mode: str = "modular"
    pipeline_stt_model: str = "whisper-tiny"
    pipeline_tts_model: str = "kokoro"
    pipeline_clm_url: str = "http://localhost:4000/chat/completions"
    pipeline_clm_model: str = "coach"
    pipeline_e2e_model: str = ""
    pipeline_e2e_checkpoint: str = ""
    interactive_checkpoint_path: str = ""     # INTERACTIVE_CHECKPOINT_PATH — local dir; empty = HF Hub
    interactive_model_repo: str = "kyutai/moshiko-pytorch-bf16"  # INTERACTIVE_MODEL_REPO
    interactive_device: str = "cuda"         # INTERACTIVE_DEVICE — "cuda" or "cpu"
    interactive_asr_model: str = "base"      # INTERACTIVE_ASR_MODEL — whisper model size for user ASR
    interactive_model_type: str = "moshi"    # INTERACTIVE_MODEL_TYPE — "moshi" or "personaplex"
    interactive_modal_endpoint: str = ""     # INTERACTIVE_PROVIDER_ENDPOINT (or INTERACTIVE_MODAL_ENDPOINT) — wss://... ProviderServer
    enable_consent: bool = False
    enable_meeting_phase_processor: bool = False
    phase_classifier_model: str = "claude-haiku-4-5-20251001"
    phase_classifier_context_turns: int = 10

    def __post_init__(self) -> None:
        """Backfill managed_api_key / managed_config_id from hume_* if absent."""
        if not self.managed_api_key and self.hume_api_key:
            object.__setattr__(self, "managed_api_key", self.hume_api_key)
        if not self.managed_config_id and self.hume_config_id:
            object.__setattr__(self, "managed_config_id", self.hume_config_id)

    @classmethod
    def from_env(cls, *, load_dotenv_file: bool = True) -> RuntimeConfig:
        """Build a runtime config from environment variables and return it."""
        if load_dotenv_file:
            # `override=True` so `.env` is authoritative for this project. A
            # stale `ANTHROPIC_API_KEY` exported in the shell silently
            # shadowing an updated `.env` value cost us a real-call outage.
            load_dotenv(override=True)

        required = {
            "TWILIO_ACCOUNT_SID": os.environ.get("TWILIO_ACCOUNT_SID"),
            "TWILIO_AUTH_TOKEN": os.environ.get("TWILIO_AUTH_TOKEN"),
            "TWILIO_PHONE_NUMBER": os.environ.get("TWILIO_PHONE_NUMBER"),
            "BASE_URL": os.environ.get("BASE_URL"),
            "HUME_API_KEY": os.environ.get("HUME_API_KEY"),
            "HUME_CONFIG_ID": os.environ.get("HUME_CONFIG_ID")
            or os.environ.get("HUME_CONFIG_ID_COACH"),
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

        session_root = Path(os.environ.get("SESSIONS_DIR", "./sessions")).resolve()
        session_root.mkdir(parents=True, exist_ok=True)

        return cls(
            twilio_account_sid=required["TWILIO_ACCOUNT_SID"],  # type: ignore[arg-type]
            twilio_auth_token=required["TWILIO_AUTH_TOKEN"],  # type: ignore[arg-type]
            twilio_from_number=required["TWILIO_PHONE_NUMBER"],  # type: ignore[arg-type]
            public_base_url=required["BASE_URL"].rstrip("/"),  # type: ignore[union-attr]
            hume_api_key=required["HUME_API_KEY"],  # type: ignore[arg-type]
            hume_config_id=required["HUME_CONFIG_ID"],  # type: ignore[arg-type]
            session_root=session_root,
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            anthropic_model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            litellm_base_url=os.environ.get("LITELLM_BASE_URL") or None,
            litellm_api_key=os.environ.get("LITELLM_API_KEY") or None,
            litellm_model=os.environ.get("LITELLM_MODEL", "coach"),
            hume_clm_secret=os.environ.get("HUME_CLM_SECRET"),
            log_level=os.environ.get("LOG_LEVEL", "info"),
            validate_twilio_signature=os.environ.get("VALIDATE_TWILIO_SIGNATURE", "1") != "0",
            disable_sms=os.environ.get("DISABLE_SMS", "0") == "1",
            consent_prompt_timeout_seconds=int(
                os.environ.get("CONSENT_PROMPT_TIMEOUT_SECONDS", "20")
            ),
            consent_reprompt_limit=int(os.environ.get("CONSENT_REPROMPT_LIMIT", "1")),
            outcome_prompt_lead_seconds=int(
                os.environ.get("OUTCOME_PROMPT_LEAD_SECONDS", "15")
            ),
            outcome_response_timeout_seconds=int(
                os.environ.get("OUTCOME_RESPONSE_TIMEOUT_SECONDS", "15")
            ),
            outcome_reprompt_limit=int(os.environ.get("OUTCOME_REPROMPT_LIMIT", "1")),
            finalize_sweep_max_call_seconds=int(
                os.environ.get("FINALIZE_SWEEP_MAX_CALL_SECONDS", "360")
            ),
            finalize_sweep_grace_seconds=int(
                os.environ.get("FINALIZE_SWEEP_GRACE_SECONDS", "60")
            ),
            finalize_sweep_interval_seconds=int(
                os.environ.get("FINALIZE_SWEEP_INTERVAL_SECONDS", "60")
            ),
            finalize_sweep_enabled=os.environ.get("FINALIZE_SWEEP_ENABLED", "1") != "0",
            honcho_api_key=os.environ.get("HONCHO_API_KEY") or None,
            honcho_workspace_id=os.environ.get("HONCHO_WORKSPACE_ID", "rehearse"),
            honcho_base_url=os.environ.get("HONCHO_BASE_URL") or None,
            memory_mcp_url=os.environ.get("MEMORY_MCP_URL") or None,
            backend_type=os.environ.get("BACKEND_TYPE", "managed"),
            managed_api_key=(
                os.environ.get("MANAGED_API_KEY")
                or os.environ.get("HUME_API_KEY")
                or ""
            ),
            managed_config_id=(
                os.environ.get("MANAGED_CONFIG_ID")
                or os.environ.get("HUME_CONFIG_ID")
                or os.environ.get("HUME_CONFIG_ID_COACH")
                or ""
            ),
            pipeline_speech_mode=os.environ.get("PIPELINE_SPEECH_MODE", "modular"),
            pipeline_stt_model=os.environ.get("PIPELINE_STT_MODEL", "whisper-tiny"),
            pipeline_tts_model=os.environ.get("PIPELINE_TTS_MODEL", "kokoro"),
            pipeline_clm_url=os.environ.get("PIPELINE_CLM_URL") or (
                f"{os.environ['LITELLM_BASE_URL'].rstrip('/')}/chat/completions"
                if os.environ.get("LITELLM_BASE_URL")
                else "http://localhost:4000/chat/completions"
            ),
            pipeline_clm_model=os.environ.get("PIPELINE_CLM_MODEL")
                or os.environ.get("LITELLM_MODEL", "coach"),
            pipeline_e2e_model=os.environ.get("PIPELINE_E2E_MODEL", ""),
            pipeline_e2e_checkpoint=os.environ.get("PIPELINE_E2E_CHECKPOINT", ""),
            interactive_checkpoint_path=os.environ.get("INTERACTIVE_CHECKPOINT_PATH", ""),
            interactive_model_repo=os.environ.get("INTERACTIVE_MODEL_REPO", "kyutai/moshiko-pytorch-bf16"),
            interactive_device=os.environ.get("INTERACTIVE_DEVICE", "cuda"),
            interactive_asr_model=os.environ.get("INTERACTIVE_ASR_MODEL", "base"),
            interactive_model_type=os.environ.get("INTERACTIVE_MODEL_TYPE", "moshi"),
            interactive_modal_endpoint=os.environ.get("INTERACTIVE_PROVIDER_ENDPOINT") or os.environ.get("INTERACTIVE_MODAL_ENDPOINT", ""),
            enable_consent=os.environ.get("ENABLE_CONSENT", "0") == "1",
            enable_meeting_phase_processor=os.environ.get("ENABLE_MEETING_PHASE_PROCESSOR", "0") == "1",
            phase_classifier_model=os.environ.get("PHASE_CLASSIFIER_MODEL", "claude-haiku-4-5-20251001"),
            phase_classifier_context_turns=int(os.environ.get("PHASE_CLASSIFIER_CONTEXT_TURNS", "10")),
        )
