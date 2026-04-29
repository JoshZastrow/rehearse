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
    hume_clm_secret: str | None = None
    log_level: str = "info"
    validate_twilio_signature: bool = True

    @classmethod
    def from_env(cls, *, load_dotenv_file: bool = True) -> RuntimeConfig:
        """Build a runtime config from environment variables and return it."""
        if load_dotenv_file:
            load_dotenv()

        required = {
            "TWILIO_ACCOUNT_SID": os.environ.get("TWILIO_ACCOUNT_SID"),
            "TWILIO_AUTH_TOKEN": os.environ.get("TWILIO_AUTH_TOKEN"),
            "TWILIO_PHONE_NUMBER": os.environ.get("TWILIO_PHONE_NUMBER"),
            "BASE_URL": os.environ.get("BASE_URL"),
            "HUME_API_KEY": os.environ.get("HUME_API_KEY"),
            "HUME_CONFIG_ID": os.environ.get("HUME_CONFIG_ID"),
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
            hume_clm_secret=os.environ.get("HUME_CLM_SECRET"),
            log_level=os.environ.get("LOG_LEVEL", "info"),
            validate_twilio_signature=os.environ.get("VALIDATE_TWILIO_SIGNATURE", "1") != "0",
        )
