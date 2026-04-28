"""Environment-backed settings.

Loads secrets and runtime configuration from environment variables (or a .env
file in dev). Nothing in the rest of the package reads os.environ directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class RuntimeConfig:
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_from_number: str
    public_base_url: str
    session_root: Path
    log_level: str = "info"
    validate_twilio_signature: bool = True

    @classmethod
    def from_env(cls, *, load_dotenv_file: bool = True) -> RuntimeConfig:
        if load_dotenv_file:
            load_dotenv()

        required = {
            "TWILIO_ACCOUNT_SID": os.environ.get("TWILIO_ACCOUNT_SID"),
            "TWILIO_AUTH_TOKEN": os.environ.get("TWILIO_AUTH_TOKEN"),
            "TWILIO_PHONE_NUMBER": os.environ.get("TWILIO_PHONE_NUMBER"),
            "BASE_URL": os.environ.get("BASE_URL"),
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
            session_root=session_root,
            log_level=os.environ.get("LOG_LEVEL", "info"),
            validate_twilio_signature=os.environ.get("VALIDATE_TWILIO_SIGNATURE", "1") != "0",
        )
