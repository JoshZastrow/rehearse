from __future__ import annotations

from pathlib import Path

import pytest

from rehearse.config import RuntimeConfig


def test_runtime_config_from_env_loads_required_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "token")
    monkeypatch.setenv("TWILIO_PHONE_NUMBER", "+15555550100")
    monkeypatch.setenv("BASE_URL", "https://example.test/")
    monkeypatch.setenv("HUME_API_KEY", "hume-key")
    monkeypatch.setenv("HUME_CONFIG_ID", "cfg-123")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-3.7-test")
    monkeypatch.setenv("HUME_CLM_SECRET", "secret-123")
    monkeypatch.setenv("SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("VALIDATE_TWILIO_SIGNATURE", "0")

    config = RuntimeConfig.from_env(load_dotenv_file=False)

    assert config.twilio_account_sid == "AC123"
    assert config.public_base_url == "https://example.test"
    assert config.hume_api_key == "hume-key"
    assert config.hume_config_id == "cfg-123"
    assert config.anthropic_api_key == "anthropic-key"
    assert config.anthropic_model == "claude-3.7-test"
    assert config.hume_clm_secret == "secret-123"
    assert config.log_level == "debug"
    assert config.validate_twilio_signature is False
    assert config.session_root == (tmp_path / "sessions").resolve()


def test_runtime_config_from_env_raises_for_missing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in [
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_PHONE_NUMBER",
        "BASE_URL",
        "HUME_API_KEY",
        "HUME_CONFIG_ID",
    ]:
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(RuntimeError, match="Missing required env vars"):
        RuntimeConfig.from_env(load_dotenv_file=False)
