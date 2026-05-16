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


def _set_required_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Set the minimum env vars required for `RuntimeConfig.from_env`."""
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "token")
    monkeypatch.setenv("TWILIO_PHONE_NUMBER", "+15555550100")
    monkeypatch.setenv("BASE_URL", "https://example.test/")
    monkeypatch.setenv("HUME_API_KEY", "hume-key")
    monkeypatch.setenv("HUME_CONFIG_ID", "cfg-123")
    monkeypatch.setenv("SESSIONS_DIR", str(tmp_path / "sessions"))


@pytest.mark.parametrize(
    "env_var,attr,value,expected",
    [
        ("CONSENT_PROMPT_TIMEOUT_SECONDS", "consent_prompt_timeout_seconds", "30", 30),
        ("CONSENT_REPROMPT_LIMIT", "consent_reprompt_limit", "2", 2),
        ("OUTCOME_PROMPT_LEAD_SECONDS", "outcome_prompt_lead_seconds", "10", 10),
        ("OUTCOME_RESPONSE_TIMEOUT_SECONDS", "outcome_response_timeout_seconds", "20", 20),
        ("OUTCOME_REPROMPT_LIMIT", "outcome_reprompt_limit", "0", 0),
    ],
)
def test_runtime_config_loads_consent_outcome_knobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    env_var: str,
    attr: str,
    value: str,
    expected: int,
) -> None:
    """Each consent/outcome knob from spec §9 reads from its env var."""
    _set_required_env(monkeypatch, tmp_path)
    monkeypatch.setenv(env_var, value)
    config = RuntimeConfig.from_env(load_dotenv_file=False)
    assert getattr(config, attr) == expected


def test_runtime_config_consent_outcome_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Defaults match spec §9 when the knobs are not set in the environment."""
    _set_required_env(monkeypatch, tmp_path)
    for key in [
        "CONSENT_PROMPT_TIMEOUT_SECONDS",
        "CONSENT_REPROMPT_LIMIT",
        "OUTCOME_PROMPT_LEAD_SECONDS",
        "OUTCOME_RESPONSE_TIMEOUT_SECONDS",
        "OUTCOME_REPROMPT_LIMIT",
    ]:
        monkeypatch.delenv(key, raising=False)
    config = RuntimeConfig.from_env(load_dotenv_file=False)
    assert config.consent_prompt_timeout_seconds == 20
    assert config.consent_reprompt_limit == 1
    assert config.outcome_prompt_lead_seconds == 15
    assert config.outcome_response_timeout_seconds == 15
    assert config.outcome_reprompt_limit == 1


def test_config_has_backend_type_field():
    import dataclasses
    from rehearse.config import RuntimeConfig
    fields = {f.name for f in dataclasses.fields(RuntimeConfig)}
    assert "backend_type" in fields
    assert "managed_api_key" in fields
    assert "managed_config_id" in fields


def test_config_backend_type_defaults_to_managed():
    from pathlib import Path
    from rehearse.config import RuntimeConfig
    cfg = RuntimeConfig(
        twilio_account_sid="x",
        twilio_auth_token="x",
        twilio_from_number="+10000000000",
        public_base_url="https://example.com",
        hume_api_key="k",
        hume_config_id="c",
        session_root=Path("/tmp"),
    )
    assert cfg.backend_type == "managed"
    assert cfg.managed_api_key == "k"
    assert cfg.managed_config_id == "c"
