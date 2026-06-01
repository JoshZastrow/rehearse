"""Tests for the backend factory."""
from __future__ import annotations

from pathlib import Path


def test_create_backend_interactive_returns_interactive_backend():
    from rehearse.backends.factory import create_backend
    from rehearse.backends.interactive import InteractiveBackend
    from rehearse.config import RuntimeConfig

    cfg = RuntimeConfig(
        twilio_account_sid="x",
        twilio_auth_token="x",
        twilio_from_number="x",
        public_base_url="x",
        hume_api_key="x",
        hume_config_id="x",
        session_root=Path("/tmp"),
        backend_type="interactive",
        interactive_endpoint="wss://example.modal.run/ws",
    )
    backend = create_backend(cfg)
    assert isinstance(backend, InteractiveBackend)


def test_create_backend_unknown_raises():
    from rehearse.backends.factory import create_backend
    from rehearse.config import RuntimeConfig

    cfg = RuntimeConfig(
        twilio_account_sid="x",
        twilio_auth_token="x",
        twilio_from_number="x",
        public_base_url="x",
        hume_api_key="x",
        hume_config_id="x",
        session_root=Path("/tmp"),
        backend_type="bogus",
    )
    try:
        create_backend(cfg)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "bogus" in str(exc)
