"""Phase R1 acceptance: SMS triggers session, voice webhook returns hard-coded
TwiML, status callback finalizes the manifest. No real Twilio calls."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rehearse.app import create_app
from rehearse.config import RuntimeConfig


class FakeTwilioClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.messages: list[tuple[str, str]] = []

    async def place_call(self, to: str, callback_url: str, status_callback: str) -> str:
        self.calls.append((to, callback_url, status_callback))
        return "CA_test_sid"

    async def send_sms(self, to: str, body: str) -> str:
        self.messages.append((to, body))
        return "SM_test_sid"


@pytest.fixture
def config(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        twilio_account_sid="AC_test",
        twilio_auth_token="test_token",
        twilio_from_number="+15555550100",
        public_base_url="https://example.test",
        session_root=tmp_path,
        log_level="warning",
        validate_twilio_signature=False,
    )


@pytest.fixture
def app_client(
    config: RuntimeConfig, monkeypatch: pytest.MonkeyPatch
) -> tuple[TestClient, FakeTwilioClient, RuntimeConfig]:
    fake = FakeTwilioClient()

    from rehearse import app as app_module

    monkeypatch.setattr(app_module, "TwilioRestClient", lambda cfg: fake)

    app = create_app(config)
    return TestClient(app), fake, config


def test_inbound_sms_mints_session_and_places_call(app_client) -> None:
    client, fake, config = app_client

    resp = client.post(
        "/twilio/sms",
        data={"From": "+15551234567", "Body": "I need to talk to my cofounder"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/xml")

    sessions = list(config.session_root.iterdir())
    assert len(sessions) == 1
    session_dir = sessions[0]
    manifest = json.loads((session_dir / "session.json").read_text())
    assert manifest["consent"] == "pending"
    assert manifest["completion_status"] == "in_progress"
    assert manifest["phone_number_hash"]
    assert manifest["id"] == session_dir.name

    assert len(fake.calls) == 1
    to, voice_url, status_url = fake.calls[0]
    assert to == "+15551234567"
    assert session_dir.name in voice_url
    assert session_dir.name in status_url


def test_voice_webhook_returns_hello_twiml(app_client) -> None:
    client, _, _ = app_client
    resp = client.post("/twilio/voice", params={"session_id": "deadbeef"})
    assert resp.status_code == 200
    body = resp.text
    assert "<Say>" in body
    assert "<Hangup" in body


def test_status_callback_finalizes_session(app_client) -> None:
    client, _, config = app_client

    sms = client.post("/twilio/sms", data={"From": "+15551234567", "Body": "hi"})
    assert sms.status_code == 200
    session_id = next(config.session_root.iterdir()).name

    resp = client.post(
        "/twilio/status",
        params={"session_id": session_id},
        data={"CallStatus": "completed", "CallSid": "CA_test_sid"},
    )
    assert resp.status_code == 204

    manifest = json.loads((config.session_root / session_id / "session.json").read_text())
    assert manifest["completion_status"] == "complete"


def test_status_callback_failed_marks_failed(app_client) -> None:
    client, _, config = app_client
    sms = client.post("/twilio/sms", data={"From": "+15551234567", "Body": "hi"})
    assert sms.status_code == 200
    session_id = next(config.session_root.iterdir()).name

    resp = client.post(
        "/twilio/status",
        params={"session_id": session_id},
        data={"CallStatus": "no-answer", "CallSid": "CA_test_sid"},
    )
    assert resp.status_code == 204
    manifest = json.loads((config.session_root / session_id / "session.json").read_text())
    assert manifest["completion_status"] == "failed"


def test_healthz(app_client) -> None:
    client, _, _ = app_client
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_session_creation_timestamp_recent(app_client) -> None:
    client, _, config = app_client
    client.post("/twilio/sms", data={"From": "+15551234567", "Body": "hi"})
    session_id = next(config.session_root.iterdir()).name
    manifest = json.loads((config.session_root / session_id / "session.json").read_text())
    created = datetime.fromisoformat(manifest["created_at"])
    assert (datetime.now(UTC) - created).total_seconds() < 5


def test_inbound_voice_mints_session_and_returns_twiml(app_client) -> None:
    client, fake, config = app_client

    resp = client.post(
        "/twilio/voice/inbound",
        data={"From": "+15551234567", "CallSid": "CA_inbound_1"},
    )
    assert resp.status_code == 200
    assert "<Say>" in resp.text
    assert "<Hangup" in resp.text

    sessions = list(config.session_root.iterdir())
    assert len(sessions) == 1
    manifest = json.loads((sessions[0] / "session.json").read_text())
    assert manifest["completion_status"] == "in_progress"
    assert manifest["phone_number_hash"]
    assert fake.calls == []  # no outbound call placed for the dial-in path


def test_inbound_voice_status_callback_finalizes_via_call_sid(app_client) -> None:
    client, _, config = app_client

    client.post(
        "/twilio/voice/inbound",
        data={"From": "+15551234567", "CallSid": "CA_inbound_2"},
    )
    session_id = next(config.session_root.iterdir()).name

    resp = client.post(
        "/twilio/status",
        data={"CallStatus": "completed", "CallSid": "CA_inbound_2"},
    )
    assert resp.status_code == 204
    manifest = json.loads((config.session_root / session_id / "session.json").read_text())
    assert manifest["completion_status"] == "complete"
