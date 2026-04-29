"""Phase R1 acceptance: SMS triggers session, voice webhook returns hard-coded
TwiML, status callback finalizes the manifest. No real Twilio calls."""

from __future__ import annotations

import asyncio
import base64
import json
import struct
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rehearse.app import create_app
from rehearse.audio.mulaw import encode_pcm16
from rehearse.config import RuntimeConfig
from rehearse.frames import AudioChunk, ProsodyEvent, TranscriptDelta
from rehearse.telephony import TwilioRestClient
from rehearse.types import (
    CounterpartyPersona,
    IntakeRecord,
    Phase,
    PhaseTiming,
    ProsodyFrame,
    ProsodyScores,
    ProsodySource,
    Session,
    Speaker,
    TranscriptFrame,
)


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
        hume_api_key="hume_test_key",
        hume_config_id="cfg_test",
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
    assert "<Connect>" in body
    assert "/media/deadbeef" in body


def test_status_callback_finalizes_session(app_client) -> None:
    client, fake, config = app_client

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
    assert manifest["artifact_paths"]["story"] == "story.md"
    assert manifest["artifact_paths"]["feedback"] == "feedback.md"
    assert fake.messages == [
        (
            "+15551234567",
            f"Your Rehearse session is ready. View your artifacts here: "
            f"https://example.test/viewer?session_id={session_id}",
        )
    ]


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
    assert "<Connect>" in resp.text
    assert "/media/" in resp.text

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


def test_viewer_renders_runtime_artifacts(app_client) -> None:
    client, _, config = app_client
    session = Session(id="viewer-session", created_at=datetime.now(UTC))
    session.intake = IntakeRecord(
        session_id=session.id,
        situation="Comp negotiation",
        counterparty_name="Dana",
        counterparty_relationship="Hiring manager",
        counterparty_description="Direct and budget-conscious",
        stakes="Offer decision this week",
        user_goal="Increase base pay",
        desired_tone="Calm and direct",
        captured_at=datetime.now(UTC),
    )
    session.persona = CounterpartyPersona(
        session_id=session.id,
        name="Dana",
        relationship="Hiring manager",
        personality_prompt="Pushes back on budget and timing.",
        hot_buttons=["Internal band limits"],
        likely_reactions=["Deflects to equity"],
        compiled_at=datetime.now(UTC),
    )
    session.phase_timings = [
        PhaseTiming(
            phase=Phase.INTAKE,
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
            budget_seconds=60,
            overran=False,
        )
    ]
    session.artifact_paths = {
        "transcript": "transcript.jsonl",
        "prosody": "prosody.jsonl",
        "audio": "audio.wav",
        "story": "story.md",
        "feedback": "feedback.md",
    }
    (config.session_root / session.id).mkdir()
    (config.session_root / session.id / "session.json").write_text(
        session.model_dump_json(indent=2)
    )
    (config.session_root / session.id / "transcript.jsonl").write_text(
        TranscriptFrame(
            session_id=session.id,
            utterance_id="u1",
            speaker=Speaker.USER,
            phase=Phase.INTAKE,
            text="I want to negotiate the offer.",
            ts_start=0.0,
            ts_end=0.5,
            is_interim=False,
        ).model_dump_json()
        + "\n"
    )
    (config.session_root / session.id / "prosody.jsonl").write_text(
        ProsodyFrame(
            session_id=session.id,
            utterance_id="u1",
            speaker=Speaker.USER,
            source=ProsodySource.HUME_LIVE,
            scores=ProsodyScores(arousal=0.7, valence=0.1, emotions={"nervousness": 0.8}),
            ts_start=0.0,
            ts_end=0.5,
        )
        .model_dump_json()
        + "\n"
    )
    (config.session_root / session.id / "audio.wav").write_bytes(b"RIFFfake")
    (config.session_root / session.id / "story.md").write_text("# Story\n\nA negotiation run.")
    (config.session_root / session.id / "feedback.md").write_text("# Feedback\n\nBe more direct.")

    resp = client.get("/viewer", params={"session_id": session.id})

    assert resp.status_code == 200
    assert "Rehearse Session Viewer" in resp.text
    assert "Comp negotiation" in resp.text
    assert "Dana" in resp.text
    assert "I want to negotiate the offer." in resp.text
    assert "nervousness 0.80" in resp.text
    assert "/sessions/viewer-session/audio.wav" in resp.text
    assert "# Feedback" in resp.text


def test_viewer_returns_404_for_unknown_session(app_client) -> None:
    client, _, _ = app_client
    resp = client.get("/viewer", params={"session_id": "missing-session"})
    assert resp.status_code == 404
    assert resp.json() == {"detail": "unknown session"}


def test_media_websocket_bridges_twilio_to_fake_hume(
    app_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, _config = app_client
    session_dir = _config.session_root / "test-session"
    session_dir.mkdir()
    (session_dir / "session.json").write_text(
        Session(id="test-session", created_at=datetime.now(UTC)).model_dump_json(indent=2)
    )

    class FakeHumeEVIClient:
        seen_audio: list[bytes] = []

        def __init__(self, *, api_key: str, config_id: str, bus, session_id: str) -> None:
            self._bus = bus
            self._session_id = session_id
            assert api_key == "hume_test_key"
            assert config_id == "cfg_test"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def send_audio(self, pcm16_16k: bytes) -> None:
            self.seen_audio.append(pcm16_16k)
            await self._bus.publish(
                TranscriptDelta(
                    session_id=self._session_id,
                    utterance_id="user-1",
                    speaker=Speaker.USER,
                    text="user said hello",
                    is_final=True,
                    ts_start=0.0,
                    ts_end=0.1,
                )
            )
            await self._bus.publish(
                ProsodyEvent(
                    session_id=self._session_id,
                    utterance_id="user-1",
                    speaker=Speaker.USER,
                    scores=ProsodyScores(arousal=0.3, valence=0.1, emotions={"joy": 0.2}),
                    ts_start=0.0,
                    ts_end=0.1,
                )
            )
            await self._bus.publish(
                TranscriptDelta(
                    session_id=self._session_id,
                    utterance_id="coach-1",
                    speaker=Speaker.COACH,
                    text="coach reply",
                    is_final=True,
                    ts_start=0.2,
                    ts_end=0.3,
                )
            )
            await self._bus.publish(
                TranscriptDelta(
                    session_id=self._session_id,
                    utterance_id="user-2",
                    speaker=Speaker.USER,
                    text="Let's practice asking for more salary and equity.",
                    is_final=True,
                    ts_start=0.31,
                    ts_end=0.4,
                )
            )
            await self._bus.publish(
                AudioChunk(
                    session_id=self._session_id,
                    speaker=Speaker.COACH,
                    pcm16_16k=struct.pack("<8h", 0, 50, 100, 150, 200, 150, 100, 50),
                    ts=0.0,
                )
            )

        async def run_event_loop(self) -> None:
            await asyncio.Event().wait()

    from rehearse import telephony as telephony_module

    monkeypatch.setattr(telephony_module, "HumeEVIClient", FakeHumeEVIClient)

    pcm8k = struct.pack("<4h", 0, 1000, -1000, 0)
    payload = base64.b64encode(encode_pcm16(pcm8k)).decode("ascii")
    with client.websocket_connect("/media/test-session") as ws:
        ws.send_json({"event": "connected"})
        ws.send_json(
            {
                "event": "start",
                "start": {
                    "streamSid": "MZ123",
                    "callSid": "CA123",
                    "customParameters": {"session_id": "test-session"},
                },
            }
        )
        ws.send_json({"event": "media", "media": {"payload": payload}})
        outbound = ws.receive_json()
        ws.send_json({"event": "stop"})

    assert FakeHumeEVIClient.seen_audio
    assert outbound["event"] == "media"
    assert outbound["streamSid"] == "MZ123"
    assert (session_dir / "transcript.jsonl").exists()
    assert (session_dir / "prosody.jsonl").exists()
    assert (session_dir / "audio.wav").exists()
    assert (session_dir / "telemetry.jsonl").exists()
    manifest = json.loads((session_dir / "session.json").read_text())
    assert manifest["phase_timings"][0]["phase"] == "intake"
    assert manifest["intake"]["counterparty_relationship"] == "counterparty"


def test_twilio_sms_signature_validation_failure(
    config: RuntimeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RejectingValidator:
        def __init__(self, _token: str) -> None:
            pass

        def validate(self, _url: str, _params: dict[str, str], _signature: str) -> bool:
            return False

    from rehearse import app as app_module
    from rehearse import telephony as telephony_module

    monkeypatch.setattr(app_module, "TwilioRestClient", lambda cfg: FakeTwilioClient())
    monkeypatch.setattr(telephony_module, "RequestValidator", RejectingValidator)

    signed_config = RuntimeConfig(
        **{
            **config.__dict__,
            "validate_twilio_signature": True,
        }
    )
    client = TestClient(create_app(signed_config))

    resp = client.post("/twilio/sms", data={"From": "+15551234567", "Body": "hello"})
    assert resp.status_code == 403


def test_twilio_sms_failed_outbound_call_marks_session_failed(
    config: RuntimeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingTwilioClient(FakeTwilioClient):
        async def place_call(self, to: str, callback_url: str, status_callback: str) -> str:
            raise RuntimeError("boom")

    from rehearse import app as app_module

    monkeypatch.setattr(app_module, "TwilioRestClient", lambda cfg: FailingTwilioClient())
    client = TestClient(create_app(config))

    resp = client.post("/twilio/sms", data={"From": "+15551234567", "Body": "hello"})
    assert resp.status_code == 200

    session_dir = next(config.session_root.iterdir())
    manifest = json.loads((session_dir / "session.json").read_text())
    assert manifest["completion_status"] == "failed"


@pytest.mark.asyncio
async def test_twilio_rest_client_wraps_underlying_sdk_calls() -> None:
    class FakeCallsApi:
        def __init__(self) -> None:
            self.last_kwargs: dict[str, object] | None = None

        def create(self, **kwargs):
            self.last_kwargs = kwargs
            return type("Call", (), {"sid": "CA_test"})()

    class FakeMessagesApi:
        def __init__(self) -> None:
            self.last_kwargs: dict[str, object] | None = None

        def create(self, **kwargs):
            self.last_kwargs = kwargs
            return type("Message", (), {"sid": "SM_test"})()

    fake_calls = FakeCallsApi()
    fake_messages = FakeMessagesApi()

    class FakeSdkClient:
        def __init__(self, sid: str, token: str) -> None:
            assert sid == "AC_test"
            assert token == "test_token"
            self.calls = fake_calls
            self.messages = fake_messages

    from rehearse import telephony as telephony_module

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(telephony_module, "TwilioClient", FakeSdkClient)
    client = TwilioRestClient(
        RuntimeConfig(
            twilio_account_sid="AC_test",
            twilio_auth_token="test_token",
            twilio_from_number="+15555550100",
            public_base_url="https://example.test",
            hume_api_key="hume_test_key",
            hume_config_id="cfg_test",
            session_root=Path("/tmp/rehearse-tests"),
            log_level="warning",
            validate_twilio_signature=False,
        )
    )

    assert await client.place_call("+15551234567", "https://voice", "https://status") == "CA_test"
    assert await client.send_sms("+15551234567", "hello") == "SM_test"
    assert fake_calls.last_kwargs is not None
    assert fake_messages.last_kwargs is not None
    monkeypatch.undo()
