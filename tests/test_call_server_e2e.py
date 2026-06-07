"""End-to-end smoke test for the live call server.

Drives a full conversation against a *real* uvicorn instance of the FastAPI
runtime (not the in-process ``TestClient``) and verifies the whole lifecycle:

1. the server starts and serves traffic,
2. an inbound call mints a session and connects the media stream,
3. a full-duplex conversation runs (caller audio in, coach audio out),
4. the call ends cleanly with no traceback logged on the server,
5. the transcript is written,
6. the per-call WAVs are saved with real audio length, and
7. the server shuts down without error.

The coach backend is faked (no Modal/GPU); everything else — routing, the
Twilio media-stream protocol, the conversation pipeline, the artifact writers,
finalization — is the production code path.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import struct
import wave
from contextlib import suppress
from pathlib import Path

import httpx
import pytest
import uvicorn
import websockets

from rehearse.audio.mulaw import encode_pcm16
from rehearse.config import RuntimeConfig
from rehearse.frames import AudioChunk, EndOfCall, ProsodyEvent, TranscriptDelta
from rehearse.types import ProsodyScores, Speaker


# --- A scripted, GPU-free coach backend -------------------------------------


class _ScriptedCoachBackend:
    """Fake ConversationBackend that emits a short scripted coach turn.

    Publishes a coach transcript line plus several frames of coach audio so the
    AudioRecorder produces a non-trivial ``audio.wav`` and a coach turn WAV.
    Returns from ``start()`` immediately so telephony drives the call to
    completion off the caller's audio stream (it ends when Twilio sends "stop").
    """

    #: 20 ms of mono PCM16 @ 16 kHz = 320 samples = 640 bytes.
    _COACH_FRAME = struct.pack("<320h", *([1200, -1200] * 160))

    def __init__(self) -> None:
        self.received_caller_audio: list[bytes] = []
        self._session_id = ""

    async def __aenter__(self) -> "_ScriptedCoachBackend":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def start(self, session_id: str, bus) -> None:
        self._session_id = session_id
        # The writers each run several `await` store-registration calls before
        # they subscribe to the bus. A real backend emits over the whole call,
        # so this never matters; our scripted burst must wait for the writers
        # to attach or its frames are dropped.
        await asyncio.sleep(0.2)
        await bus.publish(
            TranscriptDelta(
                session_id=session_id,
                utterance_id="coach-1",
                speaker=Speaker.COACH,
                text="Hello, let's begin your rehearsal.",
                is_final=True,
                ts_start=0.0,
                ts_end=0.4,
            )
        )
        await bus.publish(
            ProsodyEvent(
                session_id=session_id,
                utterance_id="coach-1",
                speaker=Speaker.COACH,
                scores=ProsodyScores(arousal=0.4, valence=0.3, emotions={"calm": 0.6}),
                ts_start=0.0,
                ts_end=0.4,
            )
        )
        for _ in range(8):
            await bus.publish(
                AudioChunk(
                    session_id=session_id,
                    speaker=Speaker.COACH,
                    pcm16_16k=self._COACH_FRAME,
                    ts=0.0,
                )
            )
        await bus.publish(
            TranscriptDelta(
                session_id=session_id,
                utterance_id="user-1",
                speaker=Speaker.USER,
                text="Okay, I'm ready.",
                is_final=True,
                ts_start=0.5,
                ts_end=0.9,
            )
        )

    async def send_caller_audio(self, pcm16_16k: bytes) -> None:
        self.received_caller_audio.append(pcm16_16k)

    async def inject_speech(self, text: str) -> None:
        return None

    async def swap_persona(self, persona) -> None:
        return None

    async def say(self, request) -> None:
        return None

    async def close(self) -> None:
        return None


# --- Log capture: any ERROR/traceback on the server fails the test ----------


class _ErrorLogCollector(logging.Handler):
    """Collect WARNING+ log records emitted anywhere in the process."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    @property
    def server_errors(self) -> list[str]:
        out: list[str] = []
        for r in self.records:
            if r.levelno >= logging.ERROR:
                out.append(f"{r.name}: {r.getMessage()}")
            elif r.exc_info:  # an exception was logged even if level < ERROR
                out.append(f"{r.name} (exc): {r.getMessage()}")
        return out


# --- Test helpers ------------------------------------------------------------


def _config(session_root: Path) -> RuntimeConfig:
    return RuntimeConfig(
        twilio_account_sid="AC_e2e",
        twilio_auth_token="token_e2e",
        twilio_from_number="+15555550100",
        public_base_url="http://127.0.0.1",  # real port supplied at connect time
        hume_api_key="hume_e2e",
        hume_config_id="cfg_e2e",
        session_root=session_root,
        backend_type="fake",  # avoids the model-loading startup handlers
        log_level="warning",
        validate_twilio_signature=False,
        disable_sms=True,  # no notifier => finalize never touches Twilio
        enable_consent=False,  # consent auto-granted; keeps the pipeline short
        finalize_sweep_enabled=False,  # no background sweeper task
        interactive_provider_endpoint="",
    )


def _twilio_media_frame() -> str:
    """One 20 ms frame of caller audio, base64 μ-law as Twilio sends it."""
    pcm8k = struct.pack("<160h", *([800, -800] * 80))  # 20 ms @ 8 kHz
    return base64.b64encode(encode_pcm16(pcm8k)).decode("ascii")


def _wav_frames(path: Path) -> int:
    with wave.open(str(path), "rb") as w:
        return w.getnframes()


# --- The test ----------------------------------------------------------------


async def test_full_call_lifecycle_on_real_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)

    # Capture every WARNING+ record so we can assert the server logged no
    # tracebacks during the call or shutdown.
    collector = _ErrorLogCollector()
    watched_loggers = [logging.getLogger(), logging.getLogger("uvicorn.error")]
    for lg in watched_loggers:
        lg.addHandler(collector)
        lg.setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").propagate = True

    # Hermetic Twilio + a GPU-free coach backend.
    from rehearse.api import app as app_module
    from rehearse.api import telephony as telephony_module

    backend = _ScriptedCoachBackend()

    class _FakeTwilio:
        async def place_call(self, *a, **k) -> str:
            return "CA_e2e"

        async def send_sms(self, *a, **k) -> str:
            return "SM_e2e"

    monkeypatch.setattr(app_module, "TwilioRestClient", lambda cfg: _FakeTwilio())
    monkeypatch.setattr(
        telephony_module, "create_backend", lambda cfg, store=None: backend
    )

    app = app_module.create_app(config)

    # ---- 1. Start the server -------------------------------------------------
    uv_config = uvicorn.Config(
        app, host="127.0.0.1", port=0, log_level="warning", log_config=None
    )
    server = uvicorn.Server(uv_config)
    server.install_signal_handlers = lambda: None  # we manage shutdown ourselves
    serve_task = asyncio.create_task(server.serve())
    try:
        async with asyncio.timeout(10):
            while not server.started:
                await asyncio.sleep(0.02)
        port = server.servers[0].sockets[0].getsockname()[1]
        base = f"http://127.0.0.1:{port}"

        async with httpx.AsyncClient(base_url=base, timeout=10.0) as http:
            # health check
            assert (await http.get("/healthz")).json() == {"status": "ok"}

            # ---- 2. Start the call: inbound voice mints a session ------------
            voice = await http.post(
                "/twilio/voice/inbound",
                data={"From": "+15551234567", "CallSid": "CA_e2e"},
            )
            assert voice.status_code == 200
            m = re.search(r"/media/([\w-]+)", voice.text)
            assert m, f"no media URL in TwiML: {voice.text!r}"
            session_id = m.group(1)
            assert (tmp_path / session_id / "session.json").exists()

            # ---- 3. Run the conversation over the media websocket ------------
            ws_url = f"ws://127.0.0.1:{port}/media/{session_id}"
            async with websockets.connect(ws_url) as ws:
                await ws.send(json.dumps({"event": "connected"}))
                await ws.send(
                    json.dumps(
                        {
                            "event": "start",
                            "start": {
                                "streamSid": "MZ_e2e",
                                "callSid": "CA_e2e",
                                "customParameters": {"session_id": session_id},
                            },
                        }
                    )
                )
                # Caller speaks: send several frames of inbound audio.
                payload = _twilio_media_frame()
                for _ in range(8):
                    await ws.send(
                        json.dumps({"event": "media", "media": {"payload": payload}})
                    )
                    await asyncio.sleep(0.01)

                # Drain coach audio the server pushes back (avoid backpressure).
                # The scripted coach starts ~0.2s in, so keep reading until the
                # stream goes quiet rather than stopping at the first gap.
                coach_media = 0
                with suppress(asyncio.TimeoutError):
                    async with asyncio.timeout(3.0):
                        while True:
                            msg = await ws.recv()
                            if json.loads(msg).get("event") == "media":
                                coach_media += 1
                                if coach_media >= 5:
                                    break
                assert coach_media > 0, "server sent no coach audio back to caller"

                # ---- 4. End the call -----------------------------------------
                await ws.send(json.dumps({"event": "stop"}))
                await asyncio.sleep(0.2)  # let writers flush + close

            # Finalize as a completed call (Twilio status callback).
            status = await http.post(
                "/twilio/status",
                params={"session_id": session_id},
                data={"CallStatus": "completed", "CallSid": "CA_e2e"},
            )
            assert status.status_code == 204

        # ---- Assertions on recorded artifacts -------------------------------
        sdir = tmp_path / session_id

        # 5. Transcript written, with both speakers.
        transcript_lines = [
            json.loads(line)
            for line in (sdir / "transcript.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert transcript_lines, "transcript.jsonl is empty"
        speakers = {row["speaker"] for row in transcript_lines}
        assert "coach" in speakers and "user" in speakers, speakers

        # 6. Audio saved with real length (header + PCM data, positive frames).
        full_wav = sdir / "audio.wav"
        assert full_wav.exists()
        assert _wav_frames(full_wav) > 0, "audio.wav has no audio frames"
        coach_turn = sdir / "audio" / "coach" / "turn_0.wav"
        user_turn = sdir / "audio" / "user" / "turn_0.wav"
        assert coach_turn.exists() and _wav_frames(coach_turn) > 0
        assert user_turn.exists() and _wav_frames(user_turn) > 0

        # The backend actually received caller audio (full-duplex confirmed).
        assert backend.received_caller_audio

        # Session finalized cleanly.
        manifest = json.loads((sdir / "session.json").read_text())
        assert manifest["completion_status"] == "complete"

        # No traceback / ERROR logged on the server during the call.
        assert collector.server_errors == [], collector.server_errors

    finally:
        # ---- 7. Shut down the server without error --------------------------
        server.should_exit = True
        async with asyncio.timeout(10):
            await serve_task
        for lg in watched_loggers:
            lg.removeHandler(collector)

    # Shutdown itself logged no errors.
    assert collector.server_errors == [], collector.server_errors
