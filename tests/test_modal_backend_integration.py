"""Integration test: Twilio WebSocket → ModalInteractiveBackend → audio back.

No FakeBackend. ModalInteractiveBackend makes a real WebSocket connection to a
local server that simulates the Modal inference container. This exercises the
full wiring of the telephony stack with the interactive backend.
"""
from __future__ import annotations

import asyncio
import base64
import json
import socket
import struct
import threading
from datetime import UTC, datetime

import pytest
import websockets.asyncio.server
from fastapi.testclient import TestClient

from rehearse.api.app import create_app
from rehearse.audio.mulaw import encode_pcm16
from rehearse.backends.interactive.modal_backend import ModalInteractiveBackend
from rehearse.config import RuntimeConfig
from rehearse.types import Session


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_fake_modal_server(
    port: int,
    ready: threading.Event,
    stop: threading.Event,
) -> None:
    """Run a local WebSocket server that behaves like the Modal inference container.

    On connect:
      - Waits for {"type": "start"} handshake
      - Sends back 3 PCM16 audio frames so the caller receives audio
      - Sends a coach transcript event
      - Stays open until stop is set or the client disconnects
    """

    async def _handler(ws) -> None:
        raw = await ws.recv()
        data = json.loads(raw)
        assert data.get("type") == "start", f"unexpected handshake: {data}"

        coach_pcm = struct.pack("<16h", *([800] * 16))
        for _ in range(3):
            await ws.send(coach_pcm)
            await asyncio.sleep(0.01)

        await ws.send(json.dumps({
            "type": "transcript",
            "utterance_id": "coach-1",
            "speaker": "coach",
            "text": "Hello.",
            "is_final": True,
        }))

        while not stop.is_set():
            try:
                await asyncio.wait_for(ws.recv(), timeout=0.05)
            except (asyncio.TimeoutError, Exception):
                break

    async def _serve() -> None:
        async with websockets.asyncio.server.serve(_handler, "127.0.0.1", port):
            ready.set()
            while not stop.is_set():
                await asyncio.sleep(0.05)

    asyncio.run(_serve())


def test_twilio_to_modal_backend_audio_roundtrip(tmp_path, monkeypatch):
    """Twilio sends audio → ModalInteractiveBackend forwards it to local Modal server
    → local server sends coach audio back → audio arrives at the Twilio stream."""

    port = _free_port()
    ready = threading.Event()
    stop = threading.Event()

    server_thread = threading.Thread(
        target=_run_fake_modal_server,
        args=(port, ready, stop),
        daemon=True,
    )
    server_thread.start()
    assert ready.wait(timeout=5.0), "fake Modal server did not start in time"

    # Skip retry sleep so test doesn't stall if there's a transient connect issue
    monkeypatch.setattr(ModalInteractiveBackend, "_CONNECT_RETRY_DELAY", 0.0)

    try:
        config = RuntimeConfig(
            twilio_account_sid="AC_test",
            twilio_auth_token="test_token",
            twilio_from_number="+15555550100",
            public_base_url="https://example.test",
            hume_api_key="x",
            hume_config_id="x",
            session_root=tmp_path,
            validate_twilio_signature=False,
            backend_type="interactive",
            interactive_modal_endpoint=f"ws://127.0.0.1:{port}/ws",
        )

        session_id = "modal-integration-test"
        session_dir = tmp_path / session_id
        session_dir.mkdir()
        session = Session(id=session_id, created_at=datetime.now(UTC))
        (session_dir / "session.json").write_text(session.model_dump_json(indent=2))

        app = create_app(config)
        client = TestClient(app)

        caller_pcm8k = struct.pack("<4h", 0, 1000, -1000, 0)
        payload = base64.b64encode(encode_pcm16(caller_pcm8k)).decode("ascii")

        audio_frames: list[dict] = []
        with client.websocket_connect(f"/media/{session_id}") as ws:
            ws.send_json({"event": "connected"})
            ws.send_json({
                "event": "start",
                "start": {
                    "streamSid": "MZ_modal_test",
                    "callSid": "CA_modal_test",
                    "customParameters": {"session_id": session_id},
                },
            })
            ws.send_json({"event": "media", "media": {"payload": payload}})

            frame = ws.receive_json()
            if frame.get("event") == "media":
                audio_frames.append(frame)

            ws.send_json({"event": "stop"})

        assert audio_frames, "no audio frames received back through the Twilio stream"
        assert audio_frames[0]["streamSid"] == "MZ_modal_test"
        assert audio_frames[0]["event"] == "media"
    finally:
        stop.set()
        server_thread.join(timeout=3.0)
