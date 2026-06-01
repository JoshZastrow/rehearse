"""Interactive inference server for Rehearse — full-duplex Moshi audio on Modal GPU.

Serves a WebSocket endpoint that accepts PCM16 16kHz audio from a caller and
streams back coach audio + transcript/prosody events in real time.

Deploy:
    modal deploy infra/interactive.py

The deployed WebSocket URL is:
    wss://<workspace>--rehearse-interactive-interactiveserver-serve.modal.run/ws

Set in .env:
    INTERACTIVE_MODAL_ENDPOINT=wss://<workspace>--rehearse-interactive-interactiveserver-serve.modal.run/ws

Wire protocol:
  Client → server: {"type": "start", "session_id": "..."} (text) then raw PCM16 16kHz bytes
  Server → client: raw PCM16 16kHz bytes (audio) or UTF-8 JSON (events)

Architecture: aiohttp web server on port 8998, proxied by @modal.web_server.
This avoids Modal's ASGI/protobuf serialization layer entirely, matching
Moshi's own server.py architecture.

Cost: ~$0 when idle (scaledown_window=3min). Spins up in ~60s from HF cache.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import uuid
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------

interactive_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "numpy>=1.26",
        "faster-whisper>=1.1.0",
        "sentencepiece>=0.2.0",
        "aiohttp>=3.9",
        "pydantic>=2.9",
        "structlog>=24.4",
        "huggingface_hub>=0.20",
        "safetensors>=0.4",
        "sphn>=0.1.12",
    )
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends git",
        "pip install 'moshi @ git+https://github.com/kyutai-labs/moshi.git#subdirectory=moshi'",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_dir(
        str(Path(__file__).parents[1] / "rehearse"),
        remote_path="/app/rehearse",
    )
)

# ---------------------------------------------------------------------------
# Volumes / App
# ---------------------------------------------------------------------------

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
sessions_vol = modal.Volume.from_name("rehearse-sessions", create_if_missing=True)
app = modal.App("rehearse-interactive")

_SESSIONS_MOUNT = "/mnt/sessions"

MINUTES = 60
_AIOHTTP_PORT = 8998

# ---------------------------------------------------------------------------
# Inference server class
# ---------------------------------------------------------------------------


@app.cls(
    image=interactive_image,
    gpu="A10G",
    scaledown_window=3 * MINUTES,
    timeout=30 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache_vol, _SESSIONS_MOUNT: sessions_vol},
)
class InteractiveServer:

    @modal.enter()
    def load(self) -> None:
        sys.path.insert(0, "/app")
        from rehearse.backends.interactive.loader import load_models  # type: ignore[import]

        repo = os.environ.get("INTERACTIVE_MODEL_REPO", "kyutai/moshiko-pytorch-bf16")
        self._mimi, self._lm_gen, self._tokenizer = load_models(
            checkpoint_path="",
            hf_repo=repo,
            device="cuda",
        )

        # Initialize streaming state once; reset per connection (Moshi's approach)
        self._mimi.streaming_forever(1)
        self._lm_gen.streaming_forever(1)
        print("[load] warming up CUDA kernels...", file=sys.stderr, flush=True)
        self._warmup()
        print("[load] warmup done, starting aiohttp server", file=sys.stderr, flush=True)

        # Start aiohttp in a daemon thread so load() returns immediately
        t = threading.Thread(
            target=lambda: asyncio.run(self._run_aiohttp()),
            daemon=True,
        )
        t.start()

    def _warmup(self) -> None:
        import torch

        frame_size = self._mimi.frame_size
        with torch.no_grad():
            for _ in range(4):
                chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device="cuda")
                codes = self._mimi.encode(chunk)
                for c in range(codes.shape[-1]):
                    tokens = self._lm_gen.step(codes[:, :, c : c + 1])
                    if tokens is None:
                        continue
                    _ = self._mimi.decode(tokens[:, 1:])
        torch.cuda.synchronize()

    async def _run_aiohttp(self) -> None:
        from aiohttp import web

        # Lock must be created in the event loop where it will be used
        self._session_lock = asyncio.Lock()

        async def health(request: web.Request) -> web.Response:
            return web.Response(status=200)

        async def ws_handler(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await self._handle_session(ws)
            return ws

        _app = web.Application()
        _app.router.add_get("/health", health)
        _app.router.add_get("/ws", ws_handler)

        runner = web.AppRunner(_app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", _AIOHTTP_PORT)
        await site.start()
        print(f"[aiohttp] listening on 0.0.0.0:{_AIOHTTP_PORT}", file=sys.stderr, flush=True)
        await asyncio.Event().wait()  # run forever

    def _log(self, msg: str) -> None:
        import time as _time
        ts = _time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", file=sys.stderr, flush=True)

    async def _handle_session(self, ws) -> None:
        import aiohttp
        import time as _time
        import numpy as np
        import torch
        import torchaudio.functional as F

        _MOSHI_SR = 24_000
        _REHEARSE_SR = 16_000
        _SILENCE_FLUSH = 10

        self._log("ws: connection received, awaiting handshake")

        # First message must be the JSON handshake
        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=10.0)
        except asyncio.TimeoutError:
            self._log("ws: handshake timeout — closing")
            await ws.close()
            return
        self._log(f"ws: first message type={msg.type!r}")
        if msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
            # Connection closed immediately (e.g. proxy health probe) — not an error
            self._log(f"ws: connection closed before handshake ({msg.type})")
            return
        if msg.type != aiohttp.WSMsgType.TEXT:
            self._log(f"ws: expected TEXT handshake, got {msg.type!r} data={msg.data!r:.100}")
            await ws.close()
            return
        try:
            handshake = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self._log(f"ws: bad handshake JSON: {e}")
            await ws.close()
            return

        session_id = handshake.get("session_id", str(uuid.uuid4()))
        self._log(f"ws: session={session_id} handshake ok, waiting for lock")

        lock_wait_start = _time.monotonic()
        async with self._session_lock:
            lock_wait = _time.monotonic() - lock_wait_start
            self._log(f"ws: session={session_id} lock acquired (waited {lock_wait:.2f}s), starting inference")

            self._mimi.reset_streaming()
            self._lm_gen.reset_streaming()

            frame_size = self._mimi.frame_size
            pcm_buf = np.array([], dtype=np.float32)
            coach_utt_id = str(uuid.uuid4())
            coach_pieces: list[str] = []
            silence_streak = 0
            skip_frames = 1
            frames_rx = 0
            frames_tx = 0
            mimi_frames = 0
            t_start = _time.monotonic()
            caller_buf = bytearray()
            provider_buf = bytearray()
            token_rows: list[str] = []

            async for msg in ws:
                if msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    self._log(f"ws: session={session_id} connection closed (type={msg.type})")
                    break
                if msg.type != aiohttp.WSMsgType.BINARY:
                    self._log(f"ws: session={session_id} unexpected msg type={msg.type}")
                    continue

                frames_rx += 1
                if frames_rx % 50 == 1:
                    elapsed = _time.monotonic() - t_start
                    self._log(
                        f"ws: session={session_id} rx={frames_rx} tx={frames_tx} "
                        f"mimi_frames={mimi_frames} elapsed={elapsed:.1f}s"
                    )

                raw: bytes = msg.data
                caller_buf += raw
                pcm16 = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                t_in = torch.from_numpy(pcm16 / 32768.0).float().unsqueeze(0)
                pcm_24k = F.resample(t_in, _REHEARSE_SR, _MOSHI_SR).squeeze(0).numpy()
                pcm_buf = np.concatenate([pcm_buf, pcm_24k])

                while len(pcm_buf) >= frame_size:
                    frame = pcm_buf[:frame_size]
                    pcm_buf = pcm_buf[frame_size:]
                    mimi_frames += 1

                    frame_t = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0).cuda()
                    with torch.no_grad():
                        codes = self._mimi.encode(frame_t)

                        if skip_frames:
                            # First encoded frame has left-padding quirk; reset encoder
                            # so next call reapplies the padding (matches Moshi server.py)
                            self._mimi.reset_streaming()
                            skip_frames -= 1

                        for c in range(codes.shape[-1]):
                            tokens = self._lm_gen.step(codes[:, :, c : c + 1])
                            if tokens is None:
                                continue

                            text_id = int(tokens[0, 0, 0].item())
                            pad_id = self._lm_gen.lm_model.text_padding_token_id
                            token_piece: str | None = None
                            if text_id != pad_id:
                                token_piece = self._tokenizer.id_to_piece(text_id)
                                coach_pieces.append(token_piece)
                                self._log(f"ws: session={session_id} token={token_piece!r}")
                                silence_streak = 0
                            else:
                                silence_streak += 1

                            t_ms = (_time.monotonic() - t_start) * 1000.0
                            token_rows.append(json.dumps({
                                "frame_idx": mimi_frames,
                                "t_ms": round(t_ms, 1),
                                "text_token_id": text_id,
                                "text_piece": token_piece or "",
                                "is_padding": text_id == pad_id,
                            }))

                            wav = self._mimi.decode(tokens[:, 1:, :])
                            audio_16k = F.resample(
                                wav.squeeze(0).squeeze(0).unsqueeze(0),
                                _MOSHI_SR,
                                _REHEARSE_SR,
                            ).squeeze(0).clamp(-1.0, 1.0)
                            pcm16_out = (audio_16k * 32767).to(torch.int16).cpu().numpy().tobytes()
                            provider_buf += pcm16_out
                            try:
                                await ws.send_bytes(pcm16_out)
                                if token_piece is not None:
                                    await ws.send_str(json.dumps({
                                        "type": "transcript",
                                        "utterance_id": coach_utt_id,
                                        "speaker": "coach",
                                        "text": token_piece,
                                        "is_final": False,
                                    }))
                            except Exception as _send_err:
                                self._log(f"ws: session={session_id} send failed: {_send_err} — client disconnected")
                                return
                            frames_tx += 1

                            if silence_streak >= _SILENCE_FLUSH and coach_pieces:
                                text = "".join(coach_pieces).replace("▁", " ").strip()
                                try:
                                    await ws.send_str(json.dumps({
                                        "type": "transcript",
                                        "utterance_id": coach_utt_id,
                                        "speaker": "coach",
                                        "text": text,
                                        "is_final": True,
                                    }))
                                except Exception:
                                    return
                                self._log(f"ws: session={session_id} transcript[final]={text!r}")
                                coach_pieces = []
                                coach_utt_id = str(uuid.uuid4())
                                silence_streak = 0

            elapsed = _time.monotonic() - t_start
            self._log(
                f"ws: session={session_id} loop ended "
                f"rx={frames_rx} tx={frames_tx} mimi_frames={mimi_frames} elapsed={elapsed:.1f}s"
            )

            if coach_pieces:
                text = "".join(coach_pieces).replace("▁", " ").strip()
                await ws.send_str(json.dumps({
                    "type": "transcript",
                    "utterance_id": coach_utt_id,
                    "speaker": "coach",
                    "text": text,
                    "is_final": True,
                }))
                self._log(f"ws: session={session_id} transcript[final]={text!r}")

            # Persist session artifacts to Modal Volume
            session_dir = Path(_SESSIONS_MOUNT) / session_id
            try:
                session_dir.mkdir(parents=True, exist_ok=True)
                (session_dir / "caller_stream.pcm").write_bytes(bytes(caller_buf))
                (session_dir / "provider_stream.pcm").write_bytes(bytes(provider_buf))
                (session_dir / "tokens.jsonl").write_text("\n".join(token_rows))
                _write_mask(session_dir / "mask.jsonl", token_rows)
                sessions_vol.commit()
                transcribe_caller_audio.spawn(session_id)
                artifacts = ["caller_stream.pcm", "provider_stream.pcm", "tokens.jsonl", "mask.jsonl"]
                self._log(
                    f"ws: session={session_id} stored "
                    f"{len(caller_buf)} caller bytes, {len(provider_buf)} provider bytes"
                )
                await ws.send_str(json.dumps({
                    "type": "session_stored",
                    "session_id": session_id,
                    "volume_path": str(session_dir),
                    "artifacts": artifacts,
                }))
            except Exception as _store_err:
                self._log(f"ws: session={session_id} storage failed: {_store_err}")

            try:
                await ws.send_str(json.dumps({"type": "end_of_call", "reason": "hangup"}))
            except Exception:
                pass
            self._log(f"ws: session={session_id} done")

    @modal.web_server(_AIOHTTP_PORT)
    def serve(self):
        pass  # server started in load()


@app.function(
    image=interactive_image,
    volumes={_SESSIONS_MOUNT: sessions_vol},
    timeout=10 * MINUTES,
)
def transcribe_caller_audio(session_id: str) -> None:
    """Run faster-whisper on caller_stream.pcm and write caller_transcript.jsonl."""
    import json as _json
    import numpy as _np
    from faster_whisper import WhisperModel

    session_dir = Path(_SESSIONS_MOUNT) / session_id
    pcm_path = session_dir / "caller_stream.pcm"
    if not pcm_path.exists():
        print(f"[transcribe] caller_stream.pcm missing for {session_id}", flush=True)
        return

    raw = pcm_path.read_bytes()
    audio = _np.frombuffer(raw, dtype=_np.int16).astype(_np.float32) / 32768.0

    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _info = model.transcribe(audio, language="en", vad_filter=True)

    rows = []
    for seg in segments:
        rows.append(_json.dumps({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
        }))

    (session_dir / "caller_transcript.jsonl").write_text("\n".join(rows))
    sessions_vol.commit()
    print(f"[transcribe] wrote {len(rows)} segments for {session_id}", flush=True)


def _write_mask(path: Path, token_rows: list[str]) -> None:
    """Write mask.jsonl — one row per lm_gen.step() with speaker label.

    Frames with a non-padding text token are labelled 'provider' (the coach
    model is actively generating). All other frames are labelled 'caller'.
    Training code zeros loss on 'caller' frames so gradients flow only
    through provider outputs.
    """
    lines = []
    for row_str in token_rows:
        row = json.loads(row_str)
        speaker = "caller" if row["is_padding"] else "provider"
        lines.append(json.dumps({
            "frame_idx": row["frame_idx"],
            "t_ms": row["t_ms"],
            "speaker": speaker,
        }))
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Smoke test (modal run infra/interactive.py)
# ---------------------------------------------------------------------------


@app.local_entrypoint()
async def smoke_test() -> None:
    """Send 5s of silence to the deployed server and verify audio frames come back."""
    import struct
    import time

    import websockets as _ws

    server = InteractiveServer()
    url = await server.serve.get_url.aio()
    ws_url = url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/") + "/ws"
    print(f"Connecting to {ws_url}")

    async with _ws.connect(ws_url, open_timeout=60.0) as ws:
        await ws.send(json.dumps({"type": "start", "session_id": "smoke-test"}))

        silence = struct.pack("<320h", *([0] * 320))
        audio_frames = 0
        deadline = time.time() + 30.0

        async def _send_silence() -> None:
            while time.time() < deadline:
                await ws.send(silence)
                await asyncio.sleep(0.02)

        send_task = asyncio.create_task(_send_silence())
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    audio_frames += 1
                    if audio_frames >= 3:
                        break
                if time.time() > deadline:
                    break
        finally:
            send_task.cancel()

    assert audio_frames >= 1, "No audio frames received from server"
    print(f"Smoke test passed — received {audio_frames} audio frame(s)")
