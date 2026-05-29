"""Interactive inference server for Rehearse — full-duplex Moshi audio on Modal GPU.

Serves a WebSocket endpoint that accepts PCM16 16kHz audio from a caller and
streams back coach audio + transcript/prosody events in real time.

Deploy:
    modal deploy infra/interactive.py

The deployed WebSocket URL is:
    wss://<workspace>--rehearse-interactive-serve.modal.run/ws

Set in .env:
    INTERACTIVE_MODAL_ENDPOINT=wss://<workspace>--rehearse-interactive-serve.modal.run/ws

Wire protocol:
  Client → server: raw PCM16 16kHz bytes (audio) or UTF-8 JSON (control)
  Server → client: raw PCM16 16kHz bytes (audio) or UTF-8 JSON (events)

Cost: ~$0 when idle (scaledown_window=5min). Spins up in ~60s from HF cache.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import queue
import sys
import threading
import time
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
        "fastapi>=0.115",
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
    # Mount rehearse source so loader.py is importable inside the container.
    # add_local_dir replaces the deprecated modal.Mount API (Modal >= 1.0).
    .add_local_dir(
        str(Path(__file__).parents[1] / "rehearse"),
        remote_path="/app/rehearse",
    )
)

# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = modal.App("rehearse-interactive")

MINUTES = 60

# ---------------------------------------------------------------------------
# Inference server class
# ---------------------------------------------------------------------------

@app.cls(
    image=interactive_image,
    gpu="A10G",
    # Keep container alive 5 minutes after last request — Moshi cold start is ~60s.
    scaledown_window=5 * MINUTES,
    timeout=30 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
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

    @modal.asgi_app()
    def serve(self):
        from fastapi import FastAPI, Response, WebSocket

        web = FastAPI()

        @web.get("/health")
        async def health() -> Response:
            """Lightweight wake-up target. Hitting this starts the container cold-start
            clock without opening a full inference session."""
            return Response(status_code=200)

        @web.websocket("/ws")
        async def ws_endpoint(ws: WebSocket) -> None:
            await self._handle_session(ws)

        return web

    async def _handle_session(self, ws) -> None:
        await ws.accept()

        # Handshake: {"type": "start", "session_id": "..."}
        try:
            raw = await asyncio.wait_for(ws.receive_text(), timeout=10.0)
        except asyncio.TimeoutError:
            await ws.close(code=1008, reason="handshake timeout")
            return
        handshake = json.loads(raw)
        session_id = handshake.get("session_id", str(uuid.uuid4()))

        audio_q: queue.Queue[bytes | object] = queue.Queue()
        stop = threading.Event()
        loop = asyncio.get_running_loop()
        _STOP = object()

        async def _send_bytes(data: bytes) -> None:
            try:
                await ws.send_bytes(data)
            except Exception:
                pass

        async def _send_json(data: dict) -> None:
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                pass

        def _pub_bytes(data: bytes) -> None:
            asyncio.run_coroutine_threadsafe(_send_bytes(data), loop)

        def _pub_json(data: dict) -> None:
            asyncio.run_coroutine_threadsafe(_send_json(data), loop)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="modal-infer")
        infer_future = loop.run_in_executor(
            executor,
            lambda: _inference_loop(
                session_id, audio_q, stop, _STOP,
                self._mimi, self._lm_gen, self._tokenizer,
                _pub_bytes, _pub_json,
            ),
        )

        try:
            while True:
                msg = await ws.receive()
                mtype = msg.get("type")
                if mtype == "websocket.disconnect":
                    break
                if mtype == "websocket.receive":
                    if msg.get("bytes"):
                        audio_q.put_nowait(msg["bytes"])
                    elif msg.get("text"):
                        ctrl = json.loads(msg["text"])
                        if ctrl.get("type") == "inject":
                            # inject_speech: not supported by Moshi, logged server-side
                            pass
        except Exception:
            pass
        finally:
            stop.set()
            audio_q.put_nowait(_STOP)

        await infer_future
        executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Inference loop (runs in ThreadPoolExecutor on the GPU container)
# ---------------------------------------------------------------------------

def _inference_loop(
    session_id: str,
    audio_q: "queue.Queue[bytes | object]",
    stop: threading.Event,
    sentinel: object,
    mimi,
    lm_gen,
    tokenizer,
    pub_bytes,  # callable(bytes) → None
    pub_json,   # callable(dict) → None
) -> None:
    import numpy as np
    import torch
    import torchaudio.functional as F

    _MOSHI_SR = 24_000
    _REHEARSE_SR = 16_000
    _SILENCE_FLUSH = 10

    pcm_buf = np.array([], dtype=np.float32)
    coach_utt_id = str(uuid.uuid4())
    coach_pieces: list[str] = []
    silence_streak = 0
    frame_size: int = mimi.frame_size

    with mimi.streaming(1), lm_gen.streaming(1), torch.no_grad():
        while not stop.is_set():
            try:
                raw = audio_q.get(timeout=0.05)
            except queue.Empty:
                continue
            if raw is sentinel:
                break
            assert isinstance(raw, bytes)

            pcm16 = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            pcm_float = pcm16 / 32768.0
            t = torch.from_numpy(pcm_float).float().unsqueeze(0)
            pcm_24k = F.resample(t, _REHEARSE_SR, _MOSHI_SR).squeeze(0).numpy()
            pcm_buf = np.concatenate([pcm_buf, pcm_24k])

            while len(pcm_buf) >= frame_size:
                frame = pcm_buf[:frame_size]
                pcm_buf = pcm_buf[frame_size:]

                frame_t = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0).cuda()
                codes = mimi.encode(frame_t)
                for c in range(codes.shape[-1]):
                    tokens = lm_gen.step(codes[:, :, c : c + 1])
                    if tokens is None:
                        continue

                    text_id = int(tokens[0, 0, 0].item())
                    pad_id = lm_gen.lm_model.text_padding_token_id
                    if text_id != pad_id:
                        piece = tokenizer.id_to_piece(text_id)
                        coach_pieces.append(piece)
                        pub_json({
                            "type": "transcript",
                            "utterance_id": coach_utt_id,
                            "speaker": "coach",
                            "text": piece,
                            "is_final": False,
                        })
                        silence_streak = 0
                    else:
                        silence_streak += 1

                    wav = mimi.decode(tokens[:, 1:, :])
                    audio = wav.squeeze(0).squeeze(0)
                    audio_16k = F.resample(audio.unsqueeze(0), _MOSHI_SR, _REHEARSE_SR)
                    audio_16k = audio_16k.squeeze(0).clamp(-1.0, 1.0)
                    pcm16_out = (audio_16k * 32767).to(torch.int16).cpu().numpy().tobytes()
                    pub_bytes(pcm16_out)

                    if silence_streak >= _SILENCE_FLUSH and coach_pieces:
                        text = "".join(coach_pieces).replace("▁", " ").strip()
                        pub_json({
                            "type": "transcript",
                            "utterance_id": coach_utt_id,
                            "speaker": "coach",
                            "text": text,
                            "is_final": True,
                        })
                        coach_pieces = []
                        coach_utt_id = str(uuid.uuid4())
                        silence_streak = 0

    if coach_pieces:
        text = "".join(coach_pieces).replace("▁", " ").strip()
        pub_json({
            "type": "transcript",
            "utterance_id": coach_utt_id,
            "speaker": "coach",
            "text": text,
            "is_final": True,
        })

    pub_json({"type": "end_of_call", "reason": "hangup"})


# ---------------------------------------------------------------------------
# Smoke test (modal run infra/interactive.py)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
async def smoke_test() -> None:
    """Send 5s of silence to the deployed server and verify audio frames come back."""
    import websockets as _ws
    import numpy as np

    server = InteractiveServer()
    url = await server.serve.get_url.aio()
    ws_url = url.replace("https://", "wss://").replace("http://", "ws://") + "ws"
    print(f"Connecting to {ws_url}")

    async with _ws.connect(ws_url) as ws:
        await ws.send(json.dumps({"type": "start", "session_id": "smoke-test"}))

        silence = (np.zeros(320, dtype=np.int16)).tobytes()
        audio_frames = 0
        deadline = time.time() + 30.0

        send_task = asyncio.create_task(_send_silence(ws, silence))
        async for msg in ws:
            if isinstance(msg, bytes):
                audio_frames += 1
                if audio_frames >= 3:
                    break
            if time.time() > deadline:
                break
        send_task.cancel()

    assert audio_frames >= 1, "No audio frames received from server"
    print(f"Smoke test passed — received {audio_frames} audio frame(s)")


async def _send_silence(ws, chunk: bytes) -> None:
    while True:
        await ws.send(chunk)
        await asyncio.sleep(0.02)
