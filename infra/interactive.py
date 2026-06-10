"""Interactive inference server for Rehearse — full-duplex speech models on Modal GPU.

Serves a WebSocket endpoint that accepts PCM16 16kHz audio from a caller and
streams back coach audio + transcript/prosody events in real time. Two model
types share the wire protocol:

  personaplex  nvidia/personaplex-7b-v1 — Moshi finetune with voice + role
               conditioning (PersonaplexProvider / PersonaplexCaller). Default.
  moshi        kyutai/moshiko-pytorch-bf16 (ProviderServer / CallerServer)

Deploy:
    modal deploy infra/interactive.py

The deployed WebSocket URLs are:
    wss://<workspace>--rehearse-interactive-providerserver-serve.modal.run/ws
    wss://<workspace>--rehearse-interactive-callerserver-serve.modal.run/ws
    wss://<workspace>--rehearse-interactive-personaplexprovider-serve.modal.run/ws
    wss://<workspace>--rehearse-interactive-personaplexcaller-serve.modal.run/ws

Set in .env:
    INTERACTIVE_PROVIDER_ENDPOINT=wss://<workspace>--rehearse-interactive-providerserver-serve.modal.run/ws
    INTERACTIVE_CALLER_ENDPOINT=wss://<workspace>--rehearse-interactive-callerserver-serve.modal.run/ws
(point the endpoint vars at the personaplex URLs to use PersonaPlex instead)

Wire protocol:
  Client → server: {"type": "start", "session_id": "..."} (text) then raw PCM16 16kHz bytes
  Server → client: raw PCM16 16kHz bytes (audio) or UTF-8 JSON (events)

Architecture: aiohttp web server on port 8998, proxied by @modal.web_server.
This avoids Modal's ASGI/protobuf serialization layer entirely, matching
Moshi's own server.py architecture.

Cost: ~$0 when idle (scaledown_window=60min). Spins up in ~60s from HF cache.
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

# PersonaPlex needs NVIDIA's moshi fork (same `moshi` module name, so it gets
# its own image). Deps are pinned to the fork's requirements — notably
# torch<2.5 — and the fork itself installs with --no-deps so pip can't
# downgrade/upgrade anything underneath us.
personaplex_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "libopus-dev", "libportaudio2")
    .uv_pip_install(
        "torch==2.4.1",
        "torchaudio==2.4.1",
        "numpy>=1.26,<2.2",
        "faster-whisper>=1.1.0",
        "sentencepiece>=0.2.0",
        "aiohttp>=3.9",
        "pydantic>=2.9",
        "structlog>=24.4",
        "huggingface_hub>=0.24",
        "safetensors>=0.4,<0.5",
        "sphn>=0.1.4,<0.2",
        "einops==0.7",
        "sounddevice==0.5",
        "tqdm>=4.48",
    )
    .run_commands(
        "pip install --no-deps 'moshi-personaplex @ "
        "git+https://github.com/NVIDIA/personaplex.git#subdirectory=moshi'",
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
# PersonaPlex persona defaults
# ---------------------------------------------------------------------------

_PERSONAPLEX_TEXT_PROMPTS = {
    "provider": (
        "You are a supportive conversation coach. Help the caller rehearse "
        "difficult conversations: ask what they want to practice, play your "
        "part, and give clear, encouraging feedback."
    ),
    "caller": "You enjoy having a good conversation.",
}
_PERSONAPLEX_VOICE_PROMPTS = {"provider": "NATF0.pt", "caller": "NATM0.pt"}


def _resolve_persona(handshake: dict, speaker_role: str) -> tuple[str, str]:
    """Return (text_prompt, voice_prompt) — handshake > env > role default."""
    text = (
        handshake.get("text_prompt")
        or os.environ.get("PERSONAPLEX_TEXT_PROMPT")
        or _PERSONAPLEX_TEXT_PROMPTS[speaker_role]
    )
    voice = (
        handshake.get("voice_prompt")
        or os.environ.get("PERSONAPLEX_VOICE_PROMPT")
        or _PERSONAPLEX_VOICE_PROMPTS[speaker_role]
    )
    return text, voice


# ---------------------------------------------------------------------------
# Inference server base class
# ---------------------------------------------------------------------------


class _InteractiveServerBase:
    """Shared inference logic for provider and caller Modal servers."""

    checkpoint_path: str = ""
    """Empty = load from HuggingFace. Non-empty = load from Modal Volume path."""

    speaker_role: str = "provider"
    """'provider' or 'caller'. Controls transcript labels and ASR target."""

    model_type: str = "personaplex"
    """'personaplex' (default) or 'moshi'. Selects the model loader and persona conditioning."""

    @property
    def _other_role(self) -> str:
        return "caller" if self.speaker_role == "provider" else "provider"

    def _asr_push_received(self, asr, raw: bytes) -> None:
        """Run ASR on received audio only when we are the provider."""
        if self.speaker_role == "provider":
            asr.push_audio(raw)

    def _asr_push_generated(self, asr, pcm16_out: bytes) -> None:
        """Run ASR on generated audio only when we are the caller."""
        if self.speaker_role == "caller":
            asr.push_audio(pcm16_out)

    @modal.enter()
    def load(self) -> None:
        sys.path.insert(0, "/app")

        if self.model_type == "personaplex":
            from rehearse.backends.interactive.personaplex_loader import (  # type: ignore[import]
                DEFAULT_PERSONAPLEX_REPO,
                ensure_voice_prompt_dir,
                load_personaplex_models,
            )

            repo = os.environ.get("INTERACTIVE_MODEL_REPO", DEFAULT_PERSONAPLEX_REPO)
            self._mimi, self._lm_gen, self._tokenizer = load_personaplex_models(
                checkpoint_path=self.checkpoint_path,
                hf_repo=repo,
                device="cuda",
            )
            self._voice_prompt_dir = ensure_voice_prompt_dir(repo)
        else:
            from rehearse.backends.interactive.loader import load_models  # type: ignore[import]

            repo = os.environ.get("INTERACTIVE_MODEL_REPO", "kyutai/moshiko-pytorch-bf16")
            self._mimi, self._lm_gen, self._tokenizer = load_models(
                checkpoint_path=self.checkpoint_path,
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
                    # First 8 codebooks are the output audio stream. Equivalent
                    # to [:, 1:] for Moshi (dep_q=8); PersonaPlex (dep_q=16)
                    # also predicts input-stream codebooks mimi must not see.
                    _ = self._mimi.decode(tokens[:, 1:9])
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
        _FIRST_FRAME_TIMEOUT = 30.0  # seconds to wait for first audio frame
        _IDLE_TIMEOUT = 10.0         # seconds without audio after stream starts → end session

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

            if self.model_type == "personaplex":
                from rehearse.backends.interactive.personaplex_loader import (  # type: ignore[import]
                    apply_persona,
                    resolve_voice_prompt,
                )

                text_prompt, voice_name = _resolve_persona(handshake, self.speaker_role)
                apply_persona(
                    self._lm_gen,
                    self._tokenizer,
                    voice_prompt_path=resolve_voice_prompt(self._voice_prompt_dir, voice_name),
                    text_prompt=text_prompt,
                )
                self._log(f"ws: session={session_id} prefilling persona (voice={voice_name})")
                t_prefill = _time.monotonic()
                # mimi is reused to encode the voice prompt, then reset before
                # the conversation starts (matches PersonaPlex server.py).
                await self._lm_gen.step_system_prompts_async(self._mimi)
                self._mimi.reset_streaming()
                self._log(
                    f"ws: session={session_id} persona prefill done "
                    f"in {_time.monotonic() - t_prefill:.1f}s"
                )

            from rehearse.backends.interactive.asr import InteractiveASR  # type: ignore[import]

            frame_size = self._mimi.frame_size
            pcm_buf = np.array([], dtype=np.float32)
            coach_utt_id = str(uuid.uuid4())
            coach_pieces: list[str] = []
            silence_streak = 0
            leading_silence = 0      # silence tokens before first speech token this turn
            turn_has_speech = False  # whether current turn has emitted any text tokens
            # Moshi's server.py resets the encoder after the first frame to
            # reapply left-padding; PersonaPlex's does not (its mimi was just
            # reset after the persona prefill).
            skip_frames = 1 if self.model_type == "moshi" else 0
            asr = InteractiveASR(model_size=os.environ.get("INTERACTIVE_ASR_MODEL", "base"))
            frames_rx = 0
            frames_tx = 0
            mimi_frames = 0
            t_start = _time.monotonic()
            turn_num = 0
            t_turn_start = t_start
            t_first_audio_sent: float | None = None
            t_first_speech_token: float | None = None
            caller_buf = bytearray()
            provider_buf = bytearray()
            token_rows: list[str] = []

            while True:
                # Before first audio: give the client time to set up the bridge.
                # After audio starts: a 3s gap means the session is over.
                timeout = _IDLE_TIMEOUT if frames_rx > 0 else _FIRST_FRAME_TIMEOUT
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=timeout)
                except asyncio.TimeoutError:
                    self._log(f"ws: session={session_id} idle timeout ({_IDLE_TIMEOUT}s) — ending session")
                    break
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
                self._asr_push_received(asr, raw)
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
                                if token_piece == "<unk>":
                                    token_piece = None
                                else:
                                    coach_pieces.append(token_piece)
                                    self._log(f"ws: session={session_id} token={token_piece!r}")
                                    if not turn_has_speech:
                                        t_first_speech_token = _time.monotonic()
                                        turn_has_speech = True
                                    silence_streak = 0
                            if token_piece is None:
                                silence_streak += 1
                                if not turn_has_speech:
                                    leading_silence += 1

                            # First 8 codebooks are the output audio stream
                            # (PersonaPlex dep_q=16 carries 16; Moshi has 8).
                            wav = self._mimi.decode(tokens[:, 1:9, :])
                            audio_16k = F.resample(
                                wav.squeeze(0).squeeze(0).unsqueeze(0),
                                _MOSHI_SR,
                                _REHEARSE_SR,
                            ).squeeze(0).clamp(-1.0, 1.0)
                            pcm16_out = (audio_16k * 32767).to(torch.int16).cpu().numpy().tobytes()
                            self._asr_push_generated(asr, pcm16_out)
                            provider_buf += pcm16_out
                            t_ms = (_time.monotonic() - t_start) * 1000.0
                            token_rows.append(json.dumps({
                                "frame_idx": mimi_frames,
                                "t_ms": round(t_ms, 1),
                                "text_token_id": text_id,
                                "text_piece": token_piece or "",
                                "is_padding": text_id == pad_id,
                            }))
                            try:
                                await ws.send_bytes(pcm16_out)
                                if t_first_audio_sent is None:
                                    t_first_audio_sent = _time.monotonic()
                                if token_piece is not None:
                                    await ws.send_str(json.dumps({
                                        "type": "transcript",
                                        "utterance_id": coach_utt_id,
                                        "speaker": self.speaker_role,
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
                                        "speaker": self.speaker_role,
                                        "text": text,
                                        "is_final": True,
                                    }))
                                except Exception:
                                    return
                                now = _time.monotonic()
                                profile = {
                                    "type": "profile",
                                    "turn": turn_num,
                                    "turn_total_ms": round((now - t_turn_start) * 1000),
                                    "leading_silence_ms": leading_silence * 80,
                                    "time_to_first_audio_ms": round((t_first_audio_sent - t_turn_start) * 1000) if t_first_audio_sent else None,
                                    "time_to_first_speech_ms": round((t_first_speech_token - t_turn_start) * 1000) if t_first_speech_token else None,
                                    "speech_tokens": len(coach_pieces),
                                    "text": text,
                                }
                                self._log(json.dumps(profile))
                                try:
                                    await ws.send_str(json.dumps(profile))
                                except Exception:
                                    return
                                coach_pieces = []
                                coach_utt_id = str(uuid.uuid4())
                                silence_streak = 0
                                leading_silence = 0
                                turn_has_speech = False
                                turn_num += 1
                                t_turn_start = now
                                t_first_audio_sent = None
                                t_first_speech_token = None
                                # Off the event loop: long turns make whisper
                                # transcription take >20s, which would block
                                # keepalive pongs and drop the connection.
                                user_text = await asyncio.to_thread(asr.transcribe_and_reset)
                                if user_text:
                                    user_utt_id = str(uuid.uuid4())
                                    try:
                                        await ws.send_str(json.dumps({
                                            "type": "transcript",
                                            "utterance_id": user_utt_id,
                                            "speaker": self._other_role,
                                            "text": user_text,
                                            "is_final": True,
                                        }))
                                        await ws.send_str(json.dumps({
                                            "type": "prosody",
                                            "utterance_id": user_utt_id,
                                            "speaker": self._other_role,
                                            "arousal": 0.0,
                                            "valence": 0.0,
                                        }))
                                    except Exception:
                                        return
                                    self._log(f"ws: session={session_id} user_transcript={user_text!r}")

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
                    "speaker": self.speaker_role,
                    "text": text,
                    "is_final": True,
                }))
                self._log(f"ws: session={session_id} transcript[final]={text!r}")

            user_text = await asyncio.to_thread(asr.transcribe_and_reset)
            if user_text:
                user_utt_id = str(uuid.uuid4())
                try:
                    await ws.send_str(json.dumps({
                        "type": "transcript",
                        "utterance_id": user_utt_id,
                        "speaker": self._other_role,
                        "text": user_text,
                        "is_final": True,
                    }))
                except Exception:
                    pass
                self._log(f"ws: session={session_id} user_transcript[final]={user_text!r}")

            # Persist session artifacts to Modal Volume
            session_dir = Path(_SESSIONS_MOUNT) / session_id
            try:
                session_dir.mkdir(parents=True, exist_ok=True)
                (session_dir / "caller_stream.pcm").write_bytes(bytes(caller_buf))
                (session_dir / "provider_stream.pcm").write_bytes(bytes(provider_buf))
                (session_dir / "tokens.jsonl").write_text("\n".join(token_rows))
                _write_mask(session_dir / "mask.jsonl", token_rows)
                sessions_vol.commit()
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


@app.cls(
    image=interactive_image,
    gpu="A10G",
    scaledown_window=60 * MINUTES,
    timeout=30 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache_vol, _SESSIONS_MOUNT: sessions_vol},
)
class ProviderServer(_InteractiveServerBase):
    """Moshi provider (coach) model endpoint. Loads from HuggingFace by default."""
    checkpoint_path = ""
    speaker_role = "provider"
    model_type = "moshi"  # this image carries upstream moshi, not the PersonaPlex fork

    @modal.web_server(_AIOHTTP_PORT)
    def serve(self):
        pass  # server started in load()


@app.cls(
    image=interactive_image,
    gpu="A10G",
    scaledown_window=60 * MINUTES,
    timeout=30 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache_vol, _SESSIONS_MOUNT: sessions_vol},
)
class CallerServer(_InteractiveServerBase):
    """Moshi caller model endpoint. Set checkpoint_path to a Volume path for fine-tuned weights."""
    checkpoint_path = ""      # e.g. "/mnt/training/runs/caller-v1"
    speaker_role = "caller"
    model_type = "moshi"  # this image carries upstream moshi, not the PersonaPlex fork

    @modal.web_server(_AIOHTTP_PORT)
    def serve(self):
        pass  # server started in load()


@app.cls(
    image=personaplex_image,
    gpu="L40S",
    scaledown_window=60 * MINUTES,
    timeout=30 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache_vol, _SESSIONS_MOUNT: sessions_vol},
    secrets=[modal.Secret.from_name("HF_TOKEN")],
)
class PersonaplexProvider(_InteractiveServerBase):
    """PersonaPlex provider (coach) endpoint — voice + role conditioned.

    nvidia/personaplex-7b-v1 is gated: the HF_TOKEN secret must belong to an
    account that accepted the license. L40S over A10G: 16.7 GB bf16 weights +
    KV + CUDA-graph pools crowd a 24 GB card, and the dep_q=16 depth
    transformer doubles per-frame depformer steps vs Moshi.
    """
    model_type = "personaplex"
    speaker_role = "provider"

    @modal.web_server(_AIOHTTP_PORT)
    def serve(self):
        pass  # server started in load()


@app.cls(
    image=personaplex_image,
    gpu="L40S",
    scaledown_window=60 * MINUTES,
    timeout=30 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache_vol, _SESSIONS_MOUNT: sessions_vol},
    secrets=[modal.Secret.from_name("HF_TOKEN")],
)
class PersonaplexCaller(_InteractiveServerBase):
    """PersonaPlex caller endpoint — voice + role conditioned synthetic caller."""
    model_type = "personaplex"
    speaker_role = "caller"

    @modal.web_server(_AIOHTTP_PORT)
    def serve(self):
        pass  # server started in load()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_mask(path: Path, token_rows: list[str]) -> None:
    """Write mask.jsonl — one row per model frame labelled caller or provider."""
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
async def smoke_test(endpoint: str = "") -> None:
    """Send 5s of silence to the deployed server and verify audio frames come back.

    Defaults to INTERACTIVE_PROVIDER_ENDPOINT from .env; pass --endpoint to
    target another deployed class, e.g.:
        modal run infra/interactive.py --endpoint \\
            wss://<workspace>--rehearse-interactive-personaplexprovider-serve.modal.run/ws
    """
    import struct
    import time
    from pathlib import Path as _Path

    from dotenv import load_dotenv as _load_dotenv
    import websockets as _ws

    if not endpoint:
        _load_dotenv(_Path(__file__).parent.parent / ".env", override=True)
        endpoint = os.environ.get("INTERACTIVE_PROVIDER_ENDPOINT", "")
    if not endpoint:
        raise RuntimeError("INTERACTIVE_PROVIDER_ENDPOINT not set — run `make deploy-interactive` and set the env var")
    ws_url = endpoint.replace("https://", "wss://").replace("http://", "ws://").rstrip("/")
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
