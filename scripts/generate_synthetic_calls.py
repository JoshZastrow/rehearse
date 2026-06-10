"""Generate synthetic call sessions between provider and caller Moshi endpoints.

Runs N sessions through two ModalInteractiveBackend instances connected via
ConversationBridge. For each session, audio from both sides is captured and
written as a 24kHz stereo WAV plus a word-alignment annotation file, producing
training-ready artifacts in sessions/<session_id>/.

Output per session:
    sessions/<session_id>/audio_stereo.wav   — 24kHz stereo (left=provider, right=caller)
    sessions/<session_id>/audio_stereo.json  — {"alignments": [[word,[t0,t1],speaker],...]}
    sessions/<session_id>/transcript.jsonl   — raw utterance log

Usage:
    set -a && source .env && set +a
    uv run python scripts/generate_synthetic_calls.py
    uv run python scripts/generate_synthetic_calls.py --count 10 --duration 60
    uv run python scripts/generate_synthetic_calls.py --sequential   # one at a time
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import math
import sys
import uuid
import wave
from pathlib import Path

import numpy as np

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generate_synthetic_calls")

_SRC_RATE = 16_000   # ModalInteractiveBackend output rate
_DST_RATE = 24_000   # Mimi / moshi-finetune expected rate
_SESSIONS_DIR = Path("sessions")


# ─── Audio helpers ────────────────────────────────────────────────────────────


def _resample_pcm16(raw: bytes, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample raw PCM16 bytes from src_rate to dst_rate using polyphase."""
    from math import gcd
    from scipy.signal import resample_poly

    g = gcd(dst_rate, src_rate)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if len(samples) == 0:
        return np.array([], dtype=np.int16)
    resampled = resample_poly(samples, dst_rate // g, src_rate // g)
    return np.clip(resampled, -32768, 32767).astype(np.int16)


def _write_stereo_wav(path: Path, left: np.ndarray, right: np.ndarray, rate: int) -> None:
    """Write a stereo PCM16 WAV from two int16 arrays (zero-padded to equal length)."""
    n = max(len(left), len(right), 1)
    if len(left) < n:
        left = np.pad(left, (0, n - len(left)))
    if len(right) < n:
        right = np.pad(right, (0, n - len(right)))
    stereo = np.stack([left, right], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(stereo.tobytes())


# ─── Alignment helpers ────────────────────────────────────────────────────────


def _build_alignments(utterances: list[dict], session_start: float) -> list:
    """Convert a list of final utterance dicts into word-level alignment entries.

    Each utterance: {"speaker": "coach"|"user", "text": "...", "ts_start": ..., "ts_end": ...}
    Returns: [["word", [rel_start, rel_end], "caller"|"provider"], ...]

    Words are distributed uniformly across the utterance time window.
    Speaker mapping: "coach" → "provider", "user" → "caller".
    """
    _SPEAKER_MAP = {"coach": "provider", "user": "caller", "provider": "provider", "caller": "caller"}
    alignments: list = []

    for utt in utterances:
        text = utt.get("text", "").strip()
        if not text:
            continue
        speaker = _SPEAKER_MAP.get(utt.get("speaker", ""), "caller")
        t0 = utt["ts_start"] - session_start
        t1 = utt["ts_end"] - session_start
        if t1 <= t0:
            t1 = t0 + 0.5

        words = text.split()
        if not words:
            continue
        step = (t1 - t0) / len(words)
        for i, word in enumerate(words):
            w_start = t0 + i * step
            w_end = t0 + (i + 1) * step
            alignments.append([word, [round(w_start, 3), round(w_end, 3)], speaker])

    return sorted(alignments, key=lambda a: a[1][0])


# ─── Session runner ───────────────────────────────────────────────────────────


async def run_one_session(
    *,
    session_id: str,
    provider_endpoint: str,
    caller_endpoint: str,
    max_duration_sec: float,
    sessions_dir: Path,
) -> dict:
    """Run one synthetic session and save artifacts to disk.

    Returns a result dict with session_id, duration_sec, end_reason, path.
    """
    import time
    from rehearse.backends.interactive.modal_backend import ModalInteractiveBackend
    from rehearse.backends.interactive.bridge import ConversationBridge
    from rehearse.bus import FrameBus
    from rehearse.frames import AudioChunk, EndOfCall, TranscriptDelta
    from rehearse.types import Speaker

    session_dir = sessions_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    provider_bus = FrameBus(session_id=session_id)
    caller_bus = FrameBus(session_id=session_id + "-caller")

    provider_backend = ModalInteractiveBackend(endpoint=provider_endpoint)
    caller_backend = ModalInteractiveBackend(endpoint=caller_endpoint)
    bridge = ConversationBridge(
        provider_backend=provider_backend,
        provider_bus=provider_bus,
        caller_backend=caller_backend,
        caller_bus=caller_bus,
    )

    provider_pcm: list[bytes] = []
    caller_pcm: list[bytes] = []
    utterances: list[dict] = []
    transcript_lines: list[str] = []

    session_start = time.time()
    end_reason = "timeout"

    async def collect_provider(bus: FrameBus) -> None:
        async for frame in bus.subscribe():
            if isinstance(frame, AudioChunk) and frame.speaker == Speaker.GUIDE:
                provider_pcm.append(frame.pcm16_16k)
            elif isinstance(frame, TranscriptDelta) and frame.is_final:
                row = {
                    "session_id": session_id,
                    "utterance_id": frame.utterance_id,
                    "ts_start": frame.ts_start,
                    "ts_end": frame.ts_end or frame.ts_start,
                    "speaker": frame.speaker.value,
                    "text": frame.text,
                    "is_interim": False,
                }
                utterances.append(row)
                transcript_lines.append(json.dumps(row))
            elif isinstance(frame, EndOfCall):
                return

    async def collect_caller(bus: FrameBus) -> None:
        async for frame in bus.subscribe():
            if isinstance(frame, AudioChunk) and frame.speaker == Speaker.GUIDE:
                caller_pcm.append(frame.pcm16_16k)
            elif isinstance(frame, EndOfCall):
                return

    await provider_backend.start(session_id, provider_bus)
    await caller_backend.start(session_id + "-caller", caller_bus)
    await bridge.start()

    collect_tasks = [
        asyncio.create_task(collect_provider(provider_bus), name="collect-provider"),
        asyncio.create_task(collect_caller(caller_bus), name="collect-caller"),
    ]

    # Send real-time silence to provider to bootstrap the conversation loop.
    # Without sustained input, neither server generates audio. We send 20ms frames
    # at real-time rate for BOOTSTRAP_SEC, then the audio loop between the two
    # servers sustains itself (each server's output feeds the other's input via bridge).
    _BOOTSTRAP_SEC = 5.0
    _FRAME = b"\x00" * 640  # 20ms @ 16kHz PCM16 (320 samples × 2 bytes)

    async def _send_bootstrap() -> None:
        n = int(_BOOTSTRAP_SEC / 0.02)
        for _ in range(n):
            await provider_backend.send_caller_audio(_FRAME)
            await asyncio.sleep(0.02)

    asyncio.create_task(_send_bootstrap(), name="bootstrap-silence")

    try:
        async with asyncio.timeout(max_duration_sec):
            async for frame in provider_bus.subscribe():
                if isinstance(frame, EndOfCall):
                    end_reason = frame.reason
                    break
    except asyncio.TimeoutError:
        pass
    finally:
        for t in collect_tasks:
            t.cancel()
        await asyncio.gather(*collect_tasks, return_exceptions=True)
        await bridge.close()
        await caller_backend.close()
        await provider_backend.close()
        await provider_bus.aclose()
        await caller_bus.aclose()

    duration = time.time() - session_start

    # Write transcript
    (session_dir / "transcript.jsonl").write_text("\n".join(transcript_lines) + "\n" if transcript_lines else "")

    # Build stereo WAV at 24kHz
    prov_bytes = b"".join(provider_pcm)
    call_bytes = b"".join(caller_pcm)
    left = _resample_pcm16(prov_bytes, _SRC_RATE, _DST_RATE)
    right = _resample_pcm16(call_bytes, _SRC_RATE, _DST_RATE)
    stereo_path = session_dir / "audio_stereo.wav"
    _write_stereo_wav(stereo_path, left, right, _DST_RATE)

    # Build word alignments from transcript
    alignments = _build_alignments(utterances, session_start)
    annotation = {"alignments": alignments}
    (session_dir / "audio_stereo.json").write_text(json.dumps(annotation, ensure_ascii=False))

    wav_samples = max(len(left), len(right))
    wav_duration = wav_samples / _DST_RATE

    log.info(
        "session %s done — reason=%s wall=%.1fs audio=%.1fs utterances=%d alignments=%d",
        session_id, end_reason, duration, wav_duration, len(utterances), len(alignments),
    )

    return {
        "session_id": session_id,
        "duration_sec": duration,
        "audio_duration_sec": wav_duration,
        "end_reason": end_reason,
        "utterances": len(utterances),
        "alignments": len(alignments),
        "path": str(stereo_path),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────


async def _run_sequential(
    session_ids: list[str],
    provider_endpoint: str,
    caller_endpoint: str,
    max_duration_sec: float,
    sessions_dir: Path,
) -> list[dict]:
    results = []
    for i, sid in enumerate(session_ids):
        log.info("[%d/%d] Starting session %s", i + 1, len(session_ids), sid)
        try:
            r = await run_one_session(
                session_id=sid,
                provider_endpoint=provider_endpoint,
                caller_endpoint=caller_endpoint,
                max_duration_sec=max_duration_sec,
                sessions_dir=sessions_dir,
            )
            results.append(r)
        except Exception as exc:
            log.error("Session %s failed: %s", sid, exc)
            results.append({"session_id": sid, "error": str(exc)})
    return results


async def _run_concurrent(
    session_ids: list[str],
    provider_endpoint: str,
    caller_endpoint: str,
    max_duration_sec: float,
    sessions_dir: Path,
    concurrency: int = 2,
) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = [{}] * len(session_ids)

    async def _one(idx: int, sid: str) -> None:
        async with sem:
            log.info("[%d/%d] Starting session %s", idx + 1, len(session_ids), sid)
            try:
                r = await run_one_session(
                    session_id=sid,
                    provider_endpoint=provider_endpoint,
                    caller_endpoint=caller_endpoint,
                    max_duration_sec=max_duration_sec,
                    sessions_dir=sessions_dir,
                )
                results[idx] = r
            except Exception as exc:
                log.error("Session %s failed: %s", sid, exc)
                results[idx] = {"session_id": sid, "error": str(exc)}

    await asyncio.gather(*[_one(i, sid) for i, sid in enumerate(session_ids)])
    return results


def _make_session_ids(count: int) -> list[str]:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return [f"syn-{stamp}-{uuid.uuid4().hex[:8]}" for _ in range(count)]


async def main() -> int:
    import os

    parser = argparse.ArgumentParser(description="Generate synthetic call sessions")
    parser.add_argument("--count", type=int, default=10, help="Number of sessions to generate")
    parser.add_argument("--duration", type=float, default=60.0, help="Max session duration (seconds)")
    parser.add_argument("--sequential", action="store_true", help="Run sessions one at a time")
    parser.add_argument("--concurrency", type=int, default=2, help="Parallel sessions (if not --sequential)")
    parser.add_argument("--sessions-dir", type=Path, default=_SESSIONS_DIR)
    parser.add_argument(
        "--provider-endpoint",
        default=os.environ.get("INTERACTIVE_PROVIDER_ENDPOINT", ""),
    )
    parser.add_argument(
        "--caller-endpoint",
        default=os.environ.get("INTERACTIVE_CALLER_ENDPOINT", ""),
    )
    args = parser.parse_args()

    if not args.provider_endpoint:
        log.error("INTERACTIVE_PROVIDER_ENDPOINT not set (use --provider-endpoint or set env var)")
        return 1
    if not args.caller_endpoint:
        log.error("INTERACTIVE_CALLER_ENDPOINT not set (use --caller-endpoint or set env var)")
        return 1

    session_ids = _make_session_ids(args.count)
    log.info(
        "Generating %d synthetic sessions (duration=%.0fs, sequential=%s)",
        args.count, args.duration, args.sequential,
    )

    if args.sequential:
        results = await _run_sequential(
            session_ids,
            args.provider_endpoint,
            args.caller_endpoint,
            args.duration,
            args.sessions_dir,
        )
    else:
        results = await _run_concurrent(
            session_ids,
            args.provider_endpoint,
            args.caller_endpoint,
            args.duration,
            args.sessions_dir,
            concurrency=args.concurrency,
        )

    ok = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    print(f"\n{'='*60}")
    print(f"Generated {len(ok)}/{len(results)} sessions successfully")
    print(f"{'='*60}")
    for r in results:
        if "error" in r:
            print(f"  FAILED  {r['session_id']}: {r['error']}")
        else:
            print(
                f"  OK      {r['session_id']}  "
                f"audio={r.get('audio_duration_sec', 0):.1f}s  "
                f"utterances={r.get('utterances', 0)}  "
                f"alignments={r.get('alignments', 0)}"
            )

    if ok:
        ids_out = args.sessions_dir / ".synthetic_session_ids.txt"
        ids_out.write_text("\n".join(r["session_id"] for r in ok) + "\n")
        print(f"\nSession IDs written to: {ids_out}")
        print("\nNext steps:")
        print("  # Build manifest and push to Modal Volume:")
        print(f"  uv run python train/pipeline/dataset.py sessions_root={args.sessions_dir} out=runs/sessions.jsonl require_annotation=true push_to_volume=true")
        print("  # Run training:")
        print("  uv run rehearse-train config=train/rehearse_smoke.yaml")

    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
