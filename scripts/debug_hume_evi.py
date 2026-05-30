"""Debug script: connect to Hume EVI, send TTS audio, print every event.

Run:
    uv run python scripts/debug_hume_evi.py

What it tests:
  1. Hume EVI WebSocket connection (auth + config_id)
  2. TTS audio synthesis (Hume Octave → PCM16 at 16kHz)
  3. Audio ingestion — does Hume transcribe the speech?
  4. CLM webhook — does Hume call back to your server for a coach response?
  5. Coach response — does an assistant_message event arrive?

Each event is printed with a timestamp so you can see the timing.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

HUME_API_KEY = os.environ.get("HUME_API_KEY", "")
if not HUME_API_KEY:
    print("ERROR: HUME_API_KEY not set", file=sys.stderr)
    sys.exit(1)

# Read the synced config ID (same logic as select_config_id)
MAPPING_PATH = Path("sessions/.hume_configs.json")
import json
if MAPPING_PATH.exists():
    mapping = json.loads(MAPPING_PATH.read_text())
    CONFIG_ID = mapping.get("default", os.environ.get("HUME_CONFIG_ID", ""))
else:
    CONFIG_ID = os.environ.get("HUME_CONFIG_ID", "")

if not CONFIG_ID:
    print("ERROR: no config_id — run 'make serve' once to sync Hume configs", file=sys.stderr)
    sys.exit(1)

print(f"Using config_id: {CONFIG_ID}")
print(f"API key: {HUME_API_KEY[:8]}...")


async def synthesize_greeting() -> bytes:
    """Generate a short greeting via Hume Octave TTS → PCM16 at 16kHz."""
    from rehearse.eval.environments.tts_bridge import HumeOctaveProvider
    tts = HumeOctaveProvider(api_key=HUME_API_KEY)
    text = "Hi, I need help preparing for a difficult conversation with my manager."
    print(f"Synthesizing TTS: {text!r}")
    t0 = time.monotonic()
    pcm = await tts.synthesize_pcm(text=text, description="anxious, hesitant")
    print(f"TTS done: {len(pcm)} bytes PCM16 16kHz in {time.monotonic()-t0:.1f}s")
    return pcm


async def run_debug():
    from hume.client import AsyncHumeClient

    audio_pcm = await synthesize_greeting()

    print(f"\nConnecting to Hume EVI (config_id={CONFIG_ID})...")
    client = AsyncHumeClient(api_key=HUME_API_KEY)
    session_id = f"debug-{int(time.time())}"

    import base64
    from hume.empathic_voice.types.audio_input import AudioInput

    _CHUNK = 3200  # 100ms at 16kHz × 2 bytes
    t_start = time.monotonic()

    async with client.empathic_voice.chat.connect(
        config_id=CONFIG_ID,
        api_key=HUME_API_KEY,
        session_settings={
            "custom_session_id": session_id,
            "audio": {"channels": 1, "encoding": "linear16", "sample_rate": 16_000},
        },
    ) as socket:
        print(f"Connected in {time.monotonic()-t_start:.2f}s\n")

        async def send_audio():
            print(f"Sending {len(audio_pcm)} bytes of audio in {_CHUNK}-byte chunks...")
            for i in range(0, len(audio_pcm), _CHUNK):
                chunk = audio_pcm[i:i + _CHUNK]
                payload = base64.b64encode(chunk).decode("ascii")
                await socket.send_audio_input(AudioInput(data=payload))
                await asyncio.sleep(0.05)
            # Send 1 second of silence to trigger Hume's VAD end-of-turn detection.
            # Without this, Hume keeps waiting for more speech and never transcribes.
            print("Sending 1s trailing silence to trigger VAD end-of-turn...")
            silence = b"\x00" * 32000  # 1s at 16kHz × 2 bytes
            for i in range(0, len(silence), _CHUNK):
                payload = base64.b64encode(silence[i:i + _CHUNK]).decode("ascii")
                await socket.send_audio_input(AudioInput(data=payload))
                await asyncio.sleep(0.05)
            print(f"Audio + silence sent. Waiting for events (30s max)...\n")

        send_task = asyncio.create_task(send_audio())

        deadline = time.monotonic() + 90
        event_count = 0
        async for event in socket:
            elapsed = time.monotonic() - t_start
            event_type = getattr(event, "type", type(event).__name__)

            if event_type == "user_message":
                text = getattr(getattr(event, "message", None), "content", "")
                print(f"  [{elapsed:.1f}s] USER_MESSAGE: {text!r}")
            elif event_type == "assistant_message":
                text = getattr(getattr(event, "message", None), "content", "")
                print(f"  [{elapsed:.1f}s] ASSISTANT_MESSAGE: {text!r}")
            elif event_type == "audio_output":
                data_len = len(getattr(event, "data", "") or "")
                print(f"  [{elapsed:.1f}s] AUDIO_OUTPUT: {data_len} chars base64")
            elif event_type == "assistant_end":
                print(f"  [{elapsed:.1f}s] ASSISTANT_END")
            elif event_type == "error":
                print(f"  [{elapsed:.1f}s] ERROR: {getattr(event, 'message', event)}")
            else:
                print(f"  [{elapsed:.1f}s] {event_type}: {event}")

            event_count += 1
            if time.monotonic() > deadline:
                print(f"\nDeadline reached after {event_count} events.")
                break

        send_task.cancel()

    print(f"\nDone. Total events: {event_count}  Wall time: {time.monotonic()-t_start:.1f}s")
    if event_count == 0:
        print("\nDIAGNOSIS: Hume EVI connected but sent NO events.")
        print("  - Check that the CLM webhook URL in the Hume config is reachable.")
        print("  - Is 'make serve' running? The coach LLM calls back to BASE_URL/chat/completions.")
        print(f"  - Config synced at: {mapping.get('synced_at', 'unknown')}")


asyncio.run(run_debug())
