#!/usr/bin/env bash
# Spin up all Rehearse components: ngrok + (optional) local Honcho + rehearse server.
#
# Memory backend selection:
#   HONCHO_API_KEY set   → use Honcho cloud  (no local processes started)
#   HONCHO_BASE_URL set  → use that URL as Honcho endpoint (you manage the server)
#   lib/honcho/ exists   → start self-hosted Honcho with embedded pg0
#   (none of the above)  → NullCallerMemory (no caller memory)
set -euo pipefail

PORT="${PORT:-8000}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load .env so all other vars (keys, etc.) are present, but let shell
# overrides win (load_dotenv override=True in config.py does the same).
if [ -f .env ]; then
  set -o allexport
  # shellcheck disable=SC1091
  source .env
  set +o allexport
fi

HONCHO_PID=""

cleanup() {
  echo "Shutting down..."
  [ -n "$HONCHO_PID" ] && kill "$HONCHO_PID" 2>/dev/null || true
  kill "$NGROK_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ── Memory backend ────────────────────────────────────────────────────────────
if [ -n "${HONCHO_API_KEY:-}" ]; then
  echo "Memory: Honcho cloud (HONCHO_API_KEY set)"

elif [ -n "${HONCHO_BASE_URL:-}" ]; then
  echo "Memory: Honcho at $HONCHO_BASE_URL (HONCHO_BASE_URL set)"

elif [ -d "$SCRIPT_DIR/../lib/honcho" ]; then
  echo "Memory: starting self-hosted Honcho with embedded pg0..."
  bash "$SCRIPT_DIR/honcho_serve.sh" &
  HONCHO_PID=$!

  # Wait for honcho_serve.sh to write the base URL.
  for i in $(seq 1 30); do
    sleep 1
    if [ -f /tmp/rehearse-honcho-url.txt ]; then break; fi
  done

  if [ ! -f /tmp/rehearse-honcho-url.txt ]; then
    echo "ERROR: local Honcho did not start within 30s." >&2
    exit 1
  fi

  export HONCHO_BASE_URL
  HONCHO_BASE_URL=$(cat /tmp/rehearse-honcho-url.txt)
  echo "Memory: Honcho ready at $HONCHO_BASE_URL"

else
  echo "Memory: none (set HONCHO_API_KEY or run 'make setup-honcho' for self-hosted)"
fi

# ── Sync Hume configs ─────────────────────────────────────────────────────────
echo "Syncing Hume EVI configs..."
uv run rehearse-hume sync 2>&1 | tail -2

# ── ngrok ─────────────────────────────────────────────────────────────────────
echo "Starting ngrok tunnel on port $PORT..."
ngrok http "$PORT" --log=stdout > /tmp/rehearse-ngrok.log 2>&1 &
NGROK_PID=$!

TUNNEL_URL=""
for i in $(seq 1 20); do
  sleep 0.5
  TUNNEL_URL=$(curl -s http://localhost:4040/api/tunnels \
    | python3 -c "import sys,json; t=json.load(sys.stdin).get('tunnels',[]); print(next((x['public_url'] for x in t if x['proto']=='https'), ''))" 2>/dev/null || true)
  if [ -n "$TUNNEL_URL" ]; then break; fi
done

if [ -z "$TUNNEL_URL" ]; then
  echo "ERROR: ngrok tunnel did not start within 10s. Check /tmp/rehearse-ngrok.log." >&2
  exit 1
fi

echo "Tunnel ready: $TUNNEL_URL"

# ── Rehearse server ───────────────────────────────────────────────────────────
export BASE_URL="$TUNNEL_URL"
echo "Starting rehearse on port $PORT"
uv run uvicorn rehearse.app:create_app --factory --host 0.0.0.0 --port "$PORT"
