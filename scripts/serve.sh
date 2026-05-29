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
NGROK_PID=""
LITELLM_PID=""

# Kill a PID and its entire process group. Sends SIGTERM, waits up to 2s,
# then SIGKILL if it's still alive. Silently no-ops on empty/dead PIDs.
_kill_tree() {
  local pid=$1
  [ -z "$pid" ] && return
  # Kill the process group so child processes (e.g. pg0 under honcho_serve.sh)
  # are also terminated.
  local pgid
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ') || true
  [ -n "$pgid" ] && kill -- "-$pgid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  # Wait up to 2s for graceful exit, then force-kill.
  for _ in 1 2 3 4; do
    sleep 0.5
    kill -0 "$pid" 2>/dev/null || return  # already gone
  done
  kill -9 "$pid" 2>/dev/null || true
}

cleanup() {
  trap '' EXIT INT TERM  # prevent re-entry when _kill_tree signals our own PGID
  echo "Shutting down..."
  _kill_tree "$LITELLM_PID"
  _kill_tree "$NGROK_PID"
  _kill_tree "$HONCHO_PID"
}
trap cleanup EXIT INT TERM

# ── Memory backend ────────────────────────────────────────────────────────────
# Priority: lib/honcho/ (self-hosted, local) > HONCHO_BASE_URL (external)
#           > HONCHO_API_KEY (cloud) > none
# lib/honcho/ always wins so you never need to comment out HONCHO_API_KEY
# just because you set up local Honcho.
if [ -d "$SCRIPT_DIR/../lib/honcho" ]; then
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

elif [ -n "${HONCHO_BASE_URL:-}" ]; then
  echo "Memory: Honcho at $HONCHO_BASE_URL (HONCHO_BASE_URL set)"

elif [ -n "${HONCHO_API_KEY:-}" ]; then
  echo "Memory: Honcho cloud (HONCHO_API_KEY set)"

else
  echo "Memory: none (run 'make setup-honcho' for self-hosted, or set HONCHO_API_KEY for cloud)"
fi

# ── LiteLLM proxy ─────────────────────────────────────────────────────────────
LITELLM_PORT=4000
BACKEND_TYPE="${BACKEND_TYPE:-pipeline}"
if [ "$BACKEND_TYPE" = "interactive" ] || [ "$BACKEND_TYPE" = "moshi" ]; then
  echo "Inference: BACKEND_TYPE=$BACKEND_TYPE — skipping LiteLLM proxy"
elif command -v litellm >/dev/null 2>&1 || uv run litellm --version >/dev/null 2>&1; then
  echo "Inference: starting LiteLLM proxy on port $LITELLM_PORT..."
  uv run litellm --config "$SCRIPT_DIR/../infra/litellm_config.yaml" \
    --port "$LITELLM_PORT" > /tmp/rehearse-litellm.log 2>&1 &
  LITELLM_PID=$!

  # Wait for proxy to be ready
  for i in $(seq 1 20); do
    sleep 0.5
    if curl -s "http://localhost:$LITELLM_PORT/health" >/dev/null 2>&1; then break; fi
  done
  echo "Inference: LiteLLM proxy ready at http://localhost:$LITELLM_PORT"
  export LITELLM_BASE_URL="http://localhost:$LITELLM_PORT"
else
  echo "Inference: litellm not installed — skipping proxy (run 'uv pip install litellm[proxy]' to enable)"
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
uv run uvicorn rehearse.api.app:create_app --factory --host 0.0.0.0 --port "$PORT"
