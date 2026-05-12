#!/usr/bin/env bash
# Start a self-hosted Honcho instance backed by embedded pg0.
#
# Usage: bash scripts/honcho_serve.sh
#
# Expects lib/honcho/ to exist (run `make setup-honcho` first).
# Reads LLM_ANTHROPIC_API_KEY from the environment for the deriver.
# Exports HONCHO_BASE_URL=http://localhost:8001 for the parent serve script.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HONCHO_DIR="$REPO_ROOT/lib/honcho"
HONCHO_PORT="${HONCHO_PORT:-8001}"
PG0_PORT="${PG0_PORT:-5433}"

if [ ! -d "$HONCHO_DIR" ]; then
  echo "ERROR: $HONCHO_DIR not found. Run 'make setup-honcho' first." >&2
  exit 1
fi

# Start embedded pg0 Postgres.
echo "Starting pg0 on port $PG0_PORT..."
uv run python3 "$REPO_ROOT/scripts/pg0_server.py" "$PG0_PORT" &
PG0_PID=$!

# Wait for pg0 to write its URI.
for i in $(seq 1 20); do
  sleep 0.5
  if [ -f /tmp/rehearse-pg0-uri.txt ]; then break; fi
done

if [ ! -f /tmp/rehearse-pg0-uri.txt ]; then
  echo "ERROR: pg0 did not start within 10s." >&2
  exit 1
fi

PG0_RAW_URI=$(cat /tmp/rehearse-pg0-uri.txt)
# Honcho requires postgresql+psycopg prefix.
DB_URI="${PG0_RAW_URI/postgresql:\/\//postgresql+psycopg://}"

echo "pg0 ready: $PG0_RAW_URI"

# Run Honcho migrations.
echo "Running Honcho migrations..."
(cd "$HONCHO_DIR" && DB_CONNECTION_URI="$DB_URI" uv run alembic upgrade head 2>&1 | tail -3)

# Start Honcho API server.
echo "Starting Honcho API on port $HONCHO_PORT..."
(
  cd "$HONCHO_DIR"
  DB_CONNECTION_URI="$DB_URI" \
  AUTH_USE_AUTH=false \
  SENTRY_ENABLED=false \
  LLM_ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
  uv run uvicorn src.main:app --host 0.0.0.0 --port "$HONCHO_PORT" 2>&1
) &
HONCHO_API_PID=$!

# Start Honcho deriver (background worker for representations/summaries).
echo "Starting Honcho deriver..."
(
  cd "$HONCHO_DIR"
  DB_CONNECTION_URI="$DB_URI" \
  AUTH_USE_AUTH=false \
  LLM_ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
  uv run python -m src.deriver 2>&1
) &
HONCHO_DERIVER_PID=$!

# Wait for the API to be ready.
echo "Waiting for Honcho API to be ready..."
for i in $(seq 1 30); do
  sleep 1
  if curl -sf "http://localhost:$HONCHO_PORT/health" > /dev/null 2>&1; then
    echo "Honcho API ready at http://localhost:$HONCHO_PORT"
    break
  fi
done

export HONCHO_BASE_URL="http://localhost:$HONCHO_PORT"
echo "$HONCHO_BASE_URL" > /tmp/rehearse-honcho-url.txt

# Keep running until killed; propagate SIGTERM to children.
cleanup() {
  kill "$HONCHO_API_PID" "$HONCHO_DERIVER_PID" "$PG0_PID" 2>/dev/null || true
  rm -f /tmp/rehearse-pg0-uri.txt /tmp/rehearse-honcho-url.txt
}
trap cleanup EXIT TERM INT

wait "$HONCHO_API_PID"
