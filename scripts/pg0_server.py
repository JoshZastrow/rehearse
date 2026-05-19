#!/usr/bin/env python3
"""Start an embedded pg0 Postgres instance and print its URI to stdout.

Usage: python3 scripts/pg0_server.py [port]

Stays alive until SIGTERM/SIGINT. Designed to be backgrounded by serve.sh:
  python3 scripts/pg0_server.py 5433 &
  PG0_URI=$(head -1 /tmp/rehearse-pg0-uri.txt)
"""

import signal
import socket
import sys
import time

port = int(sys.argv[1]) if len(sys.argv) > 1 else 5433

from pg0 import Pg0, Pg0AlreadyRunningError  # noqa: E402


def _wait_for_port(host: str, port: int, timeout: float = 30.0) -> bool:
    """Return True once the TCP port accepts a connection, False on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.25)
    return False


pg = Pg0(port=port)

print(f"[pg0] starting on port {port}...", file=sys.stderr, flush=True)
try:
    pg.start()
    print("[pg0] started", file=sys.stderr, flush=True)
except Pg0AlreadyRunningError:
    print("[pg0] stale instance detected — stopping...", file=sys.stderr, flush=True)
    pg.stop()
    # Give the OS a moment to release the port before we try to bind it again.
    time.sleep(1)
    print("[pg0] restarting...", file=sys.stderr, flush=True)
    pg.start()
    print("[pg0] restarted", file=sys.stderr, flush=True)

print(f"[pg0] waiting for port {port} to accept connections...", file=sys.stderr, flush=True)
if not _wait_for_port("127.0.0.1", port):
    print(f"[pg0] ERROR: port {port} not ready after 30s", file=sys.stderr, flush=True)
    sys.exit(1)
print(f"[pg0] port {port} ready", file=sys.stderr, flush=True)

uri = pg.uri
print(uri, flush=True)

# Also write to a temp file so serve.sh can read it without pipe timing issues.
with open("/tmp/rehearse-pg0-uri.txt", "w") as f:
    f.write(uri + "\n")

print(f"[pg0] URI written: {uri}", file=sys.stderr, flush=True)


def _stop(signum, frame):
    print("[pg0] shutting down...", file=sys.stderr, flush=True)
    pg.stop()
    sys.exit(0)


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)

print("[pg0] ready. Waiting for shutdown signal.", file=sys.stderr, flush=True)
signal.pause()
