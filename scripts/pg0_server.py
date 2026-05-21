#!/usr/bin/env python3
"""Start an embedded pg0 Postgres instance and print its URI to stdout.

Usage: python3 scripts/pg0_server.py [port]

Stays alive until SIGTERM/SIGINT. Designed to be backgrounded by serve.sh:
  python3 scripts/pg0_server.py 5433 &
  PG0_URI=$(head -1 /tmp/rehearse-pg0-uri.txt)
"""

import signal
import sys

port = int(sys.argv[1]) if len(sys.argv) > 1 else 5433

from pg0 import Pg0, Pg0AlreadyRunningError  # noqa: E402

pg = Pg0(port=port)
try:
    pg.start()
except Pg0AlreadyRunningError:
    pg.stop()
    pg.start()

def _stop(signum, frame):
    # Disarm handlers before calling pg.stop() — subprocess.run() inside
    # pg.stop() blocks on selector.select(), which is interrupted by signals,
    # causing re-entrant calls into _stop and a RecursionError.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    pg.stop()
    sys.exit(0)


# Install handlers before printing the URI so any signal arriving after the
# URI line is emitted is caught by _stop rather than the default handler.
signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)

uri = pg.uri
print(uri, flush=True)

# Also write to a temp file so serve.sh can read it without pipe timing issues.
with open("/tmp/rehearse-pg0-uri.txt", "w") as f:
    f.write(uri + "\n")

signal.pause()
