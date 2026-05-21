"""Lifecycle test for scripts/pg0_server.py.

Verifies that the embedded pg0 Postgres server starts, prints its URI,
and shuts down cleanly on SIGINT — no traceback, exit code 0.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
PG0_SCRIPT = REPO_ROOT / "scripts" / "pg0_server.py"
PG0_TEST_PORT = 5445  # distinct from dev (5433) and honcho-fixture (5434) ports


def test_pg0_server_starts_and_stops_cleanly():
    proc = subprocess.Popen(
        [sys.executable, str(PG0_SCRIPT), str(PG0_TEST_PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    uri = ""
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if line.startswith("postgresql://"):
            uri = line.strip()
            break

    if not uri:
        proc.kill()
        stderr_out = proc.stderr.read()
        pytest.fail(f"pg0_server never printed a URI within 20s.\nstderr:\n{stderr_out}")

    proc.send_signal(signal.SIGINT)

    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("pg0_server did not exit within 15s after SIGINT")

    stderr_out = proc.stderr.read()
    assert proc.returncode == 0, (
        f"pg0_server exited with code {proc.returncode}\nstderr:\n{stderr_out}"
    )
    assert "Traceback" not in stderr_out, f"Unexpected traceback:\n{stderr_out}"
    assert "RecursionError" not in stderr_out, f"Re-entrant signal bug:\n{stderr_out}"
