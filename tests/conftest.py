"""Shared pytest fixtures for the rehearse test suite.

honcho_server
─────────────
Session-scoped fixture that starts a self-hosted Honcho API server backed by
an embedded pg0 Postgres instance. Tests that need real Honcho memory (rather
than InMemoryCallerMemory) can request this fixture.

Prerequisites:
  make setup-honcho   # clones lib/honcho/ and installs its deps

The fixture is automatically skipped when lib/honcho/ is absent, so the
default `make test` run is never broken by missing infrastructure.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

REPO_ROOT = Path(__file__).parent.parent
HONCHO_DIR = REPO_ROOT / "lib" / "honcho"
HONCHO_PORT = 8002
PG0_PORT = 5434  # distinct from dev server port (5433)


@pytest.fixture(scope="session")
def honcho_server():
    """Start a local Honcho API server backed by embedded pg0.

    Yields the base URL (e.g. ``http://localhost:8002``) for the duration of
    the test session, then tears everything down.

    Skipped automatically when ``lib/honcho/`` has not been set up.
    """
    if not (HONCHO_DIR / "src" / "main.py").exists():
        pytest.skip(
            "lib/honcho/ not found — run 'make setup-honcho' to enable "
            "Honcho server fixtures"
        )

    from pg0 import Pg0  # installed by make setup-honcho indirectly via pg0-embedded

    pg = Pg0(port=PG0_PORT)
    pg.start()

    # Honcho requires the postgresql+psycopg:// prefix for SQLAlchemy.
    raw_uri = pg.uri  # postgresql://postgres:postgres@127.0.0.1:<port>/postgres
    db_uri = raw_uri.replace("postgresql://", "postgresql+psycopg://")

    env = {
        **os.environ,
        "DB_CONNECTION_URI": db_uri,
        "AUTH_USE_AUTH": "false",
        "SENTRY_ENABLED": "false",
        # LLM keys are passed through so the deriver can work if started,
        # but for peer-metadata operations (consent memory) no LLM is needed.
        "LLM_ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
    }

    # Run Alembic migrations once.
    subprocess.run(
        [sys.executable, "-m", "uv", "run", "alembic", "upgrade", "head"],
        cwd=HONCHO_DIR,
        env=env,
        check=True,
        capture_output=True,
    )

    # Start the Honcho API server as a background subprocess.
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uv",
            "run",
            "fastapi",
            "run",
            "src/main.py",
            "--port",
            str(HONCHO_PORT),
        ],
        cwd=HONCHO_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    base_url = f"http://localhost:{HONCHO_PORT}"

    # Poll until the server responds or timeout after 30 s.
    deadline = time.monotonic() + 30
    ready = False
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=1)
            if r.status_code < 500:
                ready = True
                break
        except requests.RequestException:
            pass
        time.sleep(0.5)

    if not ready:
        proc.terminate()
        pg.stop()
        pytest.fail("Honcho server did not become ready within 30 s")

    yield base_url

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    pg.stop()
