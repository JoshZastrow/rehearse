"""Build the FastAPI app for the live runtime.

This file wires together config, storage, session orchestration, Twilio route
handlers, and static artifact serving. It does not hold core business logic.
"""

from __future__ import annotations

import logging
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from rehearse.agents import build_clm_responder, mount_clm_routes
from rehearse.config import RuntimeConfig
from rehearse.session import SessionOrchestrator
from rehearse.storage import LocalFilesystemStore
from rehearse.telephony import TwilioRestClient, mount_twilio_routes


def _configure_logging(level: str) -> None:
    """Configure structlog and stdlib logging for the runtime process."""
    logging.basicConfig(level=level.upper(), format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
    )


def create_app(config: RuntimeConfig | None = None) -> FastAPI:
    """Create and return the fully wired FastAPI runtime app."""
    config = config or RuntimeConfig.from_env()
    _configure_logging(config.log_level)

    app = FastAPI(title="rehearse")
    store = LocalFilesystemStore(root=config.session_root, public_base_url=config.public_base_url)
    twilio_client = TwilioRestClient(config)
    orchestrator = SessionOrchestrator(store=store, notifier=twilio_client)
    clm_responder = build_clm_responder(config)

    mount_clm_routes(app, clm_responder, config)
    mount_twilio_routes(app, orchestrator, twilio_client, config)

    app.mount(
        "/sessions",
        StaticFiles(directory=str(config.session_root)),
        name="sessions",
    )
    web_dir = Path(__file__).resolve().parent.parent / "web"
    if web_dir.is_dir():
        app.mount("/viewer", StaticFiles(directory=str(web_dir), html=True), name="viewer")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Return a simple health check payload for uptime probes."""
        return {"status": "ok"}

    return app
