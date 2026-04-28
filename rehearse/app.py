"""FastAPI application.

Phase R1 entry point. Wires:
  - POST /twilio/sms     inbound SMS trigger
  - POST /twilio/voice   outbound-call TwiML (hard-coded <Say> in R1)
  - POST /twilio/status  call status callbacks
  - WS   /media/{id}     Twilio Media Streams (no-op accept in R1)
  - GET  /sessions/{id}/* static artifact files
  - GET  /viewer/*       static viewer page

No business logic here. Wires routes to handlers and pipelines.
"""

from __future__ import annotations

import logging
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from rehearse.config import RuntimeConfig
from rehearse.session import SessionOrchestrator
from rehearse.storage import LocalFilesystemStore
from rehearse.telephony import TwilioRestClient, mount_twilio_routes


def _configure_logging(level: str) -> None:
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
    config = config or RuntimeConfig.from_env()
    _configure_logging(config.log_level)

    app = FastAPI(title="rehearse")
    store = LocalFilesystemStore(root=config.session_root, public_base_url=config.public_base_url)
    orchestrator = SessionOrchestrator(store=store)
    twilio_client = TwilioRestClient(config)

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
        return {"status": "ok"}

    return app
