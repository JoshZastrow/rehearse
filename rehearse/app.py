"""Build the FastAPI app for the live runtime.

This file wires together config, storage, session orchestration, Twilio route
handlers, and static artifact serving. It does not hold core business logic.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from anthropic import AsyncAnthropic
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from rehearse.agents import build_clm_responder, mount_clm_routes
from rehearse.agents.clm import validate_anthropic_credentials
from rehearse.config import RuntimeConfig
from rehearse.finalize_sweeper import FinalizeSweeper
from rehearse.session import SessionOrchestrator
from rehearse.storage import LocalFilesystemStore
from rehearse.telephony import TwilioRestClient, mount_twilio_routes
from rehearse.viewer import mount_viewer_routes


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

    store = LocalFilesystemStore(root=config.session_root, public_base_url=config.public_base_url)
    twilio_client = TwilioRestClient(config)
    notifier = None if config.disable_sms else twilio_client
    orchestrator = SessionOrchestrator(store=store, notifier=notifier)
    if config.disable_sms:
        structlog.get_logger(__name__).info("sms.disabled")
    clm_responder = build_clm_responder(config)

    sweeper = FinalizeSweeper(
        orchestrator,
        store,
        max_call_seconds=config.finalize_sweep_max_call_seconds,
        grace_seconds=config.finalize_sweep_grace_seconds,
        sweep_interval_seconds=config.finalize_sweep_interval_seconds,
    )

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Validate provider credentials, then run background tasks for the app."""
        if config.anthropic_api_key:
            await validate_anthropic_credentials(
                AsyncAnthropic(api_key=config.anthropic_api_key),
                config.anthropic_model,
            )
        if config.finalize_sweep_enabled:
            # Crash recovery: any session still `in_progress` on disk has no
            # in-memory handle (we just started), so its Twilio stream is
            # gone. Finalize-as-failed before the periodic sweeper takes over.
            recovered = await sweeper.recover_orphans()
            if recovered:
                structlog.get_logger(__name__).info(
                    "finalize_sweeper.recovered_on_startup",
                    count=len(recovered),
                    session_ids=recovered,
                )
            sweeper.start()
        try:
            yield
        finally:
            if config.finalize_sweep_enabled:
                await sweeper.stop()

    app = FastAPI(title="rehearse", lifespan=_lifespan)

    mount_clm_routes(app, clm_responder, config)
    mount_twilio_routes(app, orchestrator, twilio_client, config)
    mount_viewer_routes(app, store)

    app.mount(
        "/sessions",
        StaticFiles(directory=str(config.session_root)),
        name="sessions",
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Return a simple health check payload for uptime probes."""
        return {"status": "ok"}

    return app
