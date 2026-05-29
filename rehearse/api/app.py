"""Build the FastAPI app for the live runtime.

This file wires together config, storage, session orchestration, Twilio route
handlers, and static artifact serving. It does not hold core business logic.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import Any

import structlog
from anthropic import AsyncAnthropic
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from rehearse.agents import build_clm_responder, mount_clm_routes
from rehearse.agents.clm import validate_anthropic_credentials
from rehearse.api.telephony import TwilioRestClient, _prewarm_modal, mount_twilio_routes
from rehearse.api.viewer import mount_viewer_routes
from rehearse.config import RuntimeConfig
from rehearse.session.finalize_sweeper import FinalizeSweeper
from rehearse.session.session import SessionOrchestrator
from rehearse.storage import LocalFilesystemStore

# Backends that use the CLM webhook (Anthropic/LiteLLM required).
# interactive/moshi handle the full conversation loop via Moshi — no CLM calls.
_CLM_BACKENDS: frozenset[str] = frozenset({"managed", "pipeline"})


async def _validate_anthropic(config: RuntimeConfig) -> None:
    if config.anthropic_api_key:
        await validate_anthropic_credentials(
            AsyncAnthropic(api_key=config.anthropic_api_key),
            config.anthropic_model,
        )


async def _load_interactive_models(config: RuntimeConfig) -> None:
    if config.interactive_modal_endpoint:
        return  # models run on Modal; nothing to load locally
    from rehearse.backends.interactive.loader import load_models
    await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: load_models(
            config.interactive_checkpoint_path,
            config.interactive_model_repo,
            config.interactive_device,
        ),
    )


async def _prewarm_interactive(config: RuntimeConfig) -> None:
    if config.interactive_modal_endpoint:
        await _prewarm_modal(config.interactive_modal_endpoint)


# Each entry: (handler_name, set_of_applicable_backend_types, async_handler_fn)
_STARTUP_HANDLERS: list[
    tuple[str, frozenset[str], Callable[[RuntimeConfig], Coroutine[Any, Any, None]]]
] = [
    ("anthropic_credentials", _CLM_BACKENDS, _validate_anthropic),
    ("interactive_models", frozenset({"interactive", "moshi"}), _load_interactive_models),
    ("interactive_prewarm", frozenset({"interactive", "moshi"}), _prewarm_interactive),
]


def _configure_logging(level: str) -> None:
    """Configure structlog and stdlib logging for the runtime process."""
    logging.basicConfig(level=level.upper(), format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
            structlog.dev.ConsoleRenderer(),
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
        log = structlog.get_logger(__name__)
        for name, backends, handler in _STARTUP_HANDLERS:
            if config.backend_type in backends:
                await handler(config)
        log.info(
            "Server ready",
            backend=config.backend_type,
            url=config.public_base_url,
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
