"""Factory for creating ConversationBackend instances from config."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rehearse.config import RuntimeConfig


def create_backend(config: RuntimeConfig) -> object:
    """Return the appropriate backend for the configured backend_type."""
    match config.backend_type:
        case "interactive":
            from rehearse.backends.interactive import InteractiveBackend

            return InteractiveBackend(endpoint=config.interactive_endpoint)
        case _:
            raise ValueError(f"Unknown backend_type: {config.backend_type!r}")
