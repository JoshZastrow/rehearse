from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rehearse.backends.interactive.backend import InteractiveBackend

__all__ = ["InteractiveBackend"]


def __getattr__(name: str):
    """Lazily expose InteractiveBackend without importing torch at package load.

    ModalInteractiveBackend (imported by the live API server via the factory)
    triggers this package's import; keeping the heavy local-Moshi backend behind
    a PEP 562 lazy attribute avoids pulling torch into that path while still
    making ``from rehearse.backends.interactive import InteractiveBackend`` work.
    """
    if name == "InteractiveBackend":
        from rehearse.backends.interactive.backend import InteractiveBackend

        return InteractiveBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
