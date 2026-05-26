"""Backwards-compatibility shim. Import from rehearse.backends.transport instead.

This shim re-exports both the new product-language names and the old
telecom-jargon names without firing the DeprecationWarning at import time.
The warning is reserved for direct attribute access on `rehearse.backends.transport`
itself (via PEP 562 __getattr__).
"""

from rehearse.backends.transport import (  # noqa: F401
    Channel,
    InMemoryTransportEndpoint,
    InMemoryTwoWayChannel,
    RuntimeTransport,
    TransportClosedError,
    TransportEvent,
    TransportEventKind,
    TransportSide,
    TransportStatus,
    TwoWayChannel,
)

# Old-name aliases (no warning here — this shim's whole point is back-compat).
InMemoryDuplexTransport = InMemoryTwoWayChannel
RuntimeDuplexEndpoint = TwoWayChannel
