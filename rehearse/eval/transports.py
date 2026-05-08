"""Backwards-compatibility shim. Import from rehearse.transport instead."""

from rehearse.transport import (  # noqa: F401
    InMemoryDuplexTransport,
    InMemoryTransportEndpoint,
    RuntimeDuplexEndpoint,
    RuntimeTransport,
    TransportClosedError,
    TransportEvent,
    TransportEventKind,
    TransportSide,
    TransportStatus,
)
