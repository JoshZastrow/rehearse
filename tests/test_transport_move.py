"""Verify that transport module is available from both canonical and legacy paths."""

from __future__ import annotations

import pytest


def test_canonical_import() -> None:
    from rehearse.transport import (
        InMemoryDuplexTransport,
        RuntimeDuplexEndpoint,
        RuntimeTransport,
        TransportClosedError,
        TransportEvent,
        TransportEventKind,
    )

    assert InMemoryDuplexTransport is not None
    assert RuntimeDuplexEndpoint is not None
    assert RuntimeTransport is RuntimeDuplexEndpoint


def test_legacy_import_still_works() -> None:
    from rehearse.eval.transports import (
        InMemoryDuplexTransport,
        RuntimeDuplexEndpoint,
        RuntimeTransport,
        TransportClosedError,
        TransportEvent,
    )

    assert InMemoryDuplexTransport is not None


def test_audio_bytes_round_trip() -> None:
    """Audio bytes via TransportEvent data field round-trip unchanged."""
    from rehearse.transport import InMemoryDuplexTransport, TransportEvent

    import asyncio

    async def _run() -> None:
        transport = InMemoryDuplexTransport()
        audio_bytes = b"\x00\x01\x02\x03" * 1000
        event = await transport.customer.send("audio", data=audio_bytes)
        received = await transport.runtime.receive()
        assert received.kind == "audio"
        assert received.data == audio_bytes

    asyncio.run(_run())


def test_both_paths_same_class() -> None:
    from rehearse.eval.transports import InMemoryDuplexTransport as Legacy
    from rehearse.transport import InMemoryDuplexTransport as Canonical

    assert Legacy is Canonical
