"""Verify that transport module is available from both canonical and legacy paths."""

from __future__ import annotations

import pytest


def test_canonical_import() -> None:
    from rehearse.transport import (
        InMemoryTwoWayChannel,
        TwoWayChannel,
        RuntimeTransport,
        TransportClosedError,
        TransportEvent,
        TransportEventKind,
    )

    assert InMemoryTwoWayChannel is not None
    assert TwoWayChannel is not None
    assert RuntimeTransport is TwoWayChannel


def test_legacy_import_still_works() -> None:
    from rehearse.eval.transports import (
        InMemoryTwoWayChannel,
        TwoWayChannel,
        RuntimeTransport,
        TransportClosedError,
        TransportEvent,
    )

    assert InMemoryTwoWayChannel is not None


def test_audio_bytes_round_trip() -> None:
    """Audio bytes via TransportEvent data field round-trip unchanged."""
    from rehearse.transport import InMemoryTwoWayChannel, TransportEvent

    import asyncio

    async def _run() -> None:
        transport = InMemoryTwoWayChannel()
        audio_bytes = b"\x00\x01\x02\x03" * 1000
        event = await transport.customer.send("audio", data=audio_bytes)
        received = await transport.runtime.receive()
        assert received.kind == "audio"
        assert received.data == audio_bytes

    asyncio.run(_run())


def test_both_paths_same_class() -> None:
    from rehearse.eval.transports import InMemoryTwoWayChannel as Legacy
    from rehearse.transport import InMemoryTwoWayChannel as Canonical

    assert Legacy is Canonical
