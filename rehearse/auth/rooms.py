"""Per-session LiveKit room naming.

The token server mints one room per call, embedding the Clerk user id so the
agent — which only sees the room it's dispatched to — can recover who the call
belongs to for metering. Shared here so both `web/livekit/token_server.py` and
`web/livekit/agent/agent.py` import the exact same convention.

    make_room_name("user_2N")  ->  "rehearse-user_2N-ab12cd34"
    clerk_id_from_room(...)    ->  "user_2N"

Clerk ids (`user_...`) contain no hyphens, so the trailing `-<uuid>` is
unambiguous.
"""

from __future__ import annotations

import uuid


def make_room_name(clerk_user_id: str) -> str:
    """Per-session room name embedding the user id: rehearse-<uid>-<uuid>."""
    return f"rehearse-{clerk_user_id}-{uuid.uuid4().hex[:8]}"


def clerk_id_from_room(room_name: str) -> str | None:
    """Recover the clerk_user_id from a room minted by `make_room_name`.

    Returns None if the name doesn't match the expected shape.
    """
    if not room_name.startswith("rehearse-"):
        return None
    middle = room_name[len("rehearse-") :]
    uid, _, suffix = middle.rpartition("-")
    if not uid or not suffix:
        return None
    return uid
