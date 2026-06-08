"""Minimal LiveKit JWT token server for local dev.

GET /api/livekit/token  →  {"token": "<jwt>"}

Required env vars:
  LIVEKIT_API_KEY
  LIVEKIT_API_SECRET
  LIVEKIT_ROOM_NAME   (optional, default: rehearse-room)
  TOKEN_SERVER_PORT   (optional, default: 8765)
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Walk up from web/livekit/ to find the repo-root .env regardless of CWD.
_REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev only
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/livekit/token")
async def get_token() -> dict:
    from livekit.api import AccessToken, VideoGrants  # noqa: PLC0415

    api_key = os.environ["LIVEKIT_API_KEY"]
    api_secret = os.environ["LIVEKIT_API_SECRET"]
    room_name = os.environ.get("LIVEKIT_ROOM_NAME", "rehearse-room")
    identity = f"user-{uuid.uuid4().hex[:8]}"

    token = (
        AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_grants(VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )
    return {"token": token}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("TOKEN_SERVER_PORT", "8765"))
    uvicorn.run(app, host="0.0.0.0", port=port)
