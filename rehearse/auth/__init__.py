"""Authentication helpers for the consumer-facing surface.

Currently just Clerk JWT verification for the LiveKit token endpoint.
"""

from __future__ import annotations

from rehearse.auth.clerk import (
    ClerkVerifier,
    InvalidClerkToken,
    JWKSClerkVerifier,
    build_clerk_verifier,
)

__all__ = [
    "ClerkVerifier",
    "InvalidClerkToken",
    "JWKSClerkVerifier",
    "build_clerk_verifier",
]
