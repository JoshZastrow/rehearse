"""Verify Clerk session JWTs at the token-server choke point.

The frontend gets a short-lived session token from Clerk (`useAuth().getToken()`)
and sends it as `Authorization: Bearer <jwt>`. The token server verifies the
signature against Clerk's JWKS and returns the Clerk user id (`sub`).

`ClerkVerifier` is a Protocol so the token gate can be tested with a fake
verifier that returns a canned user id — no network, no keys. `JWKSClerkVerifier`
is the production implementation over PyJWT + Clerk's rotating JWKS.
"""

from __future__ import annotations

from typing import Protocol


class InvalidClerkToken(Exception):
    """Raised when a Clerk JWT is missing, malformed, expired, or unverifiable."""


class ClerkVerifier(Protocol):
    """Verify a Clerk session JWT and return the Clerk user id."""

    def verify(self, token: str) -> str:
        """Return the Clerk user id (`sub`) or raise `InvalidClerkToken`."""
        ...


class JWKSClerkVerifier:
    """Verify Clerk JWTs against Clerk's JWKS endpoint (production).

    Args:
        jwks_url: Clerk JWKS URL, e.g.
            ``https://<slug>.clerk.accounts.dev/.well-known/jwks.json``.
        issuer: Expected `iss` claim, e.g. ``https://<slug>.clerk.accounts.dev``.
            When None, the issuer is not checked (accept any) — set it in prod.

    Uses `PyJWKClient`, which fetches and caches signing keys and picks the key
    matching the token's `kid`, so key rotation is handled transparently.
    """

    def __init__(self, jwks_url: str, issuer: str | None = None) -> None:
        import jwt  # noqa: PLC0415 — pyjwt[crypto]
        from jwt import PyJWKClient  # noqa: PLC0415

        self._jwt = jwt
        self._issuer = issuer
        self._jwk_client = PyJWKClient(jwks_url)

    def verify(self, token: str) -> str:
        try:
            signing_key = self._jwk_client.get_signing_key_from_jwt(token)
            claims = self._jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=self._issuer,
                options={
                    "require": ["sub", "exp"],
                    "verify_iss": self._issuer is not None,
                },
            )
        except Exception as exc:  # PyJWT raises many subclasses; normalize them.
            raise InvalidClerkToken(str(exc)) from exc

        sub = claims.get("sub")
        if not sub:
            raise InvalidClerkToken("token has no `sub` claim")
        return str(sub)


def build_clerk_verifier(
    jwks_url: str | None,
    issuer: str | None = None,
) -> ClerkVerifier | None:
    """Build a `JWKSClerkVerifier` when configured, else return None.

    Returns None when `jwks_url` is unset so the caller can decide the policy
    (the token server rejects all requests with 503 rather than silently
    allowing unauthenticated access).
    """
    if not jwks_url:
        return None
    return JWKSClerkVerifier(jwks_url, issuer=issuer)
