"""Pure GPU-cost → credit formula. No I/O, no external calls.

Single source of truth for what a session costs. Stripe only multiplies the
returned `credits` by a flat unit price ($0.01/credit); all the arithmetic that
matters lives here so it can be unit-tested and replayed over any past session's
`session.json`.

Billable unit — session wall-clock GPU-seconds:

    gpu_seconds  = (finalized_at - created_at).total_seconds()
    raw_cost_usd = gpu_seconds * (A10G_USD_PER_HOUR / 3600)
    billed_usd   = raw_cost_usd * MARKUP           # 70% markup
    credits      = billed_usd / USD_PER_CREDIT     # 1 credit = $0.01

Confirm A10G $/hr at modal.com/pricing at build time — it is the one constant
that moves.
"""

from __future__ import annotations

from dataclasses import dataclass

from rehearse.types import Session

# --- Rate constants (confirm at build time) --------------------------------
A10G_USD_PER_HOUR = 1.10
"""Modal A10G on-demand rate. $1.10/GPU-hr = $0.0003056/GPU-sec (2026-07)."""

MARKUP = 1.70
"""Multiplier applied to raw GPU cost: 70% margin over Modal's price."""

USD_PER_CREDIT = 0.01
"""A credit is worth one US cent. Matches the Stripe metered Price unit amount."""

_SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class SessionCost:
    """Breakdown of what one finished session costs and what it bills.

    Every field is derived from `gpu_seconds`; the struct exists so callers can
    log the intermediate cost (for margin monitoring) alongside the `credits`
    that get reported to Stripe.
    """

    gpu_seconds: float
    raw_cost_usd: float
    billed_usd: float
    credits: float


def session_credits(session: Session) -> SessionCost:
    """Compute the billable cost of a finished session from its manifest.

    Args:
        session: A finalized `Session`. `finalized_at` must be set (the metering
            hook in the agent writes it on disconnect) and must not precede
            `created_at`.

    Returns:
        A `SessionCost` with the GPU seconds, raw USD cost, marked-up billed USD,
        and the `credits` to report to Stripe.

    Raises:
        ValueError: if `finalized_at` is None (session never finalized) or is
            earlier than `created_at` (clock went backwards / corrupt manifest).
    """
    if session.finalized_at is None:
        raise ValueError(
            f"session {session.id} has no finalized_at — cannot bill an "
            "unfinished session"
        )

    gpu_seconds = (session.finalized_at - session.created_at).total_seconds()
    if gpu_seconds < 0:
        raise ValueError(
            f"session {session.id} finalized_at {session.finalized_at} precedes "
            f"created_at {session.created_at} — refusing to bill negative time"
        )

    raw_cost_usd = gpu_seconds * (A10G_USD_PER_HOUR / _SECONDS_PER_HOUR)
    billed_usd = raw_cost_usd * MARKUP
    credits = billed_usd / USD_PER_CREDIT
    return SessionCost(
        gpu_seconds=gpu_seconds,
        raw_cost_usd=raw_cost_usd,
        billed_usd=billed_usd,
        credits=credits,
    )
