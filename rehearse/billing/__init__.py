"""Metered GPU-cost → credit billing for Rehearse.

This package converts a finished session's GPU wall-clock into billable credits
and reports usage to Stripe. It has three layers, each usable in isolation:

  cost.py          pure formula — `session_credits(session)` (no I/O).
  store.py         psycopg wrapper over the `users` + `usage_events` tables.
  stripe_meter.py  report a Stripe metered-billing event (no-op when unconfigured).

The metered unit is **session wall-clock GPU-seconds**: the interactive Moshi
backend (infra/interactive.py) dedicates one A10G GPU per connection for the
session's lifetime, so seconds-held is the honest cost driver — not tokens.
"""

from __future__ import annotations

from rehearse.billing.cost import (
    A10G_USD_PER_HOUR,
    MARKUP,
    USD_PER_CREDIT,
    SessionCost,
    session_credits,
)

__all__ = [
    "A10G_USD_PER_HOUR",
    "MARKUP",
    "USD_PER_CREDIT",
    "SessionCost",
    "session_credits",
]
