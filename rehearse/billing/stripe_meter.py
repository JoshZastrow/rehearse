"""Report metered usage to Stripe.

One function: `report_meter_event(customer_id, credits, session_id)`. It wraps
Stripe's metered-billing API (`stripe.billing.MeterEvent.create`) against the
`rehearse_credits` meter.

Safe by default: when `STRIPE_SECRET_KEY` is unset (local dev, hermetic tests)
the reporter logs and returns without importing or calling Stripe, so nothing
touches the network under the default test markers. `session_id` is passed as the
event's `identifier` so Stripe itself de-dupes a retried report.
"""

from __future__ import annotations

import os

import structlog

log = structlog.get_logger(__name__)

# Matches the Stripe Meter's `event_name` created in the dashboard.
METER_EVENT_NAME = "rehearse_credits"


def report_meter_event(
    customer_id: str | None,
    credits: float,
    session_id: str,
    *,
    api_key: str | None = None,
    meter_event_name: str = METER_EVENT_NAME,
) -> bool:
    """Report `credits` of usage for one session to Stripe.

    Args:
        customer_id: Stripe customer id to bill. If None (user has no Stripe
            customer yet), the call is skipped.
        credits: Credits consumed by the session (from `session_credits`).
        session_id: Used as the Stripe event `identifier` for idempotency.
        api_key: Override for `STRIPE_SECRET_KEY` (tests). When neither is set,
            the reporter is a logging no-op.
        meter_event_name: The Stripe meter's event name.

    Returns:
        True if an event was sent to Stripe, False if the call was skipped
        (unconfigured key or missing customer).
    """
    key = api_key or os.environ.get("STRIPE_SECRET_KEY")
    if not key:
        log.info(
            "stripe_meter.skip.unconfigured",
            session_id=session_id,
            credits=credits,
            reason="STRIPE_SECRET_KEY not set",
        )
        return False
    if not customer_id:
        log.warning(
            "stripe_meter.skip.no_customer",
            session_id=session_id,
            credits=credits,
        )
        return False

    import stripe  # noqa: PLC0415 — optional dependency, only needed when configured

    stripe.api_key = key
    stripe.billing.MeterEvent.create(
        event_name=meter_event_name,
        identifier=session_id,
        payload={
            "stripe_customer_id": customer_id,
            "value": str(credits),
        },
    )
    log.info(
        "stripe_meter.reported",
        session_id=session_id,
        customer_id=customer_id,
        credits=credits,
    )
    return True
