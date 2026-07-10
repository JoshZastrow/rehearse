"""Apply Stripe webhook events to the billing store.

Two responsibilities, split so the effect logic is testable without Stripe:

  verify_stripe_event(payload, sig_header, secret)  — signature verification
      (thin wrapper over `stripe.Webhook.construct_event`); lives at the HTTP
      edge in whatever app mounts the route.

  apply_stripe_event(event, store)                   — pure effect application
      given an already-parsed event dict. This is what the tests exercise.

Event → effect:
  checkout.session.completed  → set_billing_ready(clerk_id, stripe_customer_id)
      A payment method is now on file; the user may start calls. `clerk_id` is
      read from the Checkout Session's `client_reference_id` (set to the Clerk
      user id when creating the session).
  invoice.paid                → reset_monthly_usage(clerk_id)
      New billing cycle settled; zero the post-pay tally. `clerk_id` comes from
      the customer's `metadata.clerk_user_id`.

Unrecognized events are ignored (logged and skipped), which is the correct
default for a Stripe endpoint subscribed to more events than it handles.
"""

from __future__ import annotations

import structlog

from rehearse.billing.store import BillingStore

log = structlog.get_logger(__name__)


class InvalidStripeSignature(Exception):
    """Raised when a webhook payload fails Stripe signature verification."""


def verify_stripe_event(payload: bytes, sig_header: str, secret: str) -> dict:
    """Verify a webhook payload's signature and return the parsed event.

    Raises `InvalidStripeSignature` on any verification failure.
    """
    import stripe  # noqa: PLC0415 — optional dependency, only when configured

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, secret)
    except Exception as exc:  # SignatureVerificationError, ValueError, ...
        raise InvalidStripeSignature(str(exc)) from exc
    # construct_event returns a StripeObject; normalize to a plain dict.
    return dict(event)


def _clerk_id_from_checkout(obj: dict) -> str | None:
    return obj.get("client_reference_id") or (obj.get("metadata") or {}).get(
        "clerk_user_id"
    )


def _clerk_id_from_invoice(obj: dict) -> str | None:
    # Stripe expands customer metadata onto the invoice's customer object when
    # requested; fall back to top-level metadata.
    meta = (obj.get("metadata") or {})
    if "clerk_user_id" in meta:
        return meta["clerk_user_id"]
    customer = obj.get("customer_details") or {}
    return (customer.get("metadata") or {}).get("clerk_user_id")


def apply_stripe_event(event: dict, store: BillingStore) -> bool:
    """Apply a parsed Stripe event to the billing store.

    Returns True if the event was handled (state changed), False if ignored.
    """
    event_type = event.get("type", "")
    obj = ((event.get("data") or {}).get("object")) or {}

    if event_type == "checkout.session.completed":
        clerk_id = _clerk_id_from_checkout(obj)
        customer_id = obj.get("customer")
        if not clerk_id or not customer_id:
            log.warning(
                "stripe_webhook.checkout.missing_ids",
                clerk_id=clerk_id,
                customer_id=customer_id,
            )
            return False
        store.set_billing_ready(clerk_id, customer_id)
        log.info("stripe_webhook.billing_ready", clerk_id=clerk_id)
        return True

    if event_type == "invoice.paid":
        clerk_id = _clerk_id_from_invoice(obj)
        if not clerk_id:
            log.warning("stripe_webhook.invoice.missing_clerk_id")
            return False
        store.reset_monthly_usage(clerk_id)
        log.info("stripe_webhook.usage_reset", clerk_id=clerk_id)
        return True

    log.info("stripe_webhook.ignored", event_type=event_type)
    return False
