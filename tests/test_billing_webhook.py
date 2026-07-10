"""Hermetic tests for Stripe webhook effect application (no Stripe SDK calls).

Exercises `apply_stripe_event` against an `InMemoryBillingStore` with
already-parsed event dicts, so signature verification (the network/SDK edge) is
out of scope here.
"""

from __future__ import annotations

from rehearse.billing.store import InMemoryBillingStore
from rehearse.billing.webhook import apply_stripe_event


def _checkout_event(clerk_id: str | None, customer: str | None) -> dict:
    return {
        "type": "checkout.session.completed",
        "data": {"object": {"client_reference_id": clerk_id, "customer": customer}},
    }


def test_checkout_completed_marks_billing_ready() -> None:
    store = InMemoryBillingStore()
    handled = apply_stripe_event(_checkout_event("user_1", "cus_1"), store)
    assert handled
    user = store.get_user("user_1")
    assert user is not None
    assert user.billing_ready is True
    assert user.stripe_customer_id == "cus_1"


def test_checkout_missing_ids_is_ignored() -> None:
    store = InMemoryBillingStore()
    assert apply_stripe_event(_checkout_event(None, "cus_1"), store) is False
    assert apply_stripe_event(_checkout_event("user_1", None), store) is False
    assert store.get_user("user_1") is None


def test_invoice_paid_resets_monthly_usage() -> None:
    store = InMemoryBillingStore()
    store.upsert_user("user_2", billing_ready=True, monthly_credits_used=42.0)
    event = {
        "type": "invoice.paid",
        "data": {"object": {"metadata": {"clerk_user_id": "user_2"}}},
    }
    assert apply_stripe_event(event, store) is True
    user = store.get_user("user_2")
    assert user is not None
    assert user.monthly_credits_used == 0.0
    assert user.billing_ready is True  # unchanged


def test_unhandled_event_is_ignored() -> None:
    store = InMemoryBillingStore()
    event = {"type": "customer.updated", "data": {"object": {}}}
    assert apply_stripe_event(event, store) is False


def test_billing_ready_unblocks_the_gate_end_to_end() -> None:
    """A checkout event flips the flag the token gate checks."""
    store = InMemoryBillingStore()
    assert store.get_user("user_3") is None  # gate would 402
    apply_stripe_event(_checkout_event("user_3", "cus_3"), store)
    assert store.get_user("user_3").billing_ready is True  # gate now passes
