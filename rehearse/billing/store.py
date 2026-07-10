"""Billing data store — the pre-session gate and post-session usage ledger.

Two tables (see schema.sql):

  users(clerk_id PK, stripe_customer_id, billing_ready, monthly_credits_used)
      Powers the token-server gate: is this user allowed to start a call?
      `billing_ready` is flipped true by the Stripe webhook once a payment method
      is on file. `monthly_credits_used` is the running post-pay tally checked
      against a ceiling; the webhook resets it each billing cycle.

  usage_events(session_id PK, clerk_id, gpu_seconds, credits, reported_at)
      One row per finished session. PK on `session_id` makes `record_usage`
      idempotent for free — a retried finalize can't double-bill.

Stripe stays the billing source-of-truth; this table exists for the gate, a
future balance/usage UI, and reconciliation.

`BillingStore` is a Protocol so the token gate and metering hook can be tested
against `InMemoryBillingStore` with no database. `PostgresBillingStore` is the
production implementation over psycopg v3.

Methods are synchronous; async call sites (the FastAPI token endpoint, the agent
finalize hook) wrap them in `asyncio.to_thread(...)` so a DB round-trip never
blocks the event loop — mirroring the `asyncio.to_thread` pattern in storage.py.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol


@dataclass(frozen=True)
class UserBilling:
    """A user's billing status, as needed by the pre-session gate."""

    clerk_id: str
    stripe_customer_id: str | None
    billing_ready: bool
    monthly_credits_used: float


class BillingStore(Protocol):
    """Storage surface the token gate + metering hook depend on."""

    def get_user(self, clerk_id: str) -> UserBilling | None:
        """Return the user's billing row, or None if they have no record yet."""
        ...

    def record_usage(
        self,
        session_id: str,
        clerk_id: str,
        gpu_seconds: float,
        credits: float,
        reported_at: datetime | None = None,
    ) -> bool:
        """Insert one usage event and add its credits to the monthly tally.

        Idempotent on `session_id`: a second call with the same id is a no-op.

        Returns:
            True if a new row was inserted, False if `session_id` already existed
            (so the caller knows whether to also report the Stripe meter event).
        """
        ...

    def set_billing_ready(self, clerk_id: str, stripe_customer_id: str) -> None:
        """Mark a user ready to be billed (payment method on file).

        Called by the Stripe webhook on checkout completion. Upserts the row.
        """
        ...

    def reset_monthly_usage(self, clerk_id: str) -> None:
        """Zero the monthly credit tally (new billing cycle / invoice paid)."""
        ...


# ---------------------------------------------------------------------------
# In-memory implementation — tests + local dev (no DATABASE_URL needed)
# ---------------------------------------------------------------------------


@dataclass
class InMemoryBillingStore:
    """Thread-safe dict-backed store. Not durable; for tests and local dev."""

    users: dict[str, UserBilling] = field(default_factory=dict)
    usage: dict[str, dict[str, object]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def upsert_user(
        self,
        clerk_id: str,
        *,
        stripe_customer_id: str | None = None,
        billing_ready: bool = False,
        monthly_credits_used: float = 0.0,
    ) -> None:
        """Test/seed helper — create or replace a user's billing row."""
        with self._lock:
            self.users[clerk_id] = UserBilling(
                clerk_id=clerk_id,
                stripe_customer_id=stripe_customer_id,
                billing_ready=billing_ready,
                monthly_credits_used=monthly_credits_used,
            )

    def get_user(self, clerk_id: str) -> UserBilling | None:
        with self._lock:
            return self.users.get(clerk_id)

    def record_usage(
        self,
        session_id: str,
        clerk_id: str,
        gpu_seconds: float,
        credits: float,
        reported_at: datetime | None = None,
    ) -> bool:
        with self._lock:
            if session_id in self.usage:
                return False
            self.usage[session_id] = {
                "session_id": session_id,
                "clerk_id": clerk_id,
                "gpu_seconds": gpu_seconds,
                "credits": credits,
                "reported_at": reported_at or datetime.now(UTC),
            }
            existing = self.users.get(clerk_id)
            used = (existing.monthly_credits_used if existing else 0.0) + credits
            self.users[clerk_id] = UserBilling(
                clerk_id=clerk_id,
                stripe_customer_id=existing.stripe_customer_id if existing else None,
                billing_ready=existing.billing_ready if existing else False,
                monthly_credits_used=used,
            )
            return True

    def set_billing_ready(self, clerk_id: str, stripe_customer_id: str) -> None:
        with self._lock:
            existing = self.users.get(clerk_id)
            self.users[clerk_id] = UserBilling(
                clerk_id=clerk_id,
                stripe_customer_id=stripe_customer_id,
                billing_ready=True,
                monthly_credits_used=existing.monthly_credits_used if existing else 0.0,
            )

    def reset_monthly_usage(self, clerk_id: str) -> None:
        with self._lock:
            existing = self.users.get(clerk_id)
            if existing is None:
                return
            self.users[clerk_id] = UserBilling(
                clerk_id=clerk_id,
                stripe_customer_id=existing.stripe_customer_id,
                billing_ready=existing.billing_ready,
                monthly_credits_used=0.0,
            )


# ---------------------------------------------------------------------------
# Postgres implementation — production (psycopg v3)
# ---------------------------------------------------------------------------


class PostgresBillingStore:
    """Billing store backed by Postgres via psycopg v3.

    Opens a short-lived connection per call (no pool). Adequate for v1 low
    volume; introduce psycopg_pool if the token endpoint gets hot.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def _connect(self):
        import psycopg  # noqa: PLC0415 — optional at import time

        return psycopg.connect(self._dsn)

    def get_user(self, clerk_id: str) -> UserBilling | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT clerk_id, stripe_customer_id, billing_ready, "
                "monthly_credits_used FROM users WHERE clerk_id = %s",
                (clerk_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return UserBilling(
            clerk_id=row[0],
            stripe_customer_id=row[1],
            billing_ready=bool(row[2]),
            monthly_credits_used=float(row[3]),
        )

    def record_usage(
        self,
        session_id: str,
        clerk_id: str,
        gpu_seconds: float,
        credits: float,
        reported_at: datetime | None = None,
    ) -> bool:
        reported_at = reported_at or datetime.now(UTC)
        with self._connect() as conn, conn.cursor() as cur:
            # Ensure the user row exists first — usage_events.clerk_id has an FK
            # to users(clerk_id). No tally change here; that happens only on a
            # genuinely new usage row below.
            cur.execute(
                "INSERT INTO users (clerk_id) VALUES (%s) "
                "ON CONFLICT (clerk_id) DO NOTHING",
                (clerk_id,),
            )
            # ON CONFLICT DO NOTHING → idempotent; rowcount tells us if inserted.
            cur.execute(
                "INSERT INTO usage_events "
                "(session_id, clerk_id, gpu_seconds, credits, reported_at) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (session_id) DO NOTHING",
                (session_id, clerk_id, gpu_seconds, credits, reported_at),
            )
            inserted = cur.rowcount == 1
            if inserted:
                cur.execute(
                    "UPDATE users SET monthly_credits_used = "
                    "monthly_credits_used + %s WHERE clerk_id = %s",
                    (credits, clerk_id),
                )
            conn.commit()
        return inserted

    def set_billing_ready(self, clerk_id: str, stripe_customer_id: str) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (clerk_id, stripe_customer_id, billing_ready) "
                "VALUES (%s, %s, TRUE) "
                "ON CONFLICT (clerk_id) DO UPDATE "
                "SET stripe_customer_id = EXCLUDED.stripe_customer_id, "
                "billing_ready = TRUE",
                (clerk_id, stripe_customer_id),
            )
            conn.commit()

    def reset_monthly_usage(self, clerk_id: str) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET monthly_credits_used = 0 WHERE clerk_id = %s",
                (clerk_id,),
            )
            conn.commit()


def build_billing_store(dsn: str | None) -> BillingStore:
    """Return a Postgres store when a DSN is configured, else in-memory.

    Lets the token server and agent stay database-agnostic: set `DATABASE_URL`
    in production; leave it unset for local dev and hermetic tests.
    """
    if dsn:
        return PostgresBillingStore(dsn)
    return InMemoryBillingStore()
