-- Billing schema for Rehearse metered credit billing.
-- Apply against the Postgres pointed to by DATABASE_URL:
--     psql "$DATABASE_URL" -f rehearse/billing/schema.sql
--
-- Stripe remains the billing source-of-truth. These tables power the
-- pre-session gate (token_server.py), a future balance/usage UI, and
-- reconciliation against Stripe meter events.

CREATE TABLE IF NOT EXISTS users (
    clerk_id             TEXT PRIMARY KEY,
    stripe_customer_id   TEXT,
    -- Flipped true by the Stripe webhook once a payment method is on file.
    billing_ready        BOOLEAN NOT NULL DEFAULT FALSE,
    -- Running post-pay tally checked against a monthly ceiling in the gate;
    -- the webhook resets this each billing cycle.
    monthly_credits_used DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS usage_events (
    -- PK on session_id makes record_usage idempotent: a retried finalize
    -- cannot double-bill.
    session_id  TEXT PRIMARY KEY,
    clerk_id    TEXT NOT NULL REFERENCES users(clerk_id),
    gpu_seconds DOUBLE PRECISION NOT NULL,
    credits     DOUBLE PRECISION NOT NULL,
    reported_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS usage_events_clerk_id_idx
    ON usage_events (clerk_id, reported_at);
