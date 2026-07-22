# Consumer-facing setup — Google auth, credit billing, hosting

This document is the provisioning checklist for taking the browser call product
(`web/livekit/app` → LiveKit → Moshi on Modal) from operator-only to real users.
The **code** for auth, metered billing, and the hardened token gate already
lands in the repo (see "Code map" below). What remains is provisioning external
accounts and wiring their secrets. Nothing here runs during the hermetic test
suite — all external calls are behind env-guarded lazy imports.

## Code map (already implemented)

| Concern | Location |
|---|---|
| GPU-cost → credit formula | `rehearse/billing/cost.py` (`session_credits`) |
| Billing store (users + usage_events) | `rehearse/billing/store.py`, `schema.sql` |
| Stripe meter reporter | `rehearse/billing/stripe_meter.py` |
| Stripe webhook effects | `rehearse/billing/webhook.py` |
| Clerk JWT verification | `rehearse/auth/clerk.py` |
| Per-session room naming | `rehearse/auth/rooms.py` |
| Token gate (auth + billing + room + CORS + rate limit + webhook route) | `web/livekit/token_server.py` |
| Finalize + metering hook | `rehearse/session/livekit_session.py` (`finalize_and_bill`), called from `web/livekit/agent/agent.py` |
| Frontend auth | `web/livekit/app/src/main.tsx`, `.../hooks/useVoiceSession.ts` |

## 1. Install the extra dependency groups

```bash
uv sync --group livekit --group billing     # backend: livekit-api, stripe, pyjwt[crypto]
cd web/livekit/app && npm install           # frontend: @clerk/clerk-react
```

## Chosen domain

The product is hosted at **`rehearse.conle.ai`** (a subdomain of the owned
`conle.ai`, on Cloudflare DNS). Clerk is set up as a **Secondary application**,
so its Frontend API lives at **`clerk.rehearse.conle.ai`** and auth emails send
from `@rehearse.conle.ai` — this isolates all Clerk DNS/email from the primary
`conle.ai` business email. No path-based hosting: Clerk requires a subdomain.

## 2. Clerk (Google login) — DONE

Development instance uses Clerk's shared Google credentials; **production requires
your own Google OAuth credentials**.

1. Enable **Google** under *Configure → SSO Connections* (dashboard only, no code).
2. **Google Cloud Console** (production):
   - Verify `conle.ai` (auto-verified via Cloudflare in Google Search Console).
   - OAuth consent screen (*Google Auth Platform*): **Branding** → Authorized
     domain `conle.ai`; **Data Access** → scopes `openid`, `userinfo.email`,
     `userinfo.profile` (non-sensitive, no Google review); **Audience** → publish.
   - **Clients** → OAuth client ID → Web application:
     - Authorized JavaScript origins: `https://rehearse.conle.ai`,
       `https://clerk.rehearse.conle.ai`
     - Authorized redirect URI: `https://clerk.rehearse.conle.ai/v1/oauth_callback`
   - Copy Client ID + Client Secret.
3. Clerk → *SSO Connections → Google → Use custom credentials* → paste Client ID +
   Secret. Confirm the redirect URI matches.
4. Keys (pull with `clerk env pull --instance prod`, or Dashboard → API keys):
   - `VITE_CLERK_PUBLISHABLE_KEY` = the prod **`pk_live_…`** (non-secret) → Vercel env.
   - Backend (public URLs, non-secret) → Modal Secrets:
     - `CLERK_ISSUER=https://clerk.rehearse.conle.ai`
     - `CLERK_JWKS_URL=https://clerk.rehearse.conle.ai/.well-known/jwks.json`
   - `CLERK_SECRET_KEY` (`sk_live_…`) is **not** needed by the token gate (JWKS
     verification only); add it to Modal Secrets only if backend Clerk API calls
     are added later.

## 2a. DNS (Cloudflare) — records to add under `conle.ai`

Cloudflare dashboard → select `conle.ai` → **DNS → Records → Add record**. All are
**CNAME**; leave Proxy status **DNS only** (grey cloud) for these:

| Purpose | Type | Name (host) | Target |
|---|---|---|---|
| App (Vercel) | CNAME | `rehearse` | value Vercel shows when you add the domain |
| Clerk Frontend API | CNAME | `clerk.rehearse` | value from Clerk's *Configure → Domains* |
| Clerk accounts portal | CNAME | `accounts.rehearse` | from Clerk |
| Clerk email | CNAME | `clkmail.rehearse` | from Clerk |
| Clerk DKIM 1 | CNAME | `clk._domainkey.rehearse` | from Clerk |
| Clerk DKIM 2 | CNAME | `clk2._domainkey.rehearse` | from Clerk |

Exact targets are shown in the Vercel "add domain" panel and Clerk's Domains page —
paste them verbatim. The Google Search Console `google-site-verification` TXT
record (auto-added via Cloudflare) coexists fine; don't remove it.

## 3. Postgres (billing store)

Provision a Postgres (Vercel Postgres / Neon / any) and apply the schema:

```bash
psql "$DATABASE_URL" -f rehearse/billing/schema.sql
```

Set `DATABASE_URL` on the backend. Leave it unset locally to use the in-memory
store (`build_billing_store`).

## 4. Stripe (usage-based / metered billing)

1. Create a **Meter** with event name `rehearse_credits` (matches
   `stripe_meter.METER_EVENT_NAME`).
2. Create a **metered Price** at **$0.01 / credit** on that meter
   (matches `cost.USD_PER_CREDIT`).
3. Build a **Checkout / Customer Portal** flow to capture a payment method. Set
   the Checkout Session's `client_reference_id` to the Clerk user id so the
   webhook can map the customer back to the user.
4. Create a **webhook endpoint** pointing at `POST /api/stripe/webhook`,
   subscribed to `checkout.session.completed` and `invoice.paid`. Copy the
   signing secret → `STRIPE_WEBHOOK_SECRET`.
5. Set `STRIPE_SECRET_KEY` on the backend (the meter reporter is a logging no-op
   without it).

## 5. LiveKit Cloud

Create a project; use its `wss://` URL and API key/secret:
`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`. This replaces
`livekit-server --dev`.

## 6. Deploy

- **Modal** — deploy the Moshi backend (`make deploy-interactive`), and deploy
  the token server as a `@modal.asgi_app()` wrapping `token_server.app`, and run
  the agent as a Modal function joining the per-session room. Put all secrets
  (Clerk, Stripe, LiveKit, `DATABASE_URL`) in **Modal Secrets**, not `.env`.
- **Vercel** — project **root directory `web/livekit/app`**, framework Vite,
  build `npm run build`, output `dist`. Add domain **`rehearse.conle.ai`** (Vercel
  prints the Cloudflare CNAME to add; see §2a). Env:
  `VITE_LIVEKIT_URL` (LiveKit Cloud wss), `VITE_TOKEN_ENDPOINT` (Modal token
  server URL), `VITE_CLERK_PUBLISHABLE_KEY` (prod `pk_live_…`). Set
  `ALLOWED_ORIGINS=https://rehearse.conle.ai` on the token server.
  Note: the frontend alone renders the Clerk sign-in gate, but starting a **call**
  requires the backend (Modal token server + LiveKit Cloud) to be deployed too.

## Environment variable summary

| Var | Where | Purpose |
|---|---|---|
| `VITE_CLERK_PUBLISHABLE_KEY` | frontend | Clerk client |
| `VITE_LIVEKIT_URL` | frontend | LiveKit Cloud wss (fallback; token server also returns it) |
| `VITE_TOKEN_ENDPOINT` | frontend | token server URL |
| `CLERK_JWKS_URL`, `CLERK_ISSUER` | token server | verify Clerk JWTs |
| `LIVEKIT_API_KEY/SECRET`, `LIVEKIT_URL` | token server + agent | sign + return LiveKit tokens |
| `DATABASE_URL` | token server + agent | billing store (unset → in-memory) |
| `ALLOWED_ORIGINS` | token server | CORS allowlist (comma-separated) |
| `MONTHLY_CREDIT_CEILING` | token server | per-user post-pay cap (default 5000 credits) |
| `TOKEN_TTL_SECONDS` | token server | LiveKit token TTL (default 900) |
| `RATE_LIMIT_PER_MINUTE` | token server | per-user token requests/min (default 6) |
| `STRIPE_SECRET_KEY` | agent | report meter events (unset → no-op) |
| `STRIPE_WEBHOOK_SECRET` | token server | verify webhook signatures |

## Rate / margin notes

- A10G is billed at `A10G_USD_PER_HOUR` (`cost.py`) — **confirm at
  modal.com/pricing at build time**; it is the one constant that moves.
- Billing charges active session wall-clock only; `infra/interactive.py`
  `scaledown_window` was lowered to 5 min so post-call idle-GPU burn stays inside
  the 70% margin.
- Guardrails against post-pay runaway: the 30-min Modal `timeout` (hard session
  cap), the per-user `MONTHLY_CREDIT_CEILING`, and required payment-method-on-file
  (`billing_ready`).
