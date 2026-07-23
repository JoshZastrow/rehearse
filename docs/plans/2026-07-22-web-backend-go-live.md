# Web backend go-live runbook

**Goal:** deploy the consumer web voice app to production (`rehearse.conle.ai`) and
verify a real user can load the page and have a spoken conversation with the AI —
cheaply (GPU + CPU both scale to zero).

This is a **next-steps runbook**, not a design doc. Each stage has a verification
gate; if a stage's check fails, that's where to look. This is the first deploy of
the new `infra/web.py` artifacts, so expect one or two iterations (flagged with ⚠️).

## Architecture recap

```
Browser (Vercel, rehearse.conle.ai)
  → Clerk sign-in
  → GET token_server /api/livekit/token   (Modal asgi, CPU, scale-to-zero)
        · auth + billing gate
        · mints a per-session LiveKit room  rehearse-<clerk_id>-<uuid>
        · spawns serve_room_job(room)        (Modal function, CPU, per-call)
  → room.connect() to LiveKit Cloud
  → serve_room_job joins the SAME room, opens the model socket
        · INTERACTIVE_PROVIDER_ENDPOINT      (Modal GPU, scale-to-zero, ~60s cold)
        · emits provider_ready when the model is up → UI leaves "warming"
  → transcript + audio persisted to the rehearse-sessions Volume
```

Only the model is GPU. The token server and per-call agent are cheap CPU
containers that scale to zero. Nothing is always-on.

## Prerequisites (should already exist)

- Modal workspace (`modal token` configured locally).
- LiveKit Cloud project → `LIVEKIT_URL` (wss), API key + secret.
- Clerk app on `clerk.rehearse.conle.ai` → JWKS URL, issuer, `pk_live_…`.
- Stripe account → secret key + webhook signing secret.
- Postgres (e.g. Neon) → `DATABASE_URL`.
- Vercel project rooted at `web/livekit/app`, domain `rehearse.conle.ai`.

## Stage 0 — Land the code

The changes live on the `consumer-facing-auth-billing` line. Merge/rebase this PR
so Vercel builds the frontend changes (the "warming" state) and the deploy uses
the current `infra/web.py`.

- **Verify:** the deployed frontend build includes the warming UI; `infra/web.py`
  is present on the branch you deploy from.

## Stage 1 — Fill `envs/prod.env` and push the Modal secret

```bash
cp envs/prod.env.example envs/prod.env      # then fill in real values
```

Fill: Clerk (`CLERK_JWKS_URL`, `CLERK_ISSUER`), LiveKit (`LIVEKIT_URL`,
`LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`), Stripe (`STRIPE_SECRET_KEY`,
`STRIPE_WEBHOOK_SECRET`), `DATABASE_URL`, `ALLOWED_ORIGINS=https://rehearse.conle.ai`.
Leave `AGENT_DISPATCH_URL=spawn`. `INTERACTIVE_PROVIDER_ENDPOINT` is filled in the
next stage.

- **Verify:** no `<placeholder>` values remain in `envs/prod.env`.

## Stage 2 — Deploy the model, then the web backend

```bash
make deploy-interactive     # GPU model → prints INTERACTIVE_PROVIDER_ENDPOINT
#   paste that endpoint into envs/prod.env

make sync-secrets           # envs/prod.env → Modal secret `rehearse-web` (no dashboard)
make deploy-web             # token server + agent → prints the token-server URL
```

- **Verify (secret):** `modal secret list` shows `rehearse-web`.
- **Verify (token server):** `curl <token-server-url>/api/livekit/token` with no
  auth returns **401** (not connection-refused) — proves it's live.
- ⚠️ **Most likely first-deploy hiccup:** `make deploy-web` fails with a
  `ModuleNotFoundError` from the container image. Fix: add the missing package to
  the `uv_pip_install(...)` list in `infra/web.py`, then re-run `make deploy-web`.

## Stage 3 — Point the frontend at the backend (Vercel)

Set these Vercel env vars and redeploy the frontend:

| Var | Value |
|---|---|
| `VITE_TOKEN_ENDPOINT` | `<token-server-url>/api/livekit/token` |
| `VITE_LIVEKIT_URL` | LiveKit Cloud `wss://…` |
| `VITE_CLERK_PUBLISHABLE_KEY` | `pk_live_…` |

- **Verify:** load `rehearse.conle.ai`, open DevTools → Network → the app fetches
  `VITE_TOKEN_ENDPOINT` (not `localhost`).

## Stage 4 — Unblock billing for the test user ⚠️

The token server returns **402** unless the signed-in user has `billing_ready=True`,
which only the Stripe `checkout.session.completed` webhook flips. Either:

- **(a)** sign in as the test user and complete the real Stripe checkout, or
- **(b)** for a quick test, seed the user directly against the prod DB:
  `build_billing_store(DATABASE_URL).upsert_user("<clerk_user_id>", billing_ready=True, stripe_customer_id="cus_test")`.

- **Verify:** DevTools → `GET /api/livekit/token` returns **200** (not 402).

## Stage 5 — Live end-to-end test, with checkpoints

Start a call on `rehearse.conle.ai` and watch each layer:

| Checkpoint | Where | Expected |
|---|---|---|
| Token issued | DevTools → Network | `/api/livekit/token` → 200 |
| Warm fired | DevTools → Network | `/api/livekit/warm` on call-screen load |
| Agent spawned | Modal dashboard → `rehearse-web` | a `serve_room_job` invocation appears |
| Cold-start UX | The page | "WARMING UP…" then CONNECTED (~60s first call) |
| Conversation | Your ears | you speak → the model responds |
| Data persisted | Modal → `rehearse-sessions` volume | `/<session_id>/transcript.jsonl` **+** the audio `.pcm`/`tokens.jsonl` |

## If a checkpoint fails — where to look

- **Token 402/503** → Stage 4 (billing) or Clerk/secret config.
- **Agent never spawns** → `modal app logs rehearse-web` (token_server): did
  `serve_room_job.spawn` run? did dispatch 503?
- **Warming never flips to connected** → `serve_room_job` logs: did the model
  socket open (the `provider_ready` emit)? Or is the model cold start exceeding
  the frontend's ~100s warming timeout?
- **No audio** → LiveKit Cloud room view (are both participants present?) +
  `serve_room_job` logs.

## Known untested assumptions (validate on first run)

1. **Image dependency list** in `infra/web.py` — derived from the runtime import
   path; a lazy import may surface as `ModuleNotFound` on first deploy.
2. **Concurrent Volume writes** — two containers (GPU model + CPU agent) commit to
   the same `/mnt/sessions/<session_id>/` dir. They write different filenames and
   Modal merges file-by-file, but confirm both file sets survive after a call.
3. **Spawn dispatch + `provider_ready` timing** end to end under a real cold start.

## Follow-ups (not blocking a live test)

- **Consent/retention gate** before retaining real conversations as training data
  (this is a product/legal check, not code).
- Persisted session artifacts are lost if the agent container crashes mid-call
  (commit is on call end only).
