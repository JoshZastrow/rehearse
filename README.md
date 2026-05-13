```
 ██████╗ ███████╗██╗  ██╗███████╗ █████╗ ██████╗ ███████╗███████╗
 ██╔══██╗██╔════╝██║  ██║██╔════╝██╔══██╗██╔══██╗██╔════╝██╔════╝
 ██████╔╝█████╗  ███████║█████╗  ███████║██████╔╝███████╗█████╗  
 ██╔══██╗██╔══╝  ██╔══██║██╔══╝  ██╔══██║██╔══██╗╚════██║██╔══╝  
 ██║  ██║███████╗██║  ██║███████╗██║  ██║██║  ██║███████║███████╗
 ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝
```

<p align="center"><b>Meta-conversations to help you with the ones that matter.</b></p>

---

Rehearse is a relational support call. Not a chatbot, not a journal prompt — a real phone call that helps you navigate your relationship with life.

The conversations that shape you are rarely the ones you feel ready for. A hard talk with a parent. A pitch you keep postponing. An honest moment with yourself about what you actually want. Rehearse gives you a place to try them first — five minutes on the phone with an AI counterparty who listens for what your voice is doing, not just what your words are saying.

**One call, three movements:**

| Phase | Duration | What happens |
|---|---|---|
| Intake | ~1 min | You name the conversation you need to have and why it matters |
| Practice | ~3 min | An AI counterparty holds the other side — pushes back, holds silence, reflects incongruence |
| Feedback | ~1 min | You hear what shifted in your voice: pace, certainty, the moment you meant it |

Every session is simultaneously a unit of care and a training record. The architecture is an ML data-collection loop; the product is a coaching call. Prosody is the signal — the gap between what you say and how your voice carries it.

---

## Getting Started

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) — fast Python package manager
- [`ngrok`](https://ngrok.com/) — tunnel for Twilio webhooks
- A [Twilio](https://twilio.com) account with a phone number
- A [Hume AI](https://hume.ai) account (EVI voice + prosody)
- An [Anthropic](https://console.anthropic.com) API key (coach brain)

---

## Setup

### 1. Install dependencies

```bash
git clone https://github.com/yourusername/rehearse.git
cd rehearse
make setup
```

This installs Python dependencies and creates a `.env` file from the template.

### 2. Configure your environment

```bash
# Edit .env with your API keys:
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=+1...   # E.164 format
HUME_API_KEY=...
HUME_CONFIG_ID=...
HUME_CLM_SECRET=...
ANTHROPIC_API_KEY=...
```

### 3. Set up caller memory

Returning callers hear a one-sentence reminder instead of the full consent prompt. New callers always hear the full prompt. Choose one:

**Option A — Honcho cloud (recommended for production)**

```bash
# Add to .env:
HONCHO_API_KEY=<your-key>
```

**Option B — Self-hosted (no cloud account needed)**

```bash
make setup-honcho
# Add to .env:
HONCHO_BASE_URL=http://localhost:8001
```

**Option C — No memory**

Leave both unset. Calls still work; every caller hears the full consent prompt.

### 4. Sync Hume EVI configs

Voice, greeting, prompt, and turn-detection settings are declared in code (`rehearse/services/hume_configs.py`) and synced to the live Hume workspace in one command:

```bash
BASE_URL=https://your-ngrok-url uv run rehearse-hume sync
```

Run `uv run rehearse-hume diff` anytime to see what's out of sync before applying changes.

### 5. Start everything

```bash
make serve
```

This opens an ngrok tunnel, syncs Hume configs, and starts the rehearse server. If `lib/honcho/` exists, it also starts local Honcho with embedded Postgres — no separate process to manage.

---

## Your First Call

Once `make serve` is running:

1. **Text your Twilio number** — any SMS triggers an outbound call to that number
2. **Answer** — you'll hear the intake prompt
3. **Name the conversation** you need to have ("I need to tell my sister I can't keep covering for her")
4. **Practice** — the AI counterparty holds the other side for ~3 minutes
5. **Listen to your feedback** — a one-minute reflection on what your voice revealed
6. **Check your transcript** — a viewer link is sent back to you by SMS after the call

That's it. No app to download. No account to create. Just a phone call.

---

## Running Evals

The eval harness measures voice quality, coach behavior, and rollout readiness without requiring a live phone call. Most evals run offline with `ANTHROPIC_API_KEY` only.

**List what's available:**

```bash
make eval-list
```

**Offline evals (free, no live APIs):**

```bash
make eval-voice-smoke        # fixture-audio smoke test with stub judges
make eval-voice-rollout      # runtime-sandbox rollout with stub TTS
```

**Live evals (uses real Hume TTS and Gemini/Claude judges):**

```bash
make eval-voice-smoke-live         # fixture smoke + real TTS + Gemini judges
make eval-voice-rollout-live       # full rollout with audio judges
make eval-voice-rollout-audio      # routes through live EVI (real voice pipeline)
```

**Score production sessions:**

```bash
make eval-voice-replay             # score 3 real sessions with stub judges
make eval-voice-replay-live        # score 3 real sessions with Gemini judges
```

**Watch a run in progress:**

```bash
make eval-watch RUN=<run_id>       # tail scores.jsonl with live aggregate
```

**Full test suite:**

```bash
make test     # pytest
make lint     # ruff
```

---

## Contributing

```bash
git clone https://github.com/yourusername/rehearse.git
cd rehearse
make setup
make test     # confirm everything passes
```

**Key files to know:**

| Path | What it does |
|---|---|
| `rehearse/app.py` | FastAPI entry — SMS/voice webhooks |
| `rehearse/agents/` | Claude Agent SDK roles (coach + character) |
| `rehearse/services/hume_configs.py` | All EVI persona config — declared in code |
| `rehearse/types.py` | Pydantic contracts shared by runtime and eval harness |
| `rehearse/eval/` | Eval harness (evals, datasets, scorers, environments) |
| `rehearse/memory.py` | `CallerMemory` protocol + Honcho and null backends |
| `scripts/serve.sh` | Orchestrates ngrok + Honcho + server startup |

**Hume EVI configs are code.** When you change a persona's voice, prompt, or timeouts in `rehearse/services/hume_configs.py`, run `uv run rehearse-hume sync` to apply it. The workspace stays in lockstep with the repo.

**The schema is frozen by design.** Production session artifacts and eval-harness outputs share the same Pydantic types (`rehearse/types.py`). A frozen session is replayable through any stage of the pipeline. Don't break the contract.

**Every changed line should trace to the request.** Don't improve adjacent code, refactor things that aren't broken, or add features that weren't asked for. Keep it surgical.

Open a PR when tests pass and `make eval-voice-smoke` is green.

---

## Architecture

```
rehearse/
├── rehearse/                     # application package
│   ├── app.py                    # FastAPI: SMS + voice webhooks
│   ├── agents/                   # coach + character responders (Claude Agent SDK)
│   ├── types.py                  # shared Pydantic contracts
│   ├── memory.py                 # CallerMemory protocol + backends
│   ├── services/
│   │   ├── hume_configs.py       # EVI persona registry (config as code)
│   │   └── hume_client.py        # Hume EVI WebSocket client
│   ├── audio/                    # Twilio Media Streams bridge
│   ├── phases.py                 # intake / practice / feedback timing
│   ├── synthesis.py              # post-call story + feedback generation
│   └── eval/                     # eval harness (evals, datasets, scorers)
├── scripts/
│   ├── serve.sh                  # startup orchestrator
│   └── honcho_serve.sh           # self-hosted Honcho + embedded Postgres
├── docs/specs/                   # design specs (frozen — do not edit)
├── web/viewer.html               # static session artifact viewer
├── SPEC.md                       # foundational design
└── Makefile                      # all dev commands
```

**Stack:**

| Layer | What |
|---|---|
| Voice (STT + TTS + prosody) | Hume EVI |
| Coach + character brain | Claude (Sonnet for turns, Opus for feedback) |
| Telephony | Twilio — SMS trigger, outbound call, Media Streams |
| Caller memory | Honcho (cloud or self-hosted) |
| Service | FastAPI + Uvicorn |
| Eval judges | Gemini 2.5 (audio-native) + Claude Opus |

---

## License

MIT. See [LICENSE](LICENSE).
