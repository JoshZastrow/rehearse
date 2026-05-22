```
 ██████╗ ███████╗██╗  ██╗███████╗ █████╗ ██████╗ ███████╗███████╗
 ██╔══██╗██╔════╝██║  ██║██╔════╝██╔══██╗██╔══██╗██╔════╝██╔════╝
 ██████╔╝█████╗  ███████║█████╗  ███████║██████╔╝███████╗█████╗  
 ██╔══██╗██╔══╝  ██╔══██║██╔══╝  ██╔══██║██╔══██╗╚════██║██╔══╝  
 ██║  ██║███████╗██║  ██║███████╗██║  ██║██║  ██║███████║███████╗
 ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝
```

<p align="center"><b>An experimental platform for training and evaluating long-horizon conversational agents.</b></p>

---

## What This Is

Rehearse is a research platform built around a narrow question: how well can a scaffolded voice agent hold a coherent, emotionally attuned conversation across multiple minutes and phase transitions?

The domain is interpersonal coaching. A caller names a hard conversation they need to have — with a parent, a colleague, themselves — and works through it on the phone with an AI counterparty. Three phases, one continuous call: intake, practice, feedback. The format is not incidental. Phone-only constrains the interface to audio, forcing the agent to work with prosody — pitch, pace, silence, and the gap between what is said and how it is carried — rather than retreating to text-based reasoning.

Every session is simultaneously a unit of care and a training record. The product is the coaching call. The architecture is a data-collection loop.

## The Research Problem

Standard conversational AI evaluation focuses on turn-level correctness: given this input, does the model say the right thing? Long-horizon evaluation asks a harder question: does the agent remain coherent — goal-aware, emotionally calibrated, behaviorally consistent — across an extended conversation with phase transitions and an evolving emotional arc?

Rehearse makes this problem tractable by fixing the structure. Each session has three phases with distinct behavioral requirements:

| Phase | Duration | What the agent must do |
|---|---|---|
| Intake | ~1 min | Elicit a specific, emotionally anchored goal — not a topic, a stakes-laden moment — through active listening, without leading |
| Practice | ~3 min | Hold a realistic counterparty role: push back, hold silence, reflect incongruence between words and affect |
| Feedback | ~1 min | Name the moment the caller's voice started to mean it — not generic encouragement, but specific acoustic evidence |

The challenge is phase coherence: the agent must carry emotional context and caller-specific detail from intake through practice and into feedback, across what a turn-based model experiences as a long, unstructured token sequence with no explicit memory hand-off.

## Why Prosody

Prosody is the signal that text evaluation cannot access. A caller can say "I'm ready" with a voice that says otherwise. An agent that responds to words alone will miss the incongruence — and the caller will feel it, even if they cannot articulate why.

Rehearse uses audio-native judges (Gemini 2.5 flash, which processes speech directly) to evaluate dimensions that have no text proxy:

- **Affect perception** — did the agent register the emotional state in the voice, not just the semantic content?
- **Silence management** — did the agent hold space when the caller needed it, or fill every gap?
- **Speech rate** — was the agent's pace calibrated to the conversational moment?

These are not soft metrics. They directly predict whether the call worked.

## Current Architecture

Rehearse is built on scaffolded components, not a natively interactive model. The scaffold is the experiment: the platform is studying what current scaffolded systems can and cannot do in sustained, emotionally loaded conversation.

| Layer | Component |
|---|---|
| Voice (STT + TTS + prosody) | Hume EVI |
| Coach + character brain | Claude (Sonnet for turns, Opus for synthesis) |
| Dialog management | Hume EVI turn detection + phase timer |
| Telephony | Twilio — SMS trigger, outbound call, Media Streams |
| Caller memory | Honcho (cloud or self-hosted) |
| Eval judges | Gemini 2.5 (audio-native) + Claude Opus |
| Service | FastAPI + Uvicorn |

The limitations of scaffolded turn detection are visible in the data: no proactive interjection, no response to vocal cues that don't cross an audio VAD boundary. The eval harness tracks these failure modes.

## Eval Harness

The harness is the core research artifact. It measures agent behavior across seven dimensions without requiring a live phone call. Most evals run offline against fixture audio.

**Scoring dimensions:**

| Metric | What it measures |
|---|---|
| `rwrd` | Weighted composite reward |
| `cont` | Content quality — right things said at the right moment |
| `afct` | Affect perception — response to emotional state vs. words |
| `dlvr` | Delivery — timing, naturalness, phase-appropriateness |
| `nint` | Interruption rate — how often the agent stepped on the caller |
| `slnc` | Silence after affect — did the agent hold space when warranted? |
| `spch` | Speech rate — appropriate pace for the conversational moment |

**Running evals:**

```bash
make eval-list                         # what's available

# Offline (no live APIs required)
make eval-voice-smoke                  # fixture-audio smoke test, stub judges
make eval-voice-rollout                # runtime-sandbox rollout, stub TTS

# Live (real Hume TTS + Gemini/Claude judges)
make eval-voice-smoke-live             # smoke + real TTS + Gemini judges
make eval-voice-rollout-live           # full rollout with audio judges
make eval-voice-replay-live            # score 3 real sessions, Gemini judges

# Inspect results
rehearse-eval list-runs                # per-rollout scores + audio paths
rehearse-eval list-runs --play <id>    # open audio in QuickTime
make eval-watch RUN=<run_id>           # tail scores.jsonl live
```

Each run produces `scores.jsonl` with per-turn scores, phase timings (with overrun warnings), and audio recordings. The same Pydantic types span the runtime and the eval harness — a frozen production session is replayable through any stage of the pipeline.

## Codebase

```
rehearse/
├── rehearse/
│   ├── app.py                    # FastAPI: SMS + voice webhooks
│   ├── agents/                   # coach + character responders (Claude Agent SDK)
│   ├── phases.py                 # intake / practice / feedback timing
│   ├── types.py                  # shared Pydantic contracts (runtime ↔ eval)
│   ├── memory.py                 # CallerMemory protocol + backends
│   ├── services/
│   │   ├── hume_configs.py       # EVI persona registry (config as code)
│   │   └── hume_client.py        # Hume EVI WebSocket client
│   ├── audio/                    # Twilio Media Streams bridge
│   ├── synthesis.py              # post-call feedback generation
│   └── eval/                     # eval harness (evals, datasets, scorers, environments)
├── docs/specs/                   # design specs
└── Makefile                      # all dev commands
```

## Getting Started

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- [`ngrok`](https://ngrok.com/)
- [Twilio](https://twilio.com) account with a phone number
- [Hume AI](https://hume.ai) account (EVI voice + prosody)
- [Anthropic](https://console.anthropic.com) API key

### Setup

```bash
git clone https://github.com/yourusername/rehearse.git
cd rehearse
make setup
```

Configure `.env`:

```bash
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=+1...
HUME_API_KEY=...
HUME_CONFIG_ID=...
HUME_CLM_SECRET=...
ANTHROPIC_API_KEY=...
```

**Caller memory** (optional — new callers always hear the full consent prompt):

```bash
# Option A — Honcho cloud (recommended for production)
HONCHO_API_KEY=<your-key>

# Option B — Self-hosted
make setup-honcho
HONCHO_BASE_URL=http://localhost:8001

# Option C — No memory (leave both unset)
```

**Sync Hume EVI configs** (voice, prompt, and timeouts are declared in code):

```bash
BASE_URL=https://your-ngrok-url uv run rehearse-hume sync
uv run rehearse-hume diff   # see what's out of sync before applying
```

**Start:**

```bash
make serve   # opens ngrok tunnel, syncs Hume configs, starts server
```

Text your Twilio number. Answer the call.

## Contributing

The most valuable contributions are to the eval harness: new scorers, richer scenario datasets, and alternative judge implementations — especially audio-native ones.

**Key contracts:**

| Path | What it does |
|---|---|
| `rehearse/types.py` | Pydantic contracts — runtime and eval share the same schema |
| `rehearse/eval/` | Evals, datasets, scorers, environments |
| `rehearse/agents/` | Claude Agent SDK roles (coach + character) |
| `rehearse/phases.py` | Phase timing and transitions |
| `rehearse/services/hume_configs.py` | EVI personas — declared in code, synced via CLI |

**The schema is frozen by design.** A frozen session must be replayable through any stage of the pipeline. Don't break the contract.

Open a PR when `make test` and `make eval-voice-smoke` both pass.

---

## License

MIT. See [LICENSE](LICENSE).
