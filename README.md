```
 ██████╗ ███████╗██╗  ██╗███████╗ █████╗ ██████╗ ███████╗███████╗
 ██╔══██╗██╔════╝██║  ██║██╔════╝██╔══██╗██╔══██╗██╔════╝██╔════╝
 ██████╔╝█████╗  ███████║█████╗  ███████║██████╔╝███████╗█████╗  
 ██╔══██╗██╔══╝  ██╔══██║██╔══╝  ██╔══██║██╔══██╗╚════██║██╔══╝  
 ██║  ██║███████╗██║  ██║███████╗██║  ██║██║  ██║███████║███████╗
 ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝
```

<p align="center"><b>A dialogue system for important conversations. A human rehearses the conversation they need to have against a model that plays the other side — and both improve.</b></p>

---

## What This Is

Rehearse is a prototype Full-Duplex dialogue system built into a voice assistant product, designed for conversational support.

Rehearse turns that one-shot conversation into something you can practice against. A caller brings a hard conversation they have had or need to have, and works through it on the phone with a model that plays the other side.

Both parties are learners. The **caller** rehearses the conversation itself by consolidating the handful of statements that actually land. The **provider** model rehearses being a coherent, attuned counterpart. Each session gets scored and sampled for continued training on conversation sets.

## Learning Through Conversations

Most conversational skills cannot be learned by grinding. With a human in the loop, you cannot run a thousand parallel rollouts of a hard conversation from the same starting point. There is one real take, conversations evolve. The environment (the other person, and you) never resets to the same state twice, as conversations create history and relational dynamics evolve. This is the un-grindable, non-stationary regime. 

Rehearsal is a replayable practice environment for a domain that otherwise offers none — the caller gets to run the conversation many times before the one that counts, and the model gets many scored trajectories from an interaction that, in the wild, would produce exactly one. Learning that sticks means consolidation, not recall: the caller carries a few sharpened intuitions into the real conversation, and the model distills scored sessions into its weights rather than an ever-growing context window.

## A More Natural Interface

Most conversational AI evaluation is turn-based. Give the model an input, check the output. Convert to Speech with auxiliary STT and TTS components. A turn-based cascade (STT → LLM → TTS) doesn't just add latency — it destroys the signal needed for attuned conversations. STT collapses audio to text, which throws away timing, prosody, hesitation, and silence. Many smaller model components adds infrastructure complexity that can be simplified in an encapsulated model that can improve via axes of scaling (data, compute, model size).

This project aims at implementing a time-based full-duplex audio model with a background language model that weaves in retrieved context from memory into the conversation state. This allows for both a qualitatively more natural conversation with simultaneous reasoning, tool-use, and context retrieval. 

## Current Architecture

The native architecture is a **full-duplex audio model** — a single speech-to-speech model that hears and speaks on one continuous stream, based on [PersonaPlex](https://github.com/NVIDIA/personaplex) (NVIDIA's Moshi 7B finetune with voice + role conditioning). This is the self-hosted path the whole system is built toward: no STT→LLM→TTS relay, no external voice-activity detector deciding whose turn it is. The model itself chooses when to speak, when to listen, and when to hold silence — which is exactly the behavior the eval harness scores.

**How the model runs** (`rehearse/backends/interactive/`, served from `infra/interactive.py`):

- Caller audio (PCM16 16 kHz) is resampled to 24 kHz and encoded by the **Mimi** neural codec into discrete tokens.
- The LM steps at **12.5 Hz (80 ms frames)**, emitting interleaved text tokens (the provider's transcript) and audio tokens on the same clock — this simultaneity is what makes it full-duplex rather than turn-based.
- Mimi decodes the provider's audio codebooks back to 16 kHz speech, streamed to the caller with no turn gate in between. PersonaPlex's `dep_q=16` depth transformer predicts both input- and output-stream codebooks; only the 8 output codebooks are decoded to audio.
- Persona is set by **conditioning, not prompting**: a voice prompt fixes the speaker identity and a text prompt fixes the role, both prefilled as system prompts before the stream opens. Two conditioned endpoints run the same weights — **PersonaplexProvider** (the guide) and **PersonaplexCaller** (a synthetic caller that drives evals without a live phone).
- Caller-side transcription runs faster-whisper on buffered caller audio; the inference loop lives in a thread executor bridged back to asyncio.

| Layer | Component |
|---|---|
| Speech model | **PersonaPlex 7B** — full-duplex, voice + role conditioned (upstream Moshi selectable) |
| Neural codec | Mimi — 24 kHz streaming encode/decode |
| Caller transcription | faster-whisper on buffered caller audio |
| Serving | aiohttp WebSocket on Modal **L40S GPU** |
| Telephony | Twilio — SMS trigger, outbound call, Media Streams |
| Caller memory | Honcho (cloud or self-hosted) |
| Post-session synthesis | Claude Opus (story + feedback generation) |
| Eval judges | Gemini 2.5 (audio-native) + Claude Opus |
| Service | FastAPI + Uvicorn |

**Managed fallback (no GPU required).** For environments without GPU access, a scaffolded stack is selectable via `BACKEND_TYPE=managed`: Hume EVI (STT + TTS + prosody) with Claude driving turns, dialog gated by Hume's VAD plus a phase timer. It is turn-based by construction, so its limits are visible in the data — no proactive interjection, no response to vocal cues that don't cross a turn boundary. The full-duplex model is the default specifically to remove that ceiling; the eval harness tracks both so the improvement is measurable.

## Eval Harness

The harness is the core research artifact. It measures agent behavior across seven dimensions without requiring a live phone call. Most evals run offline against fixture audio, and aim to measure audio calls.

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

## The Training Loop

The model half of co-training is a closed loop: live sessions generate scored trajectories, scored trajectories update the training data mix, and model updates improve the next session. Two improvement levers operate at different timescales.

```
[Live Session] → transcript.jsonl + prosody.jsonl + telemetry.jsonl
      ↓
[Eval Harness] → RubricScore across 7 dimensions (rwrd, cont, afct, dlvr, nint, slnc, spch)
      ↓
[Feedback Agent] → reads full trajectory + scores + improvement history
      │
      ├─→ Plateau NOT detected → HARNESS UPDATE
      │     Rewrite system prompt and persona config
      │     Loop back to live session
      │
      └─→ Plateau detected → WEIGHT UPDATE
            Dispatch LoRA training job on Modal GPU
            Loop back with adapted model
```

**Sequencing rule:** Always start with harness iteration. Scaffold improvements are faster and cheaper — tune the prompts, phase logic, and persona compiler until scores stop moving, then fine-tune weights on the validated signal. Training on a noisy scaffold amplifies the noise.

**The caller's loop.** Every rehearsal session is scored once, on one rubric, and that single scored session improves two learners at once. The model side is mechanical and lives in this repo: the scored conversation trajectory and conversation quality becomes training data, and improvement lands as a LoRA update on Modal. The caller side runs in the person: the reps plus the same rubric feedback become sharpened intuition they carry into the real conversation.

The training stack (in `train/`) implements FSDP + LoRA on Moshi 7B with manual bf16 mixed precision for sequential adaptation.

## Project Structure

```
rehearse/               # Core Python package
├── types.py            # 37+ Pydantic domain models — most-imported file in the codebase
├── config.py           # RuntimeConfig loaded from environment
├── storage.py          # Session artifact persistence
├── frames.py           # Audio/text frame types
│
├── agents/             # CLM conversation layer
│   ├── new_clm_responder.py  # Primary conversation loop — reads frames, routes turns
│   ├── clm.py          # Anthropic-SDK conversation agent with phase-aware prompting
│   ├── router.py       # Phase-aware agent dispatch
│   ├── registry.py     # Role-to-agent mapping
│   ├── timecard.py     # Phase timing enforcement
│   ├── topic.py        # Claude-backed topic classifier
│   ├── persona_router.py      # SMS → EVI persona routing
│   ├── persona_routing_agent.py  # Persona selection coordinator
│   └── roles/
│       ├── base.py     # RehearseAgent abstract base
│       ├── intake.py   # Intake role
│       ├── character.py  # Practice counterparty role
│       └── feedback.py # Feedback role
│
├── memory/             # Multi-session caller context
│   ├── memory.py       # CallerMemory — Honcho-backed + in-memory implementations
│   └── memory_manager.py  # Per-turn memory read/write orchestration
│
├── personas/           # Persona catalog and routing
│   ├── __init__.py     # Built-in persona definitions
│   ├── registry.py     # PersonaRegistry — registration and lookup
│   └── souls/          # Named persona soul documents
│
├── backends/           # Voice backend adapters
│   ├── base.py         # Backend abstract interface
│   ├── transport.py    # InMemoryTwoWayChannel — bidirectional frame exchange
│   ├── prosody.py      # Prosody feature extraction
│   └── interactive/    # Local PersonaPlex/Moshi real-time inference
│       ├── backend.py  # Thread-executor inference loop bridged to asyncio
│       ├── loader.py   # Model weight loading
│       └── asr.py      # ASR interface
│
├── transports/         # LLM API transport factory (LiteLLM / Anthropic)
├── session/
│   ├── conversation.py # ConversationBackend protocol
│   └── synthesis.py    # Post-session story/feedback generation (Claude Opus)
│
├── services/           # External service integrations
│   ├── hume_configs.py # Hume EVI configs-as-code — sync and diff
│   ├── hume_configs_cli.py  # CLI for persona config sync
│   └── memory_mcp_server.py  # MCP server exposing caller memory as tools
│
├── cli/init.py         # Interactive setup wizard (rehearse-init)
├── train/              # Training dispatch (CLI + Modal job submission)
│
└── eval/               # Evaluation harness
    ├── protocols.py    # BenchmarkExample, Scorer, Environment — shared interface contract
    ├── runner.py       # Benchmark orchestration
    ├── cli.py          # rehearse-eval entry point
    ├── benchmarks/     # Benchmark definitions (MME-emotion, etc.)
    ├── datasets/       # Eval dataset loaders
    ├── environments/   # Sandbox environments (runtime sandbox, audio fixture, production replay)
    ├── drivers/        # Synthetic caller drivers (LLM-backed, audio)
    ├── scorers/        # LLM + deterministic judges (audio-native via Gemini, Claude)
    ├── judges/         # AudioLLMProvider backends (Gemini, vLLM)
    └── harness/        # Executor, reporter, streamer, watcher

train/                  # ML training pipeline (separate package)
├── train.py            # Main training loop (FSDP + LoRA + bf16)
├── finetune/
│   ├── args.py         # TrainArgs with LoRA / full-FT mutual-exclusivity checks
│   ├── wrapped_model.py  # FSDP wrapping with regime selection
│   ├── distributed.py  # torch.distributed rank/world-size abstractions
│   ├── checkpointing.py  # FSDP checkpoint save/restore
│   ├── mixed_precision.py  # Manual master-weight bf16 precision
│   └── data/interleaver.py  # Audio+text token interleaving for Moshi
└── pipeline/
    ├── schemas.py      # Session and annotation Pydantic schemas
    ├── diarize.py      # pyannote speaker diarization on Modal GPU
    ├── annotate.py     # Whisper transcription + speaker labelling on Modal GPU
    ├── prepare.py      # Stereo WAV preparation (provider left / caller right)
    └── dataset.py      # Training manifest builder (JSONL)

infra/                  # Modal-deployed cloud services
├── interactive.py      # PersonaPlex (default) + Moshi full-duplex inference (aiohttp WebSocket, L40S/A10G GPU)
├── judge.py            # LiteLLM proxy for distributed eval scoring
└── litellm_config.yaml # Model alias routing config

evals/                  # Offline eval datasets and fixtures
scripts/                # Operational tooling (serving, diagnostics, scenario generation)
docs/                   # Specs, plans, and architecture documents
tests/                  # Unit and integration tests
```

## Getting Started

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) — fast Python package manager
- [`ngrok`](https://ngrok.com/) — tunnel for Twilio webhooks (free tier works)
- [Twilio](https://twilio.com) account with a phone number
- [Hume AI](https://hume.ai) account (EVI voice + prosody)
- [Anthropic](https://console.anthropic.com) API key

**For the interactive backend or eval judges (optional):**
- [Modal](https://modal.com) account — GPU inference and LLM judges run on Modal. Free tier is sufficient for development.

### Setup

```bash
git clone https://github.com/JoshZastrow/rehearse.git
cd rehearse
make setup       # install deps + create .env from template
uv run rehearse-init  # interactive wizard: API keys, backend, infra
```

`rehearse-init` walks through every required and optional setting and writes `.env` for you. It covers:

- Twilio, Hume, and Anthropic API keys
- Backend selection (managed / pipeline / interactive)
- Optional: Modal judge deploy (`infra/judge.py`) — needed for live evals
- Optional: Modal interactive backend deploy (`infra/interactive.py`) — full-duplex Moshi on A10G GPU
- Optional: caller memory setup (self-hosted or cloud)

Re-run any time with `--force` to update values or `--env-only` to skip deploys.

**Sync Hume EVI configs** (voice, prompt, and timeouts are declared in code):

```bash
BASE_URL=https://your-ngrok-url uv run rehearse-hume sync
uv run rehearse-hume diff   # see what's out of sync before applying
```

**Start:**

```bash
make serve   # opens ngrok tunnel, starts server (+ Honcho if self-hosted)
```

Text your Twilio number. Answer the call.

## Fine-tuning Moshi

Rehearse includes a pipeline to fine-tune [Moshi](https://github.com/kyutai-labs/moshi) on recorded session audio, producing a coach-voiced model adapted to the Rehearse conversation style.

### Prerequisites

- [Modal](https://modal.com) account with GPU access (A10G; free tier has enough credits for smoke tests)
- HuggingFace account — accept the terms for [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1) and [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0)
- A `HF_TOKEN` Modal secret: `modal secret create HF_TOKEN HF_TOKEN=<your-token>`
- Session audio in `sessions/<id>/audio.wav` (mono PCM16, 16 kHz)

### Pipeline

The pipeline runs in four stages. Each stage writes artifacts consumed by the next and can be run independently.

**0. Build initial manifest** — index all session directories into a JSONL manifest that the GPU stages consume:

```bash
uv run python train/pipeline/dataset.py \
  sessions_root=sessions/ \
  out=data/sessions.jsonl \
  push_to_volume=false \
  require_annotation=false
```

**1. Diarize** — assign speaker segments using pyannote on Modal GPU:

```bash
modal run train/pipeline/diarize.py --egs data/sessions.jsonl
```

Writes `audio_segments.json` to each session directory. Override the GPU type with `REHEARSE_GPU=A10G modal run ...`.

**2. Annotate** — Whisper word-level transcription + speaker labelling on Modal GPU:

```bash
modal run train/pipeline/annotate.py --egs data/sessions.jsonl
```

Writes `audio.json` (word alignments with `caller`/`provider` speaker labels) and automatically chains into the prepare step.

**3. Prepare** — split mono mixed recording into a stereo WAV (provider left, caller right):

```bash
uv run python train/pipeline/prepare.py egs=data/sessions.jsonl
```

Writes `audio_stereo.wav` and `audio_stereo.json` to each session directory. This step runs automatically as a post-annotation hook, so it only needs to be run manually if you skipped step 2 or need to reprocess.

**4. Rebuild manifest** — reindex sessions (now resolves to `audio_stereo.wav`) and push to the Modal Volume:

```bash
uv run python train/pipeline/dataset.py \
  sessions_root=sessions/ \
  out=data/sessions.jsonl
```

Writes `data/sessions.jsonl` (one line per session with `path` and `duration`) and syncs all audio and annotation files to the `rehearse-training` Modal Volume. Sessions without `audio_stereo.wav` fall back to `audio.wav` automatically.

### Training

Run a short smoke test (50 steps, LoRA rank 16) to verify the pipeline end-to-end:

```bash
uv run rehearse-train \
  run_dir=runs/smoke-test \
  max_steps=50 \
  batch_size=1 \
  lora_rank=16 \
  duration_sec=30
```

For a full training run, use the defaults in `rehearse/models/moshi_7B/config.yaml`:

```bash
uv run rehearse-train run_dir=runs/moshi-coach max_steps=2000
```

All training runs on a Modal A10G GPU. Checkpoints land at `/data/runs/<run_name>/checkpoints/` on the `rehearse-training` Volume. Loss and throughput are streamed to your terminal per step; add a `wandb:` block to the config for Weights & Biases logging.

**Key config parameters** (pass as CLI args to override the YAML defaults):

| Arg | Default | Description |
|---|---|---|
| `run_dir` | required | Output directory for checkpoints and logs |
| `max_steps` | 2000 | Training steps |
| `batch_size` | 16 | Examples per GPU per step — reduce if OOM |
| `lora_rank` | 128 | LoRA adapter rank |
| `duration_sec` | 100 | Max audio sequence length in seconds |
| `with_modal` | true | Set `false` to run locally with `torchrun` (requires CUDA) |

### How it works

Session audio is mono mixed (coach + caller on one channel). The pipeline:

1. **Diarizes** — pyannote identifies which time windows belong to each speaker
2. **Annotates** — Whisper provides word-level transcripts; a majority-vote over diarization segments assigns `coach` / `user` labels
3. **Prepares** — the mono recording is split into stereo: coach signal on left, user signal on right, with the opposite channel zeroed outside each speaker's diarization windows
4. **Trains** — moshi-finetune receives the stereo WAV and reads `audio_stereo.json` for word alignments; `keep_main_only=True` trains on coach turns only

The model is fine-tuned with LoRA on top of `kyutai/moshiko-pytorch-bf16`. The adapter weights (`save_adapters: true`) are saved separately so the base model is not modified.

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
