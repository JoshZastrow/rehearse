```
 ██████╗ ███████╗██╗  ██╗███████╗ █████╗ ██████╗ ███████╗███████╗
 ██╔══██╗██╔════╝██║  ██║██╔════╝██╔══██╗██╔══██╗██╔════╝██╔════╝
 ██████╔╝█████╗  ███████║█████╗  ███████║██████╔╝███████╗█████╗  
 ██╔══██╗██╔══╝  ██╔══██║██╔══╝  ██╔══██║██╔══██╗╚════██║██╔══╝  
 ██║  ██║███████╗██║  ██║███████╗██║  ██║██║  ██║███████║███████╗
 ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝
```

<p align="center"><b>A research prototype for training and evaluating long-horizon conversational voice agents.</b></p>

---

## What This Is

Rehearse is built for interactive, conversational support. How well can a voice agent hold a coherent, emotionally attuned conversation across long conversations and multiple sessions?

A caller brings a hard conversation they have had or need to have and works through it on the phone. One continuous call: intake, practice, feedback. 

## The Research Problem

Most conversational AI evaluation is turn-level. Give the model an input, check the output. Did it say the right thing?

Long-horizon evaluation asks something harder: can a voice agent stay behaviorally coherent across a structured, emotionally loaded conversation it did not fully control?

Rehearse fixes the session structure to make this measurable. Three phases. One continuous call. No reset between them.

| Phase | Duration | What the agent must do |
|---|---|---|
| Intake | ~1 min | Elicit a specific, emotionally anchored goal through active listening. Not a topic — a stakes-laden moment. |
| Practice | ~3 min | Hold a realistic counterparty role: push back, hold silence, reflect incongruence between words and affect |
| Feedback | ~1 min | Name the moment the caller's voice started to mean it. Not generic encouragement — specific acoustic evidence. |

## Capability Focus

**Naturalness at scale.** Turn-based scaffolding flattens prosody over long sessions. Pauses become mechanical. Response latency becomes predictable. The capability goal is a conversation that stays sonically natural across the full arc. Progress is measured against human baseline recordings using blind listener ratings and prosodic feature distributions.

**Persona stability across phase transitions.** The agent plays three distinct roles in one call. Role drift is detectable in the transcript and in audio. The capability goal is consistent role behavior across all phase boundaries, validated by the eval harness scoring transition quality. There is currently no  detection, emotion, or persona steering. This can drift over long contexts.

**Outcome signal.** The ground truth is behavioral change in the real conversation the caller was preparing for. That signal arrives days later, if at all. The harder question is how to benchmark real-time interaction ability in real-world use cases at all. The current eval dimensions — affect perception, silence management, delivery — are a first approximation. We think the open research community is well-positioned to contribute independent, fair benchmarks here. New scorers, richer scenario datasets, and alternative judge implementations are the highest-value contributions to this repo.

## Why Prosody

Tone matters. Prosody is the signal that text evaluation cannot access. A caller can say "I'm ready" with a voice that says otherwise. An agent that responds to words alone will miss the incongruence — and the caller will feel it, even if they cannot articulate why.

Rehearse currently uses audio-native judges (Gemini 2.5 flash, which processes speech directly) to evaluate dimensions that have no text proxy:

- **Affect perception** — did the agent register the emotional state in the voice, not just the semantic content?
- **Silence management** — did the agent hold space when the caller needed it, or fill every gap?
- **Speech rate** — was the agent's pace calibrated to the conversational moment?

These focus on measuring whether the conversation worked for the caller.

## Current Architecture

The initial design is a scaffolded, multi-model system — purpose-built to prove out the conversational agent interface before committing to model infrastructure. The roadmap is to replace off-the-shelf components with self-hosted models as the interface and eval harness mature.

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

## Project Structure

Top-level directories:

| Directory | Purpose |
|---|---|
| `rehearse/` | Core Python package |
| `tests/` | Unit and integration tests |
| `evals/` | Eval datasets, fixtures, and run artifacts |
| `scripts/` | Operational scripts (serving, diagnostics, scenario generation) |
| `infra/` | Deployment and infrastructure configuration |
| `web/` | Frontend assets |
| `docs/` | Specs, plans, and architecture documents |
| `dev/` | Local development tooling and lab configs |
| `train/` | ML training pipeline (annotation, dataset prep) |

Within `train/`:

```
train/
└── pipeline/
    ├── schemas.py    # Pydantic models for session input and annotation output
    ├── diarize.py    # Speaker diarization (pyannote) on Modal GPU
    ├── annotate.py   # Whisper word-level annotation on Modal GPU
    ├── prepare.py    # Stereo WAV preparation for moshi-finetune
    └── dataset.py    # Build training manifest (JSONL) from annotated sessions
```

Within `rehearse/`:

```
rehearse/
├── types.py            # Domain types and Pydantic models (widely imported)
├── bus.py              # FrameBus — in-process async event bus
├── frames.py           # Frame types published onto the bus
├── config.py           # RuntimeConfig loaded from environment
├── storage.py          # LocalFilesystemStore — session artifact persistence
├── pipeline.py         # Live-call assembly reference doc
│
├── session/            # Call lifecycle orchestration
│   ├── session.py      # SessionOrchestrator, SessionHandle
│   ├── conversation.py # run_session() — transport-agnostic session runner
│   ├── runtime.py      # RuntimeHost — boots one session against a transport
│   ├── finalize_sweeper.py  # Sweep stale in_progress sessions on restart
│   └── synthesis.py    # SessionSynthesizer — post-call artifact generation
│
├── phases/             # Conversation flow state machine
│   ├── phases.py       # PhaseProcessor, PhaseBudgets — phase timing and transitions
│   ├── phases_llm.py   # MeetingPhaseProcessor — LLM-driven phase detection
│   ├── intake.py       # IntakeProcessor — captures caller situation during intake
│   ├── consent.py      # ConsentGate — verbal recording-consent at call start
│   ├── outcome.py      # OutcomeProbe — post-feedback yes/no outcome capture
│   └── survey.py       # SurveyAgent — post-call satisfaction survey
│
├── memory/             # Caller memory across sessions
│   ├── memory.py       # CallerMemory protocol + implementations (Null, InMemory, Honcho)
│   └── memory_manager.py  # MemoryManager — per-turn recall and storage
│
├── api/                # HTTP layer
│   ├── app.py          # FastAPI app factory — wires routes, storage, orchestration
│   ├── telephony.py    # Twilio webhooks, outbound calls, media websocket
│   └── viewer.py       # /viewer page — renders session artifacts as HTML
│
├── agents/             # CLM agent roles and routing
│   ├── clm.py          # CLM entrypoint and route mounting
│   ├── new_clm_responder.py  # NewCLMResponder — per-turn CLM orchestration
│   ├── router.py       # AgentRouter — selects agent for each turn
│   ├── registry.py     # AgentRegistry — maps phase+intake to agent instances
│   └── roles/          # Individual agent role implementations
│
├── audio/              # Audio codecs and voice participant contracts
│   ├── participants.py # VoiceParticipant ABC and VoiceSpeaker protocol
│   ├── twilio_stream.py  # TwilioCallerParticipant and TwilioStream
│   ├── mulaw.py        # μ-law codec helpers
│   └── resample.py     # PCM resampling
│
├── backends/           # LLM and voice backend adapters
│   ├── transport.py    # RuntimeTransport — duplex transport abstraction
│   ├── pipeline.py     # PipelineBackend — local STT/TTS pipeline
│   ├── managed.py      # ManagedBackend — remote managed voice backend
│   ├── tts.py          # TTS adapter
│   └── factory.py      # Backend factory — selects backend from config
│
├── personas/           # Persona registry and prompt builders
│   ├── __init__.py     # Coach/character/feedback prompts, consent classifier, intake builder
│   ├── registry.py     # PersonaRegistry — maps intake to practice partner
│   └── souls/          # Named persona definitions
│
├── services/           # External service integrations
│   ├── hume_evi.py     # HumeEVIClient — Hume voice backend
│   ├── hume_configs.py # Hume EVI config management
│   └── memory_mcp_server.py  # MCP server exposing caller memory
│
├── transports/         # LLM API transport clients
│   ├── anthropic.py    # Anthropic streaming transport
│   └── openai_compat.py  # OpenAI-compatible streaming transport
│
├── writers/            # Session artifact writers
│   └── artifacts.py    # AudioRecorder, TranscriptWriter, ProsodyWriter, TimingWriter
│
└── eval/               # Evaluation harness
    ├── cli.py          # rehearse-eval entry point
    ├── runner.py       # Eval run orchestration
    ├── scorers/        # LLM and deterministic judges
    ├── providers/      # LLM provider adapters for eval
    ├── targets/        # Eval targets (echo, raw LLM)
    ├── environments/   # Sandbox environments (in-process, subprocess)
    ├── customers/      # Synthetic customer drivers
    └── executors/      # Task executors
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
