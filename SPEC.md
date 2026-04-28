# rehearse — SPEC v0.1

## 0. Summary

**rehearse is a voice agent inference engine that runs a time-boxed 5-minute coaching call and captures the full multi-modal session (transcript + prosody + audio) as a structured training example.** The product is a coaching call. The architecture is an ML data-collection loop. Every session is simultaneously a unit of user value and a training record for the purpose-built voice model that replaces the off-the-shelf stack in v1.

Three supporting claims shape the design:

1. **Prosody is a first-class signal, not a byproduct.** The coach's differentiated value is detecting incongruence between what is said and how it is said. Transcript-only systems cannot do this. Every frame in the pipeline carries a transcript stream and a prosody stream in sync.
2. **The inference engine and the training data pipeline share one schema.** Production session artifacts and eval harness outputs are the same pydantic types. A frozen session can be re-run through any stage of the pipeline, and any model swap is validated against the same 100-example eval bundle. There is no separate "prod" and "training" data format.
3. **Model slots, not model choices.** v0 runs Claude (intake, feedback) + Hume EVI (practice voice). v1 swaps in fine-tuned open-weights models slot by slot, driven by eval-deltas on a held-out set. The architecture does not change when the models change.

## 1. Problem

A person needs to rehearse a specific conversation — a hard talk with a cofounder, a vulnerable ask, a pitch to a potential collaborator. They pick up the phone. A coach gets 60 seconds of context, steps aside for 3 minutes of live practice with an AI counterparty compiled from that context, then returns for 60 seconds of grounded feedback. The artifact (story, transcript, prosody, feedback) is texted back as a link. The user speaks better by speaking.

The broader thesis: people improve at hard conversations through reps with feedback. The system collects those reps and the feedback, building a dataset that trains a model specifically tuned to this task — one that eventually outperforms frontier models on emotional calibration, turn-taking, and structural diagnosis in coaching dialogue.

## 2. Scope (v0.1)

In scope:
- Single-user, single-session coaching loop over a phone call
- Full artifact capture (intake, story, transcript, prosody, audio, feedback)
- Eval harness with 10 seed examples, scaling to 100
- Pydantic-defined interfaces used identically in production and eval
- Latency and quality instrumentation at every hop

Out of scope:
- Cross-session memory / user profiles
- Fine-tuned models (v1 concern)
- Accounts, auth, billing
- Multi-user social features
- Native mobile app

## 3. Functional requirements

| ID | Requirement |
|---|---|
| F1 | User triggers a session by texting a designated number with a one-line situation description. |
| F2 | System returns a phone call to the user within 30 seconds of the trigger. |
| F3 | **Phase 1 — Intake (≤60s).** Coach voice asks for situation, stakes, counterparty, goal. Output: structured `IntakeRecord`. |
| F4 | Between phases, system compiles a `CounterpartyPersona` from intake in under 3 seconds. |
| F5 | **Phase 2 — Practice (≤180s).** Character voice (distinct from coach) plays the counterparty. User rehearses live. Transcript and prosody frames are captured throughout. |
| F6 | **Phase 3 — Feedback (≤60s).** Coach voice returns, delivers structured feedback citing specific transcript turns and prosody moments. |
| F7 | Phases transition on soft cues, not hard cuts (e.g., "let's hear how this would actually go" → character enters). |
| F8 | After hangup, system generates `story.md` (Claude) and `feedback.md` (Claude) from frozen artifacts within 60 seconds. |
| F9 | System texts the user a link to the full artifact bundle via SMS. |
| F10 | All artifacts are persisted to `sessions/{session_id}/` as pydantic-serialized records. |

## 4. Non-functional requirements

### 4.1 Latency (user-perceived)

| Metric | Budget | Measured |
|---|---|---|
| Trigger SMS → outbound call starts | ≤ 30s | end-to-end |
| User utterance end → AI first audio out (Phase 2) | **≤ 800ms p50, ≤ 1.5s p95** | speech-to-speech roundtrip |
| Intake end → character first utterance | ≤ 5s | includes persona compilation |
| Call end → artifact SMS delivered | ≤ 90s | includes story + feedback synthesis |
| Phase timer enforcement | ±3s of stated budget | wall clock |

The 800ms p50 roundtrip is the load-bearing number. Above ~1s the conversation stops feeling live and users start filling silences. Hume EVI's speech-to-speech path is the only architecture that plausibly hits it; cascade (STT→LLM→TTS) does not.

### 4.2 Quality

| Metric | Target (v0) | Source |
|---|---|---|
| Fault detection recall (feedback names injected faults) | ≥ 0.60 | eval harness, deterministic scorer |
| Fault detection precision (no hallucinated faults) | ≥ 0.80 | eval harness + LLM judge |
| Prosody citation accuracy (citations match prosody stream) | ≥ 0.90 | deterministic scorer |
| Character persona fidelity | ≥ 3.5 / 5 | LLM judge on eval set |
| Feedback usefulness (holistic) | ≥ 3.5 / 5 | LLM judge, human-validated on 10-sample gold |
| Pacing adherence | 100% of sessions within phase timeboxes + 10% grace | deterministic |

### 4.3 Reliability

| Metric | Target |
|---|---|
| Session completion rate (no crash mid-call) | ≥ 0.98 |
| Artifact persistence (all 6 files written) | 1.00 — if any fail, session marked `partial` and retained |
| Hume EVI reconnect on WebSocket drop | 1 attempt with ≤ 2s backoff |

### 4.4 Observability

Every inference call, every frame, every phase transition is logged with correlation IDs. See §12.

## 5. Inputs and outputs

### 5.1 System inputs

| Source | Format | Purpose |
|---|---|---|
| Twilio inbound SMS webhook | JSON | Trigger message |
| Twilio Media Streams | μ-law 8kHz PCM over WebSocket | User voice audio |
| Twilio voice webhook | Form-encoded | Call lifecycle events |
| Wall clock | — | Phase timers |

### 5.2 System outputs

| Target | Format | Purpose |
|---|---|---|
| Twilio Media Streams | μ-law 8kHz PCM over WebSocket | Coach + character voice audio |
| Twilio SMS (outbound) | Text + artifact URL | Post-call delivery |
| `sessions/{id}/intake.json` | pydantic `IntakeRecord` | Phase 1 structured capture |
| `sessions/{id}/story.md` | Markdown | Claude-synthesized situation summary |
| `sessions/{id}/transcript.jsonl` | pydantic `TranscriptFrame` per line | Speaker-tagged utterances |
| `sessions/{id}/prosody.jsonl` | pydantic `ProsodyFrame` per line | Per-utterance Hume emotion scores |
| `sessions/{id}/audio.wav` | 16kHz PCM | Full-call recording |
| `sessions/{id}/feedback.md` | Markdown | Claude-synthesized coach reflection |
| `sessions/{id}/session.json` | pydantic `Session` | Index record with all paths + metadata |
| `sessions/{id}/telemetry.jsonl` | pydantic `InferenceLogEntry` per line | Latency + model call trace |

The pair of streams (transcript, prosody) on the same timeline is the core unit of data. Everything downstream — eval, training, review UI — reads from this pair.

## 6. Architecture

### 6.1 Runtime topology

```
 SMS ─▶ FastAPI webhook ─▶ spawn outbound call (Twilio) 
                                       │
 User's phone ◀──── Twilio voice ◀─────┤
            audio duplex              ▼
                                Twilio Media Streams WS
                                       │
                                  TwilioStream
                                       │
                                  HumeEVIClient
                                       │
     ┌─────────────────────────────────┼────────────────────────────────┐
     │                                 │                                │
 FrameBus                    PhaseProcessor               TranscriptWriter
 (runtime fanout to          (owns phase state,           ProsodyWriter
  subscribers)               swaps persona per phase,     AudioRecorder
                             emits control frames)        TelemetryLogger
     │                                 │                                │
     └────────────── frames ──────────┴────────── frames ──────────────┘
                                       │
                                       ▼
                               session directory
                                       │
                         (post-call synthesis: Claude)
                                       │
                                       ▼
                          outbound SMS with artifact link
```

### 6.2 Frame flow (what moves through the pipeline)

- `AudioChunk` — user or assistant PCM16 audio on the runtime bus
- `TranscriptDelta` — Hume-emitted utterance text + timing
- `ProsodyEvent` — Hume-emitted emotion scores per utterance
- `PhaseSignal` — phase boundary / control signal
- `EndOfCall` — live-call termination signal

These are rehearse-owned runtime frames, not Pipecat frame types.

### 6.3 Phase state

A single `PhaseProcessor` owns phase state. It consumes wall-clock ticks and transcript markers (e.g., "the user said goodbye"), emits `PhaseTransitionFrame` and `PersonaSwitchFrame`. It is the only stateful processor in the pipeline. Tested in isolation by feeding frames and asserting frames out.

### 6.4 Voice brain

Hume EVI via an owned `HumeEVIClient`. Speech-to-speech with prosody as first-class event output. Two Hume configs: one for the coach voice, one for the character voice. The character config's system prompt is overridden per-session with the compiled `CounterpartyPersona`.

## 7. Data model

All interfaces defined in [`rehearse/types.py`](rehearse/types.py). Pydantic v2, strict mode.

Model groupings:

- **Identity & enums**: `SessionId`, `Phase`, `Speaker`, `FaultLabel`, `ModelProvider`, `RubricDimension`
- **Domain**: `IntakeRecord`, `CounterpartyPersona`, `TranscriptFrame`, `ProsodyScores`, `ProsodyFrame`, `Session`
- **Eval**: `ExampleScenario`, `SyntheticUserProfile`, `RubricScore`, `EvalRun`
- **Training**: `TrainingExample`, `PreferencePair`
- **Telemetry**: `InferenceLogEntry`, `LatencyBreakdown`, `PhaseTiming`

Production artifacts and eval outputs use the same types. A `Session` loaded from disk and a `Session` produced by the eval harness are indistinguishable downstream.

## 8. Data sources, preparation, processing

### 8.1 Data sources

| Source | Volume (v0) | Volume (target) | Notes |
|---|---|---|---|
| Live sessions (founder, then invited users) | 10–50 | 1,000+ | Gold, sparse, human-consented |
| Synthetic self-play sessions | 100–1,000 | 10,000+ | Cheap, balanced across fault taxonomy |
| Seed scenarios (hand-curated) | 10 | 100 | `evals/examples/*.json` |
| Human-labeled gold subset | 10 | 50 | `evals/gold/*.json` — ground truth for judge calibration |

### 8.2 Preparation

- **Consent and redaction.** Every live session includes an explicit verbal consent gate before Phase 1 ("this call will be recorded and used to improve the coach — say yes to continue"). Session metadata records consent. Personal names + locations are redacted from training corpora via a preprocessing pass before any training use. Raw sessions stay user-private.
- **Label enrichment.** After a session, the eval rubric is run against it using the same scorers used on synthetic data. This gives every real session a vector of quality scores — usable as weak labels for training.
- **Post-conversation outcome label.** After the real conversation the session was for, the user is asked (SMS): "did rehearsing help? what specifically?" This sparse, high-signal label anchors the critic.

### 8.3 Processing

Two pipelines consume the unified `Session` schema:

**Eval pipeline** (per-release):
```
session → rubric scorers (det + LLM judge) → RubricScore rows → aggregate → report
```

**Training data pipeline** (async, ongoing):
```
sessions  ─┬─▶ preference pair miner ─▶ PreferencePair rows ─▶ DPO corpus
           └─▶ critic training set ──▶ TrainingExample rows ─▶ SFT corpus
```

Preference pairs are generated three ways (per §8.4 of the training plan, deferred to v1):

1. **Critic-ranked self-play**: two rollout samples from the same state, critic picks winner.
2. **Human-ranked subset**: founder hand-ranks 50 pairs for alignment grounding.
3. **Outcome-weighted**: sessions with positive outcome labels contribute higher-weight pairs.

## 9. Model selections

### 9.1 v0 inference stack

| Slot | Model | Rationale |
|---|---|---|
| Intake conversationalist | Claude Sonnet 4.6 (via Hume CLM or direct) | Strong short conversational elicitation with structured output. |
| Practice voice (character) | Hume EVI (Hume's hosted model + their TTS) | Speech-to-speech; prosody events are the product. |
| Practice voice (coach transitions) | Hume EVI (separate config, distinct voice) | F7 — distinguishable voice. |
| Story synthesis (post-P1) | Claude Sonnet 4.6 | Structured summarization. |
| Feedback synthesis (post-P3) | Claude Opus 4.7 | Higher reasoning for multi-modal analysis over transcript + prosody. |
| Judge (eval) | Claude Opus 4.7 | Rubric scoring with calibration against human gold. |

### 9.2 v1 migration path

The model slots are the migration plan. No architectural change required; each slot is swapped when eval-delta justifies it.

1. **Critic first** (feedback + judge): fine-tuned 4B model (Gemma 3 or similar) via DPO on preference pairs from §8.3. Lower stakes, denser signal, cheaper to run. Success criterion: eval scores match or beat Claude Opus on held-out set.
2. **Character second** (practice voice brain, via Hume CLM): fine-tuned 8–27B rollout model via DPO on critic-ranked preference pairs. Served behind a CLM endpoint that Hume calls for token generation while Hume handles TTS. Success criterion: character-believability score ≥ Claude baseline at >2x lower cost and lower p95 latency.
3. **Intake last**: lowest-value swap; optimization target if cost matters at scale.

### 9.3 Voice model (long-term)

The prosody-aware speech-to-speech path is the hardest slot to replace. We do not attempt to train our own voice model in v0 or v1. Hume stays. Training focus is on the LLM brain behind Hume CLM.

## 10. Evaluation

### 10.1 Rubric dimensions

| # | Dimension | Scorer | Target |
|---|---|---|---|
| 1 | Intake fidelity | LLM judge | ≥ 3.5 / 5 |
| 2 | Character persona fidelity | LLM judge | ≥ 3.5 / 5 |
| 3 | Character believability | LLM judge | ≥ 3.5 / 5 |
| 4 | Fault detection recall | Deterministic | ≥ 0.60 |
| 5 | Fault detection precision | Deterministic + LLM | ≥ 0.80 |
| 6 | Feedback groundedness (citations resolvable) | Deterministic | ≥ 0.90 |
| 7 | Pacing adherence | Deterministic | 1.00 |
| 8 | Incongruence detection (word/prosody mismatch) | Deterministic + LLM | ≥ 0.50 |
| 9 | Prosody citation accuracy | Deterministic | ≥ 0.90 |

Dimensions 4, 5, 8, 9 are the load-bearing ones — they have objective ground truth. 1, 2, 3 are calibrated against human labels on a 10-sample gold set.

### 10.2 Eval harness

```
evals/examples/*.json (10 → 100 scenarios)
        │
        ▼
  synthetic_user agent  ──emits──▶  TranscriptFrame + ProsodyFrame streams
        │                                           │
        │                                           ▼
        │                                  SimulatedTransport
        │                                           │
        │                                           ▼
        │                              build_pipeline(transport)
        │                                           │
        │                                           ▼
        │                                 session artifacts
        │                                           │
        ▼                                           ▼
  ground truth                              rubric scorers
  (injected faults,                                 │
   expected diagnosis)                              ▼
        │                                     RubricScore
        └──────────────────┬──────────────────────┘
                           ▼
                     evals/runs/{run_id}/results.jsonl
```

Three tiers of prosody in eval:
- **Tier 1 (scripted)**: synthetic user emits `ProsodyFrame` values directly per behavior profile. 80 of 100 examples. Cheap, deterministic.
- **Tier 2 (TTS → real Hume)**: synthetic user speaks via emotion-controlled TTS; real Hume produces real prosody. 20 of 100 examples. Validates Tier 1.
- **Tier 3 (human recordings)**: small library of real humans performing each fault, tagged. Calibration of Tier 1 scripts.

### 10.3 Scorers

- **Deterministic**: pacing, groundedness (citations resolve to real turns), prosody citation accuracy, fault recall/precision (fault taxonomy matches in feedback.md), incongruence detection (any prosody-referencing claim has a transcript-prosody mismatch at the cited turn).
- **LLM judge**: intake fidelity, persona fidelity, believability, holistic usefulness. Claude Opus with structured JSON rubric output.

### 10.4 Judge validation

Judge-against-gold correlation is computed per dimension on every eval run. If any dimension drops below ρ = 0.7 on the gold subset, that dimension's judge prompt is flagged for iteration before the run's scores are trusted.

### 10.5 Regression workflow

Every prompt, model, or pipeline change runs the full eval harness. A pull request is blocked if:
- Any load-bearing dimension (4, 5, 8, 9) regresses by > 0.05
- Any judge dimension regresses by > 0.3 / 5

Run metadata is diffed against the previous baseline in `evals/runs/`.

## 11. Observability

### 11.1 Latency instrumentation

Every frame of interest carries a `correlation_id = session_id`. For each user turn in Phase 2, the pipeline emits a `LatencyBreakdown`:

- `user_speech_end_ts` — last audio frame of user utterance
- `first_model_token_ts` — first token from the LLM
- `first_tts_audio_ts` — first audio frame out
- `roundtrip_ms` = `first_tts_audio_ts - user_speech_end_ts`

These land in `sessions/{id}/telemetry.jsonl` as `InferenceLogEntry` records and are aggregated per session. p50 and p95 per phase are the headline numbers on the dashboard.

### 11.2 Quality instrumentation

Every model call logs:
- provider, model name
- prompt token count
- completion token count
- stop reason
- wall-clock latency
- session_id + phase

Structured logs via `structlog`, JSONL to stdout (for service log aggregators) and to `telemetry.jsonl`.

### 11.3 Training-data quality

A nightly (or per-session) job runs over the session store and emits:
- Count of sessions with all 6 artifacts present
- Count of sessions with user consent flag set
- Fault distribution across sessions (are we over-represented in one category?)
- Prosody score distribution anomalies (flatline indicators for failed prosody capture)

Output: `evals/data_health/{date}.json`.

### 11.4 Dashboards

v0: a single Markdown report generated per eval run. No real-time dashboard. Good enough for prototype.

v1 (when volume justifies): Grafana with Prometheus metrics exported from the FastAPI app.

## 12. Repository layout

```
rehearse/
├── pyproject.toml
├── README.md
├── SPEC.md                        # this file
├── .env.example
├── .gitignore
├── rehearse/                      # application package
│   ├── __init__.py
│   ├── types.py                   # all pydantic interfaces (THE contract)
│   ├── app.py                     # FastAPI: Twilio webhooks + Media Streams WS
│   ├── config.py                  # env-backed settings
│   ├── session.py                 # Session artifact I/O (load/persist)
│   ├── pipeline.py                # runtime wiring entrypoint
│   ├── phases.py                  # PhaseProcessor — the one stateful custom processor
│   ├── personas.py                # coach prompt (const) + compile_character(intake)
│   ├── synthesis.py               # post-call Claude: story + feedback (pure, replayable)
│   └── eval/
│       ├── __init__.py
│       ├── harness.py             # run examples → artifacts → scores
│       ├── synthetic_user.py      # emits (transcript, prosody) frame pairs
│       ├── prosody_scripts.py     # behavior → Hume emotion trajectories
│       ├── tts_bridge.py          # tier-2: text+emotion → audio → Hume
│       ├── judge.py               # LLM judge prompts + parsing
│       └── scorers.py             # deterministic scorers
├── tests/
│   ├── __init__.py
│   ├── test_types.py              # round-trip pydantic contracts
│   ├── test_phases.py             # frame-in, frame-out on PhaseProcessor
│   ├── test_personas.py           # character compiler
│   ├── test_synthesis.py          # replay post-call on frozen fixture
│   └── eval/
│       ├── __init__.py
│       └── test_scorers.py        # deterministic scorer correctness
├── evals/
│   ├── examples/                  # scenario JSONs (10 → 100)
│   ├── gold/                      # human-labeled subset
│   └── runs/                      # per-run outputs (gitignored)
├── web/
│   └── viewer.html                # static artifact viewer
└── sessions/                      # runtime session store (gitignored)
```

## 13. Open questions / deferred decisions

| # | Question | Needed by |
|---|---|---|
| Q1 | Hume EVI config switching vs. two separate WebSockets for coach/character voice swap — which has lower latency on swap? | Pipeline build (week 1) |
| Q2 | Twilio call recording vs. in-pipeline audio capture — which gives us cleaner `audio.wav` artifacts for training? | First live session |
| Q3 | Consent phrasing and legal record-keeping for session storage. | First non-founder user |
| Q4 | Gold set labeling process — who labels, what rubric, how often re-labeled? | First eval run |
| Q5 | Fault taxonomy final list — currently ~25 items across pure-transcript, prosody-only, mixed. | Eval scorer implementation |
| Q6 | Preference pair mining heuristics for v1 training — deferred. | v1 work |

## 14. Out-of-band design commitments

These are not sections but properties the whole system must hold:

- **Single schema everywhere.** Production sessions and eval runs produce the same pydantic types. If a field exists in one and not the other, that is a bug.
- **Pure functions at stage boundaries.** `compile_character(intake) → persona`, `synthesize_story(intake, transcript) → story`, `synthesize_feedback(session) → feedback`. All deterministic modulo the model temperature. All replayable on frozen inputs.
- **Frames are the only runtime coupling.** Processors know nothing about each other; they know only the frame types they consume and emit.
- **Correlation by session_id, always.** Every log line, every artifact, every eval row carries session_id.
