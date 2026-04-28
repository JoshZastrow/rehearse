# rehearse

Voice agent inference engine for real-time conversation coaching.

A 5-minute phone call: 1 minute intake, 3 minutes live practice with an AI counterparty, 1 minute feedback. Built as a prototype ML system — live sessions are the data source for continual improvement of a **purpose-built audio LLM** that replaces the off-the-shelf stack in v1.

The product is a coaching call. The architecture is an ML data-collection loop. Every session is simultaneously a unit of user value and a training record.

See [SPEC.md](SPEC.md) for the foundational design.
See [`docs/specs/MANIFEST.md`](docs/specs/MANIFEST.md) before committing to spec-driven work; it tracks which specs are active, WIP, done, or historical.

Scaffold + eval harness (Phases 1–2). Runtime not implemented yet.

- Eval harness: see [`rehearse/eval/README.md`](rehearse/eval/README.md). Runs against `noop` and `eq-bench` today; `rehearse-seed` and `full` targets land in Phases 3–4.
- Runtime: see the spec at [`docs/specs/v2026-04-27-runtime.md`](docs/specs/v2026-04-27-runtime.md).

## Status (2026-04-28)

| Workstream | Stage |
|---|---|
| Pydantic data contracts (`rehearse/types.py`) | ✅ frozen |
| Eval harness skeleton (Phases 1–2) | ✅ shipped |
| EQ-Bench adapter | 🗑 deprecated, scheduled for removal (text-only, stale) |
| MME-Emotion adapter + audio targets | 📝 spec'd, build pending |
| Runtime (Twilio + Pipecat + Hume + Claude Agent SDK) | 📝 spec'd, build pending |

## Strategic frame

Three load-bearing claims:

1. **Prosody is the product, not a byproduct.** The coach's value is detecting incongruence between what is said and how it is said. Transcript-only systems cannot do this.
2. **Purpose-built model, not a wrapper.** v0 uses Claude + Hume; v1 fine-tunes an open-weights audio LLM (Gemma 4 E4B) on preference pairs mined from real sessions. The architecture stays constant; the model slots swap.
3. **Single schema everywhere.** Production session artifacts and eval-harness outputs are the same pydantic types. A frozen session is replayable through any stage.

## Roadmap

Three workstreams advance in parallel. Each ships in numbered phases; each phase is one PR with green tests + a runnable demo.

### Eval harness (`rehearse-eval`) — `rehearse/eval/`

Plugin-shaped: benchmarks, targets, scorers, executors are each a `Protocol`. Independent of runtime.

| Phase | Status | Scope |
|---|---|---|
| 1 — Skeleton | ✅ shipped | Protocols, registries, runner, CLI, `LocalSubprocessExecutor`, `noop` benchmark, `echo` target. 20 tests green. |
| 2 — EQ-Bench end-to-end | ✅ shipped | EQ-Bench adapter, `raw-llm` target, Pearson correlation scorer. **Being removed** — see Phase A1. |
| A1 — Strip EQ-Bench | 🚧 next | Delete EQ-Bench adapter, sample data, tests, README sections. |
| A2 — Provider plugin layer + Gemini provider | 📝 spec'd | `AudioLLMProvider` protocol, Gemini 2.5 Pro via `google-genai`, `list-providers` CLI. |
| A3 — `multimodal-llm` target + 10-clip MME-Emotion + recognition scorer | 📝 spec'd | First audio-native eval. Hand-curated 10 clips from `ER_Lab`. Deterministic accuracy scorer. |
| A4 — Reasoning scorer (Claude Opus judge) | 📝 spec'd | LLM-judge scorer. `MMEReasoningScorer` + reusable `LLMJudge` primitive. |
| A5 — vLLM provider (Gemma 4 E4B) | 📝 spec'd | OpenAI-compatible client pointed at a self-hosted vLLM server. Gated on a live endpoint. |
| A6 — Side-by-side comparison | 📝 spec'd | `rehearse-eval diff <run_a> <run_b>` per-dimension delta table. Gemini vs Gemma on the same clips. |
| A7 — Scale to 100 clips | 🔮 future | Move past the v0 hand-curated subset; `scripts/fetch_mme_emotion.py` from HuggingFace. |
| A8 — Rehearse-seed scenarios + `synthesis` / `full` targets | 🔮 future | First product-quality eval (not capability eval). Fault-recall, persona fidelity, holistic usefulness. |
| A9 — CI gating + regression workflow | 🔮 future | PR-blocking eval delta checks. |

Spec: [`docs/specs/v2026-04-27-eval-harness.md`](docs/specs/v2026-04-27-eval-harness.md), [`docs/specs/v2026-04-28-mme-emotion-and-audio-targets.md`](docs/specs/v2026-04-28-mme-emotion-and-audio-targets.md). User-facing: [`rehearse/eval/README.md`](rehearse/eval/README.md).

### Runtime (`rehearse-app`) — `rehearse/app.py`, `rehearse/agents/`, `rehearse/pipeline.py`, ...

Single FastAPI service: SMS triggers an outbound call, runs the 3-phase Pipecat pipeline with Hume EVI for voice and the Claude Agent SDK for the brain, persists every frame to disk, then synthesizes story + feedback post-call and SMSes a viewer link back.

| Phase | Status | Scope |
|---|---|---|
| R1 — Skeleton + Telephony | 📝 spec'd | Twilio SMS/voice/Media Streams handlers; `SessionOrchestrator`; `LocalFilesystemStore`. Demo: SMS triggers a call that says hello and hangs up. |
| R2 — Pipecat + Hume EVI scaffold | 📝 spec'd | Pipeline builder, single Hume coach config, no phases yet. Demo: real Hume voice on the call. |
| R3 — Phase machine + writers | 📝 spec'd | `PhaseProcessor`, transcript/prosody/audio/telemetry writers. Demo: 3 phases, all 4 artifacts populated. |
| R4 — Claude CLM + agents | 📝 spec'd | CLM endpoint, `CoachAgent` (intake), `CharacterAgent`, `compile_character` real. Demo: real intake → compiled character → live practice. |
| R5 — Post-call synthesis + viewer | 📝 spec'd | `StoryAgent`, `FeedbackAgent` via Claude Agent SDK; viewer page; SMS notification. Demo: full SMS-to-SMS round-trip. |
| R6 — Reliability + polish | 📝 spec'd | Soft-cue phase transitions, Hume reconnect, hardened consent gate, signature validation. |
| R7 — Storage option B (S3 mirror) | 📝 spec'd | New `S3MirrorStore` backend. Migration trigger: first non-founder user invited. |

Spec: [`docs/specs/v2026-04-27-runtime.md`](docs/specs/v2026-04-27-runtime.md).

### ML data pipeline & training — future

Consumes frozen sessions from the runtime + scored runs from the harness; produces training corpora.

| Phase | Status | Scope |
|---|---|---|
| T1 — Preference pair mining heuristics | 🔮 future | Critic-ranked self-play rollouts; outcome-weighted pairs from sessions with positive `OutcomeLabel`. |
| T2 — DPO on Gemma 4 E4B | 🔮 future | First fine-tune target. Success: matches Gemini on MME-Emotion at ≥2× lower cost / lower p95 latency. |
| T3 — Critic LLM for online-eval | 🔮 future | Per-session weak labels via the same rubric. Closes the data loop. |
| T4 — Voice model fine-tune | 🔮 not pursued in v1 | Hume stays. Per SPEC §9.3, voice model training is deferred indefinitely. |

Spec: not yet written. Contingent on the runtime producing a meaningful volume of consented sessions (target: 100+).

## Stack

| Layer | v0 | v1 (eval-driven swap) |
|---|---|---|
| Voice in/out (STT + TTS + prosody events) | Hume EVI | Hume EVI (unchanged — see SPEC §9.3) |
| Practice character brain (per-turn) | Claude Sonnet 4.6 via Hume CLM | Fine-tuned Gemma 4 E4B via Hume CLM |
| Intake + feedback synthesis | Claude Agent SDK (Sonnet for intake, Opus for feedback) | Same; lower priority to swap |
| Telephony | Twilio (SMS + Voice + Media Streams) | unchanged |
| Pipeline | Pipecat | unchanged |
| Data contracts | Pydantic v2 (`rehearse/types.py`) | unchanged — single schema everywhere |
| Service | FastAPI + Uvicorn | unchanged |
| Storage | Local filesystem | S3 mirror, then Postgres+S3 (per runtime spec §5) |
| Eval — hosted baseline | Gemini 2.5 Pro (audio-native) | Gemini 2.5 Pro (still the bar to beat) |
| Eval — open-weights candidate | Gemma 4 E4B via vLLM | Same model, fine-tuned on rehearse data |
| Eval — judge | Claude Opus 4.7 | Claude Opus or self-hosted critic |

## Repo layout

```
rehearse/
├── SPEC.md                       # foundational design (frozen)
├── README.md                     # this file
├── docs/
│   └── specs/
│       ├── MANIFEST.md
│       ├── v2026-04-27-eval-harness.md
│       ├── v2026-04-27-runtime.md
│       └── v2026-04-28-mme-emotion-and-audio-targets.md
├── rehearse/                     # application package
│   ├── types.py                  # all pydantic interfaces
│   ├── eval/                     # eval harness (Phases 1–2 shipped)
│   │   └── README.md
│   ├── app.py                    # runtime FastAPI entry (spec'd)
│   ├── agents/                   # Claude Agent SDK roles (spec'd)
│   ├── pipeline.py               # Pipecat wiring (spec'd)
│   ├── phases.py                 # PhaseProcessor (spec'd)
│   ├── personas.py               # prompts + compile_character (spec'd)
│   ├── synthesis.py              # post-call story + feedback (spec'd)
│   └── writers/                  # artifact writers (spec'd)
├── tests/
├── evals/
│   ├── benchmarks/               # vendored datasets
│   └── runs/                     # eval run outputs (gitignored)
├── sessions/                     # runtime session store (gitignored)
└── web/
    └── viewer.html               # static artifact viewer (spec'd)
```
