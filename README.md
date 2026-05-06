# rehearse

Voice agent inference engine for real-time conversation coaching.

A 5-minute phone call: 1 minute intake, 3 minutes live practice with an AI counterparty, 1 minute feedback. Built as a prototype ML system — live sessions are the data source for continual improvement of a **purpose-built audio LLM** that replaces the off-the-shelf stack in v1.

The product is a coaching call. The architecture is an ML data-collection loop. Every session is simultaneously a unit of user value and a training record.

See [SPEC.md](SPEC.md) for the foundational design.

Live runtime + eval harness scaffold.

- Runtime: SMS triggers an outbound call. Twilio Media Streams bridges audio to Hume EVI. A custom-language-model webhook serves coach/character turns. Transcript, prosody, audio, and telemetry artifacts persist per session. Post-call synthesis writes `story.md` + `feedback.md` and SMSes a viewer link back. End-to-end verified on a real phone call (2026-05-01). See [`docs/specs/v2026-04-28-runtime-workstream.md`](docs/specs/v2026-04-28-runtime-workstream.md).
- Eval harness: see [`rehearse/eval/README.md`](rehearse/eval/README.md). Public shape is evals, datasets, scorers, and environments. Runs `noop` offline and has the MME-Emotion audio eval scaffold.

## Status (2026-05-01)

| Workstream | Stage |
|---|---|
| Pydantic data contracts (`rehearse/types.py`) | ✅ frozen |
| Eval harness skeleton | ✅ shipped |
| MME-Emotion eval + audio environment scaffold | ✅ shipped |
| Runtime (Twilio + owned audio bridge + Hume EVI + CLM webhook) | ✅ shipped — verified end-to-end |

## Strategic frame

Three load-bearing claims:

1. **Prosody is the product, not a byproduct.** The coach's value is detecting incongruence between what is said and how it is said. Transcript-only systems cannot do this.
2. **Purpose-built model, not a wrapper.** v0 uses Claude + Hume; v1 fine-tunes an open-weights audio LLM (Gemma 4 E4B) on preference pairs mined from real sessions. The architecture stays constant; the model slots swap.
3. **Single schema everywhere.** Production session artifacts and eval-harness outputs are the same pydantic types. A frozen session is replayable through any stage.

## Roadmap

Three workstreams advance in parallel. Each ships in numbered phases; each phase is one PR with green tests + a runnable demo.

### Eval harness (`rehearse-eval`) — `rehearse/eval/`

Plugin-shaped: evals, datasets, scorers, environments, providers, and executors are each small protocol-style units. Independent of runtime.

| Phase | Status | Scope |
|---|---|---|
| 1 — Skeleton | ✅ shipped | Protocols, registries, runner, CLI, `LocalSubprocessExecutor`, `noop` eval, `echo` environment. |
| A1 — MME-Emotion replacement | ✅ shipped | Removed text-only eval path, added MME-Emotion dataset/eval scaffold, deterministic recognition scorer. |
| A2 — Provider plugin layer + Gemini provider | ✅ shipped | `AudioLLMProvider` protocol, Gemini provider wrapper, `list-providers` CLI. |
| A3 — `multimodal-llm` environment | ✅ shipped | Audio-native environment, provider selection, vLLM provider wrapper. Real runs require media files + credentials. |
| A4 — Reasoning scorer (Claude Opus judge) | 📝 spec'd | LLM-judge scorer. `MMEReasoningScorer` + reusable `LLMJudge` primitive. |
| A5 — vLLM provider (Gemma 4 E4B) | 📝 spec'd | OpenAI-compatible client pointed at a self-hosted vLLM server. Gated on a live endpoint. |
| A6 — Side-by-side comparison | 📝 spec'd | `rehearse-eval diff <run_a> <run_b>` per-dimension delta table. Gemini vs Gemma on the same clips. |
| A7 — Scale to 100 clips | 🔮 future | Move past the v0 hand-curated subset; `scripts/fetch_mme_emotion.py` from HuggingFace. |
| A8 — Rehearse-seed scenarios + `synthesis` / `full` environments | 🔮 future | First product-quality eval (not capability eval). Fault-recall, persona fidelity, holistic usefulness. |
| A9 — CI gating + regression workflow | 🔮 future | PR-blocking eval delta checks. |

Spec: [`docs/specs/v2026-04-27-eval-harness.md`](docs/specs/v2026-04-27-eval-harness.md), [`docs/specs/v2026-04-28-mme-emotion-and-audio-targets.md`](docs/specs/v2026-04-28-mme-emotion-and-audio-targets.md). User-facing: [`rehearse/eval/README.md`](rehearse/eval/README.md).

### Runtime (`rehearse-app`) — `rehearse/app.py`, `rehearse/agents/`, owned audio runtime modules

Single FastAPI service: SMS triggers an outbound call, runs an owned Twilio Media Streams to Hume EVI live-call loop with the Claude Agent SDK as the coach/character brain, persists every frame to disk, then synthesizes story + feedback post-call and SMSes a viewer link back.

| Phase | Status | Scope |
|---|---|---|
| R1 — Skeleton + Telephony | ✅ shipped | Twilio SMS/voice/Media Streams handlers; `SessionOrchestrator`; `LocalFilesystemStore`. |
| R2 — Owned Twilio/Hume runtime scaffold | ✅ shipped | `TwilioStream`, `HumeEVIClient`, `FrameBus`, live audio loop. |
| R3 — Phase machine + writers | ✅ shipped | `PhaseProcessor`, transcript/prosody/audio/telemetry writers. Default budgets: 60s intake / 180s practice / 60s feedback (5-min call). |
| R4 — Claude CLM + agents | ✅ shipped | `POST /chat/completions` CLM endpoint with `HUME_CLM_SECRET` bearer auth; coach + character responders; scripted fallback when no Anthropic key. |
| R5 — Post-call synthesis + viewer | ✅ shipped | Replayable `SessionSynthesizer` (story + feedback), viewer route, SMS notification. End-to-end SMS-to-SMS round-trip verified 2026-05-01. |
| R6 — Reliability + polish | 🟡 partial | Soft-cue phase transitions ✅, Twilio signature validation ✅. Open items tracked in [`docs/specs/v2026-05-06-r6-reliability-backlog.md`](docs/specs/v2026-05-06-r6-reliability-backlog.md): stream WAV to disk, Hume reconnect with backoff, `/twilio/status` finalize fallback, persist `SessionHandle` across restarts. |
| R7 — Storage option B (S3 mirror) | 🔮 future | New `S3MirrorStore` backend. Migration trigger: first non-founder user invited. |
| R8 — Consent + inline outcome capture | 📝 spec'd, stubs landed | Verbal consent gate during intake; on-the-line "did this rehearsal feel useful?" probe at end of feedback phase. Deterministic classifiers — no LLM. Schema fields and `rehearse/outcome.py` stubbed; full implementation per spec is the next runtime PR. Blocker for inviting non-founder callers (legal + training-data). Spec: [`docs/specs/v2026-05-01-consent-and-outcome-capture.md`](docs/specs/v2026-05-01-consent-and-outcome-capture.md). |

Spec: [`docs/specs/v2026-04-28-runtime-workstream.md`](docs/specs/v2026-04-28-runtime-workstream.md), [`docs/specs/v2026-04-28-drop-pipecat.md`](docs/specs/v2026-04-28-drop-pipecat.md), with historical detail in [`docs/specs/v2026-04-27-runtime.md`](docs/specs/v2026-04-27-runtime.md).

#### Hume EVI configs (`rehearse-hume`)

Voice, greeting copy, prompt, timeouts, turn detection, nudges, and interruption thresholds are declared in `rehearse/services/hume_configs.py` (`PERSONAS` registry) and reconciled against the live Hume workspace via the `rehearse-hume` CLI.

Run `rehearse-hume diff` or `sync` whenever you change a persona — the workspace stays in lock-step with the repo, so iteration on voice/prompt/etc. is a code change + one command, not a console click trail.

```bash
uv run rehearse-hume diff   # show planned actions; exit 1 if drift
uv run rehearse-hume sync   # apply create/new-version actions; write sessions/.hume_configs.json
```

**When to run:**
- Before placing a call after editing `PERSONAS` (voice swap, prompt tweak, timeout change).
- In CI on PRs that touch `rehearse/services/hume_configs.py` (use `diff` as a gate — exit 1 means the workspace will drift if the PR lands).
- Once after pulling a teammate's change to `PERSONAS`, so your local workspace matches.

**First-run migration:** Hume's existing console-edited config is named e.g. `"Your smart companion (5/1/2026, ...)"`. Either rename it in the console to match the registry's `display_name` (`"Rehearse Coach (default)"`) before the first `sync`, or accept that `sync` will create a second config alongside it. Spec: [`docs/specs/v2026-05-06-hume-config-as-code.md`](docs/specs/v2026-05-06-hume-config-as-code.md).

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
| Runtime audio loop | Owned `TwilioStream` + `HumeEVIClient` + `FrameBus` | unchanged |
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
│       ├── v2026-04-27-eval-harness.md
│       ├── v2026-04-27-runtime.md
│       ├── v2026-04-28-drop-pipecat.md
│       ├── v2026-04-28-runtime-workstream.md
│       └── v2026-04-28-mme-emotion-and-audio-targets.md
├── rehearse/                     # application package
│   ├── types.py                  # all pydantic interfaces
│   ├── eval/                     # eval harness (Phases 1–2 shipped)
│   │   └── README.md
│   ├── app.py                    # runtime FastAPI entry (spec'd)
│   ├── agents/                   # Claude Agent SDK roles (spec'd)
│   ├── pipeline.py               # runtime wiring entrypoint (spec'd)
│   ├── phases.py                 # phase timing / transitions (spec'd)
│   ├── personas.py               # prompts + compile_character (spec'd)
│   ├── synthesis.py              # post-call story + feedback (spec'd)
│   ├── audio/                    # Twilio audio bridge (spec'd)
│   ├── services/                 # Hume EVI client (spec'd)
│   └── writers/                  # artifact writers (spec'd)
├── tests/
├── evals/
│   ├── datasets/                 # vendored eval datasets
│   └── runs/                     # eval run outputs (gitignored)
├── sessions/                     # runtime session store (gitignored)
└── web/
    └── viewer.html               # static artifact viewer (spec'd)
```
