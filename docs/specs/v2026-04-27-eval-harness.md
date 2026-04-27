# rehearse — Spec: Eval Harness v0

**Status**: draft
**Owner**: jz
**Depends on**: `rehearse/types.py` (frozen), SPEC.md §7, §10
**Supersedes**: nothing

---

## 0. One-line summary

Build a stand-alone eval harness that scores the rehearse system against a pluggable list of benchmarks — starting with three hand-seeded rehearse scenarios and the EQ-Bench public set — running rollouts in parallel sandboxed workers, with the runtime mocked at the frame boundary so no Twilio / Hume / Pipecat infrastructure is required to execute a full evaluation.

## 1. Goal

Make "did this change make the system better?" a single CLI command, answerable in minutes, without spinning up the live phone path. Every architectural decision below serves that goal.

Concretely, after this spec ships:

```
$ rehearse-eval run --benchmark eq-bench --target synthesis --limit 50 --concurrency 8
$ rehearse-eval run --benchmark rehearse-seed --target full
$ rehearse-eval list-benchmarks
$ rehearse-eval list-targets
```

…produces `evals/runs/{run_id}/results.jsonl` of `RubricScore` rows plus a `summary.md`, comparable run-to-run.

## 2. Non-goals

- **Not** the live runtime. No Twilio, no Pipecat audio, no Media Streams. (See §6 for what we *do* exercise of the runtime code.)
- **Not** model fine-tuning, DPO mining, or training-corpus assembly. Those consume eval output; this spec is upstream.
- **Not** a UI. CLI + markdown reports only.
- **Not** real Hume EVI prosody capture. Tier-1 scripted prosody only in v0; Tier 2 deferred.
- **Not** a service. Single-process orchestrator, local subprocess workers.

## 3. Design commitments

These follow from SPEC.md §14 and the user's eval-driven brief; they constrain every choice below.

1. **Eval is independent of runtime.** The harness imports nothing from `rehearse/app.py`, `rehearse/pipeline.py`, or any module that touches Twilio/Pipecat I/O. It imports `rehearse/types.py`, `rehearse/personas.py`, `rehearse/synthesis.py`. Anything else used by both must be moved into a pure module.
2. **Single schema everywhere.** Eval-produced `Session` artifacts are byte-equivalent in shape to runtime-produced ones. Tested by a round-trip property check.
3. **Benchmarks are plugins.** Adding EQ-Bench, MMLU-EQ, or a future bespoke benchmark is implementing one interface and registering it. No core-harness changes.
4. **Targets are plugins.** What is being evaluated (full session, just feedback synthesis, just persona compile, raw LLM call) is a `Target`. Each benchmark declares which targets it can run against.
5. **Rollouts are sandboxed and parallel.** Each example runs in an isolated worker so a crash, hang, or stateful bug in one rollout cannot pollute another. Concurrency is a runtime flag, not a code change.
6. **Replayable.** Given an `EvalRun` ID, the inputs (benchmark snapshot, target version, model slots, seed) fully determine outputs modulo model temperature. Results carry enough metadata to re-run.

## 4. Architecture

```
                ┌──────────────────────────────────────────────────┐
                │                   rehearse-eval                  │
                │                       (CLI)                      │
                └───────────────────────────┬──────────────────────┘
                                            │
                                            ▼
                              ┌─────────────────────────┐
                              │   Runner (orchestrator) │
                              │   - resolves benchmark  │
                              │   - resolves target     │
                              │   - schedules rollouts  │
                              │   - aggregates scores   │
                              └────────┬────────────────┘
                                       │
                  ┌────────────────────┼────────────────────┐
                  ▼                    ▼                    ▼
           Benchmark             Executor              Target
           (plugin)              (plugin)              (plugin)
                  │                    │                    │
        loads ExampleBatch     runs N rollouts in    given an Example,
        defines scoring        parallel, isolated    produces a Session
        plan                   workers; collects     (or partial artifact)
                               artifacts             via mocked or live deps
                  │                    │                    │
                  └─────────┬──────────┴──────────┬─────────┘
                            ▼                     ▼
                       Scorer(s)            evals/runs/{id}/
                       det + LLM            ├ results.jsonl
                       per dimension        ├ session/{example_id}/...
                       │                    ├ summary.md
                       ▼                    └ run.json (EvalRun metadata)
                   RubricScore
```

Four interfaces. That's the whole vocabulary.

## 5. Interfaces

All defined as Python `Protocol` types in `rehearse/eval/protocols.py`. No inheritance required; ducks welcome.

### 5.1 `Benchmark`

```python
class Benchmark(Protocol):
    name: str                                    # "eq-bench", "rehearse-seed"
    version: str                                 # snapshot tag, e.g. "v3-2024-09"
    supported_targets: frozenset[str]            # {"synthesis", "feedback-only", "raw-llm"}

    def load(self) -> Iterable[BenchmarkExample]: ...
    def scoring_plan(self) -> list[ScoringStep]: ...
```

A `BenchmarkExample` is the input contract: an `id`, a `payload` (benchmark-specific dict), and an `expected` (benchmark-specific dict for ground truth). The `Target` is responsible for translating `payload` into a runnable rollout; the `Scorer`s in `scoring_plan` know how to read `expected` against the produced `Session`.

### 5.2 `Target`

```python
class Target(Protocol):
    name: str                                    # "full", "synthesis", "raw-llm"
    version: str                                 # bumped on prompt/model change

    async def rollout(
        self,
        example: BenchmarkExample,
        run_dir: Path,
        rng_seed: int,
    ) -> RolloutResult: ...
```

`RolloutResult` carries either a full `Session` (artifacts written under `run_dir`) or a partial artifact (e.g. a single feedback string + telemetry) plus an exit status. Targets are stateless; the rollout's only side effects are file writes under `run_dir` and outbound model calls.

Three v0 targets:

| Target | What it runs | What it mocks |
|---|---|---|
| `full` | Intake → persona compile → simulated practice transcript+prosody → synthesis | Hume (scripted prosody), Twilio (none — synthetic transcript fed directly), audio (no audio path) |
| `synthesis` | Just `synthesize_story` + `synthesize_feedback` on a pre-baked `Session` fixture | Everything upstream of synthesis is a fixture |
| `raw-llm` | Single Claude call with the benchmark's prompt verbatim | Everything (used only for EQ-Bench-style single-shot evals) |

`raw-llm` exists specifically so EQ-Bench can run without any rehearse code path, giving us a baseline for "is our model better than a vanilla call to the same model."

### 5.3 `Scorer`

```python
class Scorer(Protocol):
    name: str
    dimension: RubricDimension | str             # str for benchmark-specific dimensions

    async def score(
        self,
        example: BenchmarkExample,
        rollout: RolloutResult,
    ) -> list[RubricScore]: ...
```

Two flavors live in `rehearse/eval/scorers/`: `deterministic.py` (pacing, citations, fault recall/precision, EQ-Bench correlation math) and `llm_judge.py` (Claude Opus, structured output). Each `Benchmark.scoring_plan()` returns the list of scorers to apply.

`RubricDimension` is extended in `types.py` with new variants only when a dimension is reusable across benchmarks. Benchmark-private dimensions stay as strings — we don't pollute the enum.

### 5.4 `Executor`

```python
class Executor(Protocol):
    async def submit(
        self,
        target: Target,
        example: BenchmarkExample,
        run_dir: Path,
        timeout_s: int,
        rng_seed: int,
    ) -> RolloutResult: ...
```

v0 implementation: `LocalSubprocessExecutor` — each rollout is a `python -m rehearse.eval.worker` subprocess with stdin=example JSON, stdout=`RolloutResult` JSON. This gives us:

- Crash isolation (one rollout's segfault doesn't kill the run).
- Clean memory reset between rollouts (no leaked event loops, no Hume client state).
- Trivial parallelism via an `asyncio.Semaphore`-bounded worker pool.
- A clean upgrade path to `ContainerExecutor` / `ModalExecutor` later — same protocol, different backend.

Concurrency is `--concurrency N` on the CLI; default 4. Hard timeout per rollout is benchmark-declared (EQ-Bench: 60s; rehearse-seed: 300s).

## 6. What's mocked, what's real

This is the load-bearing question for "eval-driven without runtime."

| Component | In `target=full` rollout | Why |
|---|---|---|
| Twilio webhook + Media Streams | **mocked** (skipped) | Eval drives transcript directly |
| Pipecat pipeline / `PhaseProcessor` | **mocked** (orchestrator drives phase boundaries by counting synthetic turns) | The phase machine is tested in isolation already; eval doesn't re-test it |
| Hume EVI voice / prosody capture | **mocked** (Tier-1 scripted prosody from `prosody_scripts.py`) | Cheap, deterministic, 80% of eval volume per SPEC §10.2 |
| Coach LLM (intake, feedback synthesis) | **real** Claude calls | This is the thing being evaluated |
| Character LLM (practice rollouts) | **real** Claude calls (via Hume CLM-style local stub) | We need real persona behavior to score believability |
| Synthetic user | **real** Claude-driven agent in `eval/synthetic_user.py` | Ground-truth fault injection per `SyntheticUserProfile` |
| Audio recording | **stub** (zero-byte `audio.wav` placeholder) | Audio is for review UI, not scoring; eval doesn't need it |
| Telemetry | **real** | Latency budgets without I/O are still meaningful for model latency |

The seam is clean: every component that is the *product* runs real; every component that is *infrastructure* is mocked. This is exactly the design `SimulatedTransport` from SPEC §10.2 implies.

## 7. Benchmark v0: rehearse-seed (3 examples)

Hand-authored, hand-labeled. Lives in `evals/examples/seed/`. One per category we expect to lean on first.

| ID | Category | Why this seed |
|---|---|---|
| `seed-001-cofounder-equity` | `professional_conflict` | High-stakes, asymmetric power, classic "bury the lede" + "over-justify" injected faults. Tests fault recall on the most common founder use case. |
| `seed-002-vulnerable-ask` | `vulnerability` | Tests prosody-incongruence detection: scripted false-confidence prosody on a hedging transcript. Forces the system to notice the said-vs-meant gap. |
| `seed-003-difficult-feedback-up` | `relationship_conflict` | Cross-status confrontation. Tests pacing (user tends to ramble) and missing-ask detection. |

Each example file contains a full `ExampleScenario` — `situation`, `counterparty`, `user_goal`, `synthetic_user` profile with `injected_faults` and `prosody_baseline`/`trajectory`, and a `ground_truth_diagnosis` list.

Scoring plan for `rehearse-seed` against `target=full`:

1. `pacing_adherence` — det
2. `fault_recall` — det (set match of `feedback.md` faults against `injected_faults`)
3. `fault_precision` — det + LLM (fault claim is grounded in transcript turn)
4. `feedback_groundedness` — det (citations resolve to real `utterance_id`s)
5. `prosody_citation_accuracy` — det
6. `incongruence_detection` — det + LLM
7. `character_persona_fidelity` — LLM judge
8. `usefulness_holistic` — LLM judge

## 8. Benchmark v0: EQ-Bench

### 8.1 Source

Public dataset at `https://github.com/EQ-bench/EQ-Bench`. We pin a commit SHA and vendor the prompt set into `evals/benchmarks/eq-bench/{commit}/questions.json` so eval runs are reproducible offline. License-check before vendoring; if license forbids, we fetch at run time and cache by SHA.

EQ-Bench format (summary): each question gives a short emotional dialogue and asks the model to rate the intensity (0–10) of four named emotions for one character. Reference scores exist; benchmark score is correlation between model ratings and reference, normalized to a 0–100 scale.

### 8.2 Adapter

`rehearse/eval/benchmarks/eq_bench.py`:

- `EQBenchBenchmark.load()` yields one `BenchmarkExample` per question; `payload` = the prompt dict, `expected` = the reference scores.
- `supported_targets = {"raw-llm"}`. EQ-Bench is single-turn text; running it through `target=full` would be type-confusion. We deliberately *do not* force-fit it.
- `scoring_plan()` returns one scorer: `EQBenchCorrelationScorer` (det), producing `RubricScore(dimension="eq_bench_score", value=...)`.

### 8.3 Why bother running EQ-Bench at all

EQ-Bench scores a single LLM call, which is exactly the shape of our **feedback synthesis** model slot. Running EQ-Bench against `raw-llm` with our current synthesis-slot model gives us:

1. A model-level baseline that's comparable to public leaderboards.
2. A regression canary when we swap the synthesis model (SPEC §9.2 step 1: critic-first DPO). If our fine-tuned 4B critic beats Claude on rehearse-seed but tanks on EQ-Bench, that's a generality red flag worth seeing.
3. A free sanity check that our model wiring works end-to-end with no rehearse-specific prompts in the way.

It is *not* a measure of product quality. The summary report labels it accordingly.

## 9. CLI

Entry point declared in `pyproject.toml`:

```toml
[project.scripts]
rehearse-eval = "rehearse.eval.cli:main"
```

Commands:

```
rehearse-eval list-benchmarks
rehearse-eval list-targets
rehearse-eval run \
    --benchmark <name>         # required
    --target <name>            # required; defaults to benchmark's preferred target
    --limit N                  # optional cap on examples
    --concurrency N            # default 4
    --seed N                   # default 0; deterministic example ordering & RNG
    --model-slot feedback=claude-opus-4-7  # repeatable; overrides defaults
    --tag <label>              # human label for the run
    --dry-run                  # resolve plan, print, exit
rehearse-eval show <run_id>    # print summary.md to stdout
rehearse-eval diff <run_a> <run_b>  # per-dimension delta table; nonzero exit if regression > spec gate
```

The `diff` command implements SPEC §10.5 regression gates: exit 1 if any load-bearing dimension regresses > 0.05 or any judge dimension regresses > 0.3/5. CI wires this up.

## 10. Outputs

```
evals/runs/{run_id}/
├ run.json                # EvalRun metadata: benchmark, target, versions, model_slots, seed, started_at, completed_at, git_sha
├ results.jsonl           # one RubricScore per line
├ summary.md              # aggregate scores, top-5 regressions vs previous run, sample failures
├ failures/               # rollouts that errored; one subdir per example_id with stderr + partial artifacts
└ sessions/               # one subdir per example_id with the full session artifact bundle (transcript, prosody, story, feedback, session.json, telemetry)
```

`summary.md` is the only human-facing artifact; it's the thing you paste into a PR description.

## 11. Repo additions

```
rehearse/eval/
├── __init__.py
├── cli.py                       # NEW — argparse entry point
├── protocols.py                 # NEW — Benchmark, Target, Scorer, Executor protocols
├── runner.py                    # NEW — orchestrator: resolve, schedule, aggregate
├── worker.py                    # NEW — subprocess entry point: stdin example → stdout RolloutResult
├── executors/
│   ├── __init__.py
│   └── local_subprocess.py      # NEW — v0 executor
├── benchmarks/
│   ├── __init__.py              # NEW — registry
│   ├── rehearse_seed.py         # NEW — loads evals/examples/seed/*.json
│   └── eq_bench.py              # NEW — EQ-Bench adapter
├── targets/
│   ├── __init__.py              # NEW — registry
│   ├── full.py                  # NEW — full mocked rollout
│   ├── synthesis.py             # NEW — synthesis-only on fixture
│   └── raw_llm.py               # NEW — single-shot Claude call
├── scorers/
│   ├── __init__.py
│   ├── deterministic.py         # MOVED from scorers.py; pacing, citations, fault sets, EQ-Bench correlation
│   └── llm_judge.py             # MOVED from judge.py; Opus structured-output rubric
├── synthetic_user.py            # EXTENDED — already stubbed
├── prosody_scripts.py           # EXTENDED — already stubbed
├── tts_bridge.py                # untouched (Tier 2, deferred)
└── harness.py                   # DELETED — replaced by runner.py + cli.py

evals/
├── examples/
│   └── seed/
│       ├── seed-001-cofounder-equity.json
│       ├── seed-002-vulnerable-ask.json
│       └── seed-003-difficult-feedback-up.json
├── benchmarks/
│   └── eq-bench/
│       └── {commit-sha}/
│           └── questions.json   # vendored snapshot
├── gold/                        # untouched (deferred)
└── runs/                        # gitignored

tests/eval/
├── test_protocols.py            # NEW — concrete check that each plugin satisfies its protocol
├── test_runner.py               # NEW — runner with two no-op benchmarks + targets, asserts ordering / parallelism / failure isolation
├── test_seed_examples.py        # NEW — every seed example loads + validates against ExampleScenario
├── test_eq_bench_adapter.py     # NEW — fixture snapshot of 3 EQ-Bench questions; correlation scorer math
├── test_local_subprocess.py     # NEW — executor isolates crashes, honors timeouts
└── test_scorers.py              # MOVED — deterministic scorer correctness
```

## 12. Phasing

Build order. Each phase ends with a green `pytest` and a working CLI command.

**Phase 1 — Skeleton + protocols + 1 trivial benchmark + target.** No model calls.
- `protocols.py`, `runner.py`, `cli.py` (`list-*`, `run`).
- `LocalSubprocessExecutor`.
- Stub benchmark (`noop` — emits 2 fake examples) and stub target (`echo`).
- `rehearse-eval run --benchmark noop --target echo` produces a valid `run.json` + empty `results.jsonl`.

**Phase 2 — EQ-Bench end-to-end.** Pulls the harness through with a real, simple benchmark.
- Vendor EQ-Bench snapshot.
- `EQBenchBenchmark`, `RawLLMTarget`, `EQBenchCorrelationScorer`.
- `rehearse-eval run --benchmark eq-bench --target raw-llm --limit 10` produces real scores.

**Phase 3 — rehearse-seed scenarios + synthesis target.** Validates the seam without running a full session.
- Author the 3 seed examples.
- `synthesis.py` becomes real (story + feedback Claude prompts).
- `SynthesisTarget` runs `synthesize_story` + `synthesize_feedback` against a hand-baked fixture session.
- `rehearse-eval run --benchmark rehearse-seed --target synthesis` produces real `RubricScore` rows on dimensions 1, 6, 7, 9.

**Phase 4 — `full` target with mocked runtime.** Closes the loop.
- `personas.compile_character` becomes real.
- `synthetic_user.py` + `prosody_scripts.py` become real (Tier-1 only).
- Orchestrator-driven phase advancement (no `PhaseProcessor` import).
- `rehearse-eval run --benchmark rehearse-seed --target full` produces full session bundles + scores on all 8 rubric-plan dimensions.

**Phase 5 — `diff` + CI gating.** Locks in the regression workflow.
- `rehearse-eval diff` with exit-code semantics.
- GitHub Actions job runs eval on PR, fails on regression, posts `summary.md` as a comment.

Phases 1–3 are the minimum viable harness; phases 4–5 close the loop. Phase 4 is the largest because it's where the synthetic-user agent gets real.

## 13. Open questions

| # | Question | Needs answer by |
|---|---|---|
| EQ1 | Do we vendor EQ-Bench data or fetch-at-run? Depends on license. | Phase 2 |
| EQ2 | Which EQ-Bench variant — v3 standard, Creative Writing, or both? | Phase 2 |
| EQ3 | Is `raw-llm` allowed to use `model_slots["feedback"]` only, or do we run all four slots through it independently? Probably the former — keep target singular. | Phase 2 |
| H1 | Should `Executor` accept a list of (target, example) tuples and parallelize internally, or does the runner own concurrency and the executor is per-rollout? Current spec says the latter for protocol simplicity. Revisit if we add Modal. | Phase 1 |
| H2 | Where does `model_slots` resolution live — CLI parses, runner injects into Target ctor, or Target reads env? Lean: runner injects, Target is pure. | Phase 1 |
| H3 | Random seed propagation: deterministic example order is easy; deterministic Claude calls require `temperature=0` + `seed` param. Do we force `temperature=0` in eval mode? | Phase 2 |
| S1 | Three seed examples is the floor. When do we hand-author the next 7 to hit the SPEC §2 "10 seed" target? | Phase 3 |

## 14. Out of scope, for the record

- Tier-2 prosody (real Hume in eval). Stub stays; pick up post-Phase-5.
- Human-labeled gold subset + judge calibration (SPEC §10.4). Pick up when we have ≥ 10 real session recordings.
- Preference pair mining, DPO corpus assembly. v1.
- Live runtime (Twilio + Pipecat + Hume EVI). Separate spec; this work does not block it and does not depend on it.
