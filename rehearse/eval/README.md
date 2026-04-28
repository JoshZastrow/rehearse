# rehearse-eval

Eval harness for the rehearse system. Plugin-shaped: benchmarks, targets, scorers,
and executors are each one Python `Protocol`. Adding a new benchmark or new
target is implementing one class and registering it.

The harness is **independent of the runtime** — it imports nothing from
`rehearse/app.py`, `rehearse/pipeline.py`, or anything that touches Twilio
or Pipecat I/O. You can run evaluations with no live phone path.

Design spec: [`docs/specs/v2026-04-27-eval-harness.md`](../../docs/specs/v2026-04-27-eval-harness.md)

---

## Install

```bash
uv sync
export ANTHROPIC_API_KEY=sk-ant-...   # only needed for targets that hit Claude
```

The `rehearse-eval` console script lands on `PATH` after `uv sync`. Run it via
`uv run rehearse-eval ...` or activate the venv directly.

## What's there today

| Benchmark | Examples | Supported targets | Notes |
|---|---|---|---|
| `noop` | 2 synthetic | `echo` | Smoke test. No model calls. |
| `eq-bench` | 3 sample questions (vendored stub) | `raw-llm` | Drop the real upstream `questions.json` at `evals/benchmarks/eq-bench/{commit}/questions.json` to run against the public set. |

| Target | What it does | Reads model slot |
|---|---|---|
| `echo` | Returns the example payload unchanged. No model call. | — |
| `raw-llm` | Single Claude call with `example.payload["prompt"]`. | `raw_llm` (default `claude-sonnet-4-6`) |

## Five-minute tour

```bash
# 1. List what's registered
uv run rehearse-eval list-benchmarks
uv run rehearse-eval list-targets

# 2. Smoke test (no API key needed)
uv run rehearse-eval run --benchmark noop --target echo

# 3. Resolve a plan without running anything
uv run rehearse-eval run --benchmark eq-bench --dry-run

# 4. Real EQ-Bench run (needs ANTHROPIC_API_KEY)
uv run rehearse-eval run --benchmark eq-bench --limit 3 --concurrency 2

# 5. View the summary for a previous run
uv run rehearse-eval show <run_id>
```

Each `run` prints the `run_id` and where artifacts live. Default: `evals/runs/{run_id}/`.

## CLI reference

```
rehearse-eval list-benchmarks
rehearse-eval list-targets
rehearse-eval run \
    --benchmark <name>           # required
    --target <name>              # defaults to benchmark.preferred_target
    --limit N                    # cap number of examples
    --concurrency N              # parallel rollouts (default 4)
    --seed N                     # rollout RNG seed (default 0)
    --model-slot KEY=VALUE       # repeatable; e.g. --model-slot raw_llm=claude-opus-4-7
    --tag LABEL                  # human label for the run
    --runs-root PATH             # where to write results (default evals/runs)
    --dry-run                    # resolve and print plan, don't execute
rehearse-eval show <run_id> [--runs-root PATH]
```

## Output layout

```
evals/runs/{run_id}/
├ run.json          # EvalRun manifest: benchmark, target, versions, seed, model_slots, timing
├ results.jsonl     # one RubricScore per (example × scorer)
├ summary.md        # human-facing aggregate; paste into PR descriptions
├ sessions/{ex}/    # per-example artifact dirs (used by full / synthesis targets)
└ failures/{ex}/    # error details for non-ok rollouts
```

`run_id` format: `YYYYMMDDTHHMMSS-{8-hex}` (e.g. `20260427T150709-0bbb81eb`).

## Concurrency and isolation

Each rollout runs in its own subprocess (`python -m rehearse.eval.worker`) — crash
isolation, clean memory between rollouts, and a hard timeout. The runner caps
concurrency with an `asyncio.Semaphore`; `--concurrency` controls the cap.

The executor is a `Protocol`. The default `LocalSubprocessExecutor` can be
swapped for a `ContainerExecutor` or a Modal-backed executor without touching
benchmarks or targets.

## Adding a benchmark

```python
# rehearse/eval/benchmarks/my_bench.py
from collections.abc import Iterable
from rehearse.eval.protocols import BenchmarkExample, Scorer

class MyBenchmark:
    name = "my-bench"
    version = "v1"
    supported_targets = frozenset({"raw-llm"})
    preferred_target = "raw-llm"

    def load(self) -> Iterable[BenchmarkExample]:
        yield BenchmarkExample(
            id="ex1",
            benchmark=self.name,
            payload={"prompt": "..."},
            expected={"answer": "..."},
        )

    def scoring_plan(self) -> list[Scorer]:
        return [MyScorer()]

    def rollout_timeout_s(self) -> int:
        return 60
```

Register it in `rehearse/eval/benchmarks/__init__.py`:

```python
from rehearse.eval.benchmarks.my_bench import MyBenchmark
BENCHMARKS["my-bench"] = MyBenchmark
```

## Adding a target

```python
# rehearse/eval/targets/my_target.py
from datetime import datetime
from pathlib import Path
from rehearse.eval.protocols import BenchmarkExample, RolloutResult

class MyTarget:
    name = "my-target"
    version = "v0"

    def __init__(self, model_slots: dict[str, str] | None = None) -> None:
        self.model_slots = model_slots or {}

    async def rollout(
        self,
        example: BenchmarkExample,
        run_dir: Path,
        rng_seed: int,
    ) -> RolloutResult:
        ...
```

Register it in `rehearse/eval/targets/__init__.py`:

```python
TARGETS["my-target"] = lambda slots: MyTarget(model_slots=slots)
```

## Adding a scorer

A scorer is anything with `name`, `dimension`, and an `async def score(example, rollout, run_id) -> list[RubricScore]`. Return one `RubricScore` per dimension you measure. Dimensions can be benchmark-private strings (e.g. `"my_bench_score"`) or members of the `RubricDimension` enum in `rehearse/types.py`.

## Vendoring real EQ-Bench data

The default sample at `evals/benchmarks/eq-bench/sample/questions.json` is a
3-question stub with the same shape as the upstream EQ-Bench format. To run
against the real set:

1. License-check the upstream repo: <https://github.com/EQ-bench/EQ-Bench>.
2. Pin a commit. Place the questions JSON at `evals/benchmarks/eq-bench/{commit-sha}/questions.json`.
3. Either edit `_DEFAULT_PATH` in `rehearse/eval/benchmarks/eq_bench.py` or set:
   ```bash
   export EQ_BENCH_DATA_PATH=evals/benchmarks/eq-bench/{commit-sha}/questions.json
   ```
4. Bump `EQBenchBenchmark.version` to the commit SHA so run manifests are reproducible.

Expected JSON shape:

```json
{
  "_meta": { "vendored_commit": "<sha>" },
  "questions": [
    {
      "id": "...",
      "dialogue": "...",
      "character": "...",
      "emotions": ["e1", "e2", "e3", "e4"],
      "reference_ratings": { "e1": 5, "e2": 7, "e3": 2, "e4": 8 }
    }
  ]
}
```

EQ-Bench scoring: per-question Pearson correlation between predicted and
reference ratings, rescaled to 0–100 via `(corr + 1) * 50`. Aggregate score
is the mean across questions.

## Running tests

```bash
uv run pytest tests/eval/
```

Tests cover protocol conformance, runner end-to-end (no model calls), the
EQ-Bench adapter (loading, prompt construction, parsing, correlation math),
and the subprocess executor (timeout, crash isolation).

## What's deliberately not built yet

The eval spec phases this as 1–5; only Phases 1 and 2 ship today. Not yet:

- **Phase 3** — `synthesis` target + 3 hand-authored `rehearse-seed` scenarios + LLM-judge scorers.
- **Phase 4** — `full` target with mocked runtime (synthetic-user agent, Tier-1 prosody scripts).
- **Phase 5** — `rehearse-eval diff <run_a> <run_b>` with regression-gate exit codes, plus a CI workflow that posts `summary.md` on PRs.

See [`docs/specs/v2026-04-27-eval-harness.md`](../../docs/specs/v2026-04-27-eval-harness.md) §12 for the phasing.
