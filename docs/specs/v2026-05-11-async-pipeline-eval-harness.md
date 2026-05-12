# rehearse — Spec: Async Pipeline Eval Harness (v1)

**Status**: draft
**Owner**: jz
**Date**: 2026-05-11
**Depends on**:
- `rehearse/eval/runner.py`
- `rehearse/eval/cli.py`
- `rehearse/eval/executors/in_process.py`
- `rehearse/eval/executors/local_subprocess.py`
- `rehearse/eval/score_stream.py`
- `rehearse/eval/report.py`
- `docs/specs/v2026-05-11-live-audio-eval-sandbox.md`
**Amends**: `docs/specs/v2026-05-06-eval-system-roadmap.md`
**Supersedes**: nothing

---

## 0. One-line Summary

Replace the eval runner's synchronous "wait for all rollouts, then score"
barrier with an async pipeline: rollout workers continuously produce completed
samples, scoring workers consume them from a queue, and the report updates as
scores arrive.

---

## 1. Outcome

`rehearse-eval run` should keep available local compute and provider sessions
busy throughout an eval run, especially for heavy-tailed voice rollouts.

The target operator experience:

```bash
uv run rehearse-eval run \
  --eval voice-rollout-judges \
  --environment live-audio-sandbox \
  --limit 20 \
  --rollout-workers 8 \
  --scoring-workers 2 \
  --scheduler async
```

During the run:

```text
[live-audio-sandbox] queued 20 rollout jobs
[live-audio-sandbox] vrj-s01 rollout done (31s) -> ok
[voice-rollout-judges] vrj-s01 scored (12s) -> rwrd=0.62
[report] n=1/20 rwrd=0.62 elapsed=43s rollout_util=0.91 scorer_util=0.48
```

At the end, the CLI still prints the same Metrics legend and run-history table
from `rehearse/eval/report.py`.

---

## 2. Problem

The current runner starts rollouts concurrently, but then waits for every
rollout before it scores anything:

```python
rollouts = await asyncio.gather(*(run_one(i, rep, ex) for i, rep, ex in plan))
```

This wastes useful wall-clock time when rollout durations have a long tail:

- fast rollouts sit unscored while the slowest rollout finishes
- `scores.jsonl`, `results.jsonl`, and the report table do not reflect
  completed work until the rollout barrier is crossed
- failures are surfaced late
- expensive live-audio rollouts cannot be inspected incrementally
- local cores and subprocess slots are underutilized during the scoring phase,
  because scoring does not overlap with rollout execution

This is analogous to synchronous RL: sampling and training happen in lockstep,
and straggler samples set the step time. For eval, "sampling" is rollout
generation and "training" is scoring/aggregation.

---

## 3. Proposed Architecture

```text
                         +----------------------+
                         | Eval CLI             |
                         | RunConfig            |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | RolloutJob Queue     |
                         | example x rep x seed |
                         +----------+-----------+
                                    |
            +-----------------------+-----------------------+
            |                       |                       |
            v                       v                       v
   +----------------+      +----------------+      +----------------+
   | RolloutWorker  | ...  | RolloutWorker  | ...  | RolloutWorker  |
   | live/runtime   |      | live/runtime   |      | live/runtime   |
   +-------+--------+      +-------+--------+      +-------+--------+
           |                       |                       |
           +-----------------------+-----------------------+
                                   |
                                   v
                         +----------------------+
                         | Completed Queue      |
                         | RolloutEnvelope      |
                         +----------+-----------+
                                    |
                 +------------------+------------------+
                 |                                     |
                 v                                     v
        +----------------+                    +----------------+
        | ScoringWorker  |        ...         | ScoringWorker  |
        | scorers        |                    | scorers        |
        +--------+-------+                    +--------+-------+
                 |                                     |
                 +------------------+------------------+
                                    |
                                    v
                         +----------------------+
                         | ScoreStreamWriter    |
                         | scores.jsonl         |
                         | results.jsonl        |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | RollingAggregator    |
                         | partial/final means  |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Report + Telemetry   |
                         | Rich table + charts  |
                         +----------------------+
```

### 3.1 Work Items

```python
@dataclass(frozen=True)
class RolloutJob:
    idx: int
    rep: int
    example: BenchmarkExample
    run_dir: Path
    rng_seed: int


@dataclass(frozen=True)
class RolloutEnvelope:
    job: RolloutJob
    rollout: RolloutResult
    started_at: datetime
    completed_at: datetime
    worker_id: int


@dataclass(frozen=True)
class ScoredEnvelope:
    rollout_envelope: RolloutEnvelope
    scores: list[RubricScore]
    started_at: datetime
    completed_at: datetime
    worker_id: int
```

### 3.2 Queues

Use bounded queues to create backpressure:

```python
pending_rollouts: asyncio.Queue[RolloutJob]
completed_rollouts: asyncio.Queue[RolloutEnvelope]
scored_rollouts: asyncio.Queue[ScoredEnvelope]
```

Recommended defaults:

```text
completed_rollouts.maxsize = max(1, scoring_workers * 2)
scored_rollouts.maxsize = max(1, scoring_workers * 2)
```

If scoring falls behind, rollout workers naturally pause when
`completed_rollouts` fills. This prevents memory growth from unbounded
artifact metadata and large rollout payloads.

### 3.3 Worker Allocation

The old `--concurrency` remains as a compatibility alias for rollout workers.

New CLI flags:

```text
--scheduler sync|async             default: async after migration
--rollout-workers N                default: min(8, effective_worker_slots)
--scoring-workers N                default: min(2, effective_worker_slots)
--max-worker-slots N               default: os.cpu_count() or 10
--io-oversubscription FLOAT        default: 1.0
--memory-budget-mb MB              default: auto-detect available memory
--min-free-memory-mb MB            default: 1024
--report-interval-s SECONDS        default: 10
--utilization-sample-interval-s S  default: 1
```

For a 10-core machine, use this initial layout:

```text
rollout_workers = 8
scoring_workers = 2
```

For live provider-limited runs, the operator can lower rollout workers:

```bash
uv run rehearse-eval run \
  --eval voice-rollout-judges \
  --environment live-audio-sandbox \
  --limit 20 \
  --rollout-workers 2 \
  --scoring-workers 2
```

### 3.4 Threads, Cores, and Memory Limits

Do not model "one worker equals one CPU core" too literally. The eval workload
is mixed:

- live rollouts are mostly network and provider I/O
- local artifact writes are I/O
- deterministic scorers are usually light CPU
- multimodal/LLM judges are provider I/O
- local model judges or audio processing may become CPU/GPU/memory-bound

For I/O-bound rollouts, it is safe and desirable to run more logical worker
slots than physical cores, as long as memory and provider limits allow it.
In Python this should be implemented as asyncio tasks and subprocess slots,
not CPU-bound Python threads.

For CPU-bound local scoring, use process-level parallelism rather than Python
threads because of the GIL.

Effective worker slots:

```text
core_slots = --max-worker-slots or os.cpu_count() or 10
memory_slots = floor((available_memory_mb - min_free_memory_mb) / per_slot_memory_mb)
provider_slots = environment/provider limit, if known
oversubscribed_slots = floor(core_slots * io_oversubscription)

effective_worker_slots =
  max(1, min(oversubscribed_slots, memory_slots, provider_slots))
```

`per_slot_memory_mb` starts as a conservative config value, then becomes a
measured rolling p95 from prior jobs in the same run:

```text
per_slot_memory_mb = max(configured_floor_mb, p95(observed_job_peak_rss_mb))
```

If memory pressure rises during a run, the scheduler should stop launching new
rollouts until RSS returns below the budget. Existing rollouts are not killed
unless they hit their normal timeout.

---

## 4. Aggregation Semantics

Per-rollout scorers stream immediately. Meta-scorers run later.

### 4.1 Rolling Aggregates

Keep append-only score rows as the source of truth, plus an in-memory aggregate:

```python
class RollingAggregate:
    def __init__(self) -> None:
        self.values_by_dimension: dict[str, list[float]] = defaultdict(list)

    def add_many(self, scores: Iterable[RubricScore]) -> None:
        for score in scores:
            self.values_by_dimension[score.dimension].append(score.value)

    def snapshot(self) -> dict[str, float]:
        return {
            dimension: statistics.mean(values)
            for dimension, values in self.values_by_dimension.items()
            if values
        }
```

The partial report uses `n_scored`, not `n_planned`:

```text
planned: 20
completed rollouts: 7
scored rollouts: 5
report row n: 5
```

### 4.2 Meta-scorers

Meta-scorers such as stability need grouped rollouts. They are not emitted
for individual rollout completions unless the relevant group is complete.

Rules:

- ordinary `Scorer.score(...)`: run as soon as one rollout is available
- `MetaScorer.score_meta(...)`: run after all rollouts for that example group
  are available, or at the end of the run
- final `run.json` includes both per-rollout aggregate scores and meta-scores

---

## 5. Utilization and Speedup Proof

This change must ship with instrumentation. We should be able to prove it is
faster and uses resources better, not merely believe it.

### 5.1 New Artifacts

Each run writes:

```text
evals/runs/{run_id}/scheduler_events.jsonl
evals/runs/{run_id}/utilization.jsonl
evals/runs/{run_id}/pipeline_metrics.json
evals/runs/{run_id}/pipeline_speedup.csv
evals/runs/{run_id}/pipeline_speedup.svg
```

`scheduler_events.jsonl` rows:

```json
{"t": 0.001, "event": "job_queued", "example_id": "vrj-s01", "rep": 0}
{"t": 0.012, "event": "rollout_started", "worker_id": 3, "example_id": "vrj-s01"}
{"t": 31.4, "event": "rollout_completed", "worker_id": 3, "example_id": "vrj-s01", "status": "ok"}
{"t": 31.5, "event": "scoring_started", "worker_id": 1, "example_id": "vrj-s01"}
{"t": 43.2, "event": "scoring_completed", "worker_id": 1, "example_id": "vrj-s01", "scores": 8}
```

`utilization.jsonl` rows sampled every second:

```json
{
  "t": 42.0,
  "active_rollout_workers": 8,
  "active_scoring_workers": 2,
  "pending_rollouts": 9,
  "completed_queue_depth": 1,
  "scored_count": 5,
  "process_rss_mb": 1832,
  "child_rss_mb": 4201,
  "cpu_percent": 741.2
}
```

Use `psutil` if available; otherwise fall back to `resource`/`tracemalloc`
for the main process and mark child process memory as unavailable.

### 5.2 Metrics

`pipeline_metrics.json`:

```json
{
  "scheduler": "async",
  "planned_rollouts": 20,
  "scored_rollouts": 20,
  "rollout_workers": 8,
  "scoring_workers": 2,
  "wall_time_s": 612.4,
  "time_to_first_rollout_s": 28.7,
  "time_to_first_score_s": 41.9,
  "rollout_worker_utilization": 0.87,
  "scoring_worker_utilization": 0.62,
  "peak_rss_mb": 6033,
  "p95_rss_mb": 5510,
  "mean_rollout_s": 91.3,
  "p95_rollout_s": 188.9,
  "max_rollout_s": 243.1
}
```

Utilization formulas:

```text
rollout_worker_utilization =
  sum(rollout_worker_busy_seconds) / (rollout_workers * wall_time_s)

scoring_worker_utilization =
  sum(scoring_worker_busy_seconds) / (scoring_workers * wall_time_s)

speedup =
  sync_wall_time_s / async_wall_time_s

memory_delta =
  async_peak_rss_mb - sync_peak_rss_mb
```

### 5.3 Benchmark Command

Add a benchmark mode that runs the same deterministic fixture twice:

```bash
uv run rehearse-eval benchmark-scheduler \
  --eval voice-rollout-judges \
  --environment runtime-sandbox \
  --limit 10 \
  --sync-concurrency 10 \
  --async-rollout-workers 8 \
  --async-scoring-workers 2
```

For hermetic CI, add a synthetic environment:

```text
staggered-sleep-sandbox
```

It emits valid `RolloutResult` objects after deterministic heavy-tailed
durations, for example:

```text
1s, 1s, 2s, 2s, 3s, 5s, 8s, 13s, 21s, 34s
```

This gives a cheap, deterministic proof that async scoring begins before the
slowest rollout completes.

### 5.4 Chart

The benchmark command generates `pipeline_speedup.svg` and prints a small
terminal chart.

Example terminal chart:

```text
Scheduler Benchmark: voice-rollout-judges, n=10

Wall time
sync   412s |########################################|
async  258s |#########################               | 1.60x faster

Time to first score
sync   392s |########################################|
async   41s |####                                    | 9.56x faster

Peak RSS
sync  4.8GB |########################                |
async 5.6GB |############################            | +0.8GB

Worker utilization
sync rollout   0.54 |######################          |
async rollout  0.87 |###################################|
async scoring  0.62 |#########################       |
```

The SVG chart should include:

- wall time before/after
- time to first score before/after
- rollout worker utilization before/after
- scoring worker utilization before/after
- peak RSS before/after
- queue depth over time for async runs

Acceptance threshold for the deterministic benchmark:

```text
async wall time <= 0.75 * sync wall time
async time_to_first_score <= 0.25 * sync time_to_first_score
async peak RSS <= sync peak RSS + configured memory budget headroom
```

For real live-audio runs, report the measured speedup but do not gate CI on it,
because provider latency and credits make it noisy.

---

## 6. Implementation Steps

### Phase 1 — Data Model and Compatibility Switch

1. Add `scheduler: Literal["sync", "async"] = "sync"` to `RunConfig`.
2. Add `rollout_workers`, `scoring_workers`, `max_worker_slots`,
   `io_oversubscription`, `memory_budget_mb`, `min_free_memory_mb`,
   `report_interval_s`, and `utilization_sample_interval_s`.
3. Preserve `concurrency` as an alias for `rollout_workers`.
4. Add `RolloutJob`, `RolloutEnvelope`, and `ScoredEnvelope` dataclasses.
5. Keep the existing path as `execute_run_sync(...)`.
6. Add `execute_run_async(...)` behind `--scheduler async`.

Verification:

```bash
uv run pytest tests/eval/test_runner.py -q
uv run rehearse-eval run --eval noop --environment echo --limit 1 --scheduler sync
uv run rehearse-eval run --eval noop --environment echo --limit 1 --scheduler async
```

Both scheduler modes produce equivalent final `run.json` aggregate scores for
deterministic evals.

### Phase 2 — Rollout Worker Queue

1. Build `pending_rollouts` from `(example, repetition, seed)`.
2. Start `rollout_workers` tasks.
3. Each worker calls `executor.submit(...)`.
4. Each worker writes a `RolloutEnvelope` into `completed_rollouts`.
5. Write failure files as soon as a rollout fails.

Verification:

- test with one slow rollout and one fast rollout
- assert the fast rollout enters `completed_rollouts` before the slow rollout
  finishes
- assert worker slots refill when jobs remain

### Phase 3 — Streaming Scoring

1. Start `scoring_workers` tasks.
2. Each worker consumes `completed_rollouts`.
3. Run the normal scoring plan for each rollout.
4. Publish scores through `ScoreStreamWriter` immediately.
5. Append the same score rows to final `all_scores`.
6. Update `RollingAggregate`.
7. Refresh `results.jsonl` or write a durable append-only interim file.

Verification:

- test that `scores.jsonl` receives rows before the slowest rollout completes
- test scorer crashes still produce crash `RubricScore`
- test failed rollouts are scored where possible and written to `failures/`

### Phase 4 — Meta-scorer Barrier

1. Store `rollouts_by_example` and `scores_by_example` as envelopes arrive.
2. Run meta-scorers only when all planned jobs are done, or when one example's
   full repetition group is complete.
3. Add meta-score rows to `all_scores` and final aggregate.

Verification:

- stability scorer still emits `stability_unmeasurable` for one repetition
- repeated examples preserve grouped meta-scoring semantics

### Phase 5 — Utilization Telemetry

1. Add `SchedulerTelemetryWriter`.
2. Emit scheduler events at every queue/worker state transition.
3. Sample utilization every second.
4. Track worker busy seconds.
5. Track peak and p95 RSS.
6. Write `pipeline_metrics.json`.

Verification:

- unit test busy-time accounting with fake clock
- integration test confirms telemetry files exist and contain queue depth,
  worker activity, and memory fields

### Phase 6 — Benchmark and Chart

1. Add `staggered-sleep-sandbox`.
2. Add `rehearse-eval benchmark-scheduler`.
3. Run sync and async modes on the same deterministic plan.
4. Write `pipeline_speedup.csv`.
5. Generate `pipeline_speedup.svg`.
6. Print the terminal chart.

Verification:

```bash
uv run rehearse-eval benchmark-scheduler --limit 10
```

Must produce:

```text
pipeline_metrics.sync.json
pipeline_metrics.async.json
pipeline_speedup.csv
pipeline_speedup.svg
```

And satisfy:

```text
async wall time <= 0.75 * sync wall time
async time_to_first_score <= 0.25 * sync time_to_first_score
```

### Phase 7 — Make Async Default

After the async scheduler is green on deterministic tests and one real
`voice-rollout-judges` run:

1. Change default scheduler to `async`.
2. Keep `--scheduler sync` for one release as a fallback.
3. Update `README.md`, `rehearse/eval/README.md`, and Makefile targets.

---

## 7. Verifiable Goal

Build is complete when this command:

```bash
uv run rehearse-eval benchmark-scheduler --limit 10
```

prints a before/after chart and writes benchmark artifacts showing:

```text
async wall time <= 75% of sync wall time
async time to first score <= 25% of sync time to first score
scores.jsonl receives rows before all rollouts complete
peak memory stays below configured memory budget
final aggregate scores match sync mode for deterministic scorers
```

For live-audio evals, the non-CI verification command is:

```bash
uv run rehearse-eval run \
  --eval voice-rollout-judges \
  --environment live-audio-sandbox \
  --limit 10 \
  --scheduler async \
  --rollout-workers 2 \
  --scoring-workers 2
```

Expected proof:

- report table updates after the first scored rollout
- no completed rollout waits for the slowest rollout before scoring
- `pipeline_metrics.json` reports non-zero rollout and scoring utilization
- `pipeline_speedup.csv` can compare against a previous sync run

---

## 8. Non-goals

- Updating model weights during an eval run. Eval policy/config remains
  immutable for comparability.
- Changing scorer semantics.
- Replacing `LocalSubprocessExecutor`.
- Guaranteeing speedup for provider-limited runs. The harness should expose
  bottlenecks clearly; if Hume or a judge provider is the limiter, the chart
  should make that obvious.

---

## 9. Future Extension: Online RL Data Generation

If this harness later drives online RL or preference-data generation, add:

```text
policy_version
sample_started_policy_version
sample_completed_policy_version
max_allowed_staleness
```

Eval runs should keep these fixed. Training runs may allow bounded staleness,
but only with explicit importance-ratio accounting in the trainer.
