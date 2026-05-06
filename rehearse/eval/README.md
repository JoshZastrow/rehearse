# rehearse-eval

Eval harness for the rehearse system. Plugin-shaped: evals, datasets, scorers,
environments, providers, and executors are small Python `Protocol`-style units.

The harness is independent of the runtime. You can run smoke evals with no live
phone path, no media files, and no model API keys.

Design specs:
- [`docs/specs/v2026-04-27-eval-harness.md`](../../docs/specs/v2026-04-27-eval-harness.md)
- [`docs/specs/v2026-04-28-mme-emotion-and-audio-targets.md`](../../docs/specs/v2026-04-28-mme-emotion-and-audio-targets.md)
- [`docs/specs/v2026-05-06-eval-system-roadmap.md`](../../docs/specs/v2026-05-06-eval-system-roadmap.md)
  — sequencing for multimodal scoring + DeepEval adoption.

## Install

```bash
uv sync
```

Optional env vars for live model runs:

```bash
export GOOGLE_API_KEY=...        # --provider gemini
export VLLM_BASE_URL=http://...  # --provider vllm
export VLLM_API_KEY=dummy
```

## What's There Today

| Eval | Dataset | Environments | Scorers | Notes |
|---|---|---|---|---|
| `noop` | `noop` | `echo` | `noop_score` | Offline smoke test. |
| `mme-emotion` | `mme-emotion` | `multimodal-llm` | `mme_recognition_accuracy` | 10-clip manifest scaffold. Real run needs media files + provider credentials. |

| Environment | What it does | Reads model slots |
|---|---|---|
| `echo` | Returns the example payload unchanged. | - |
| `raw-llm` | Single Claude call with `example.payload["prompt"]`. Kept for text diagnostics. | `raw_llm` |
| `multimodal-llm` | Loads an audio/video file and calls an audio LLM provider. | `provider`, `multimodal_hosted`, `multimodal_open` |

| Provider | Used by | Required env |
|---|---|---|
| `gemini` | hosted frontier baseline | `GOOGLE_API_KEY` |
| `vllm` | open-weights Gemma endpoint | `VLLM_BASE_URL`, `VLLM_API_KEY` |

## Five-Minute Tour

```bash
# 1. List what's registered
uv run rehearse-eval list-evals
uv run rehearse-eval list-datasets
uv run rehearse-eval list-environments
uv run rehearse-eval list-providers

# 2. Smoke test, no API key needed
uv run rehearse-eval run --eval noop --environment echo

# 3. Resolve the MME-Emotion plan without running provider calls
uv run rehearse-eval run --eval mme-emotion --dry-run

# 4. Real MME-Emotion run, after media files and GOOGLE_API_KEY are present
uv run rehearse-eval run --eval mme-emotion --environment multimodal-llm --provider gemini --limit 10

# 5. View the summary for a previous run
uv run rehearse-eval show <run_id>

# 6. Watch a sandbox rollout turn-by-turn (subprocess isolation is bypassed)
uv run rehearse-eval run --eval coach-dialogue-smoke --limit 1 --verbose
```

Deprecated aliases still work during migration: `list-benchmarks`,
`list-targets`, `--benchmark`, and `--target`.

## CLI Reference

```bash
rehearse-eval list-evals
rehearse-eval list-datasets
rehearse-eval list-environments
rehearse-eval list-providers
rehearse-eval run \
    --eval <name>                # required; --benchmark is a deprecated alias
    --environment <name>         # defaults to eval.preferred_environment
    --provider gemini|vllm       # shortcut for multimodal provider slot
    --limit N                    # cap number of examples
    --concurrency N              # parallel rollouts (default 4)
    --seed N                     # rollout RNG seed (default 0)
    --model-slot KEY=VALUE       # repeatable
    --tag LABEL                  # human label for the run
    --runs-root PATH             # where to write results (default evals/runs)
    --dry-run                    # resolve and print plan, don't execute
    --verbose                    # stream transport events live; forces in-process execution
rehearse-eval show <run_id> [--runs-root PATH]
```

## Output Layout

```text
evals/runs/{run_id}/
├ run.json          # EvalRun manifest: eval/environment versions, seed, model_slots
├ results.jsonl     # one RubricScore per example x scorer
├ summary.md        # human-facing aggregate
├ sessions/{ex}/    # per-example artifact dirs
└ failures/{ex}/    # error details for non-ok rollouts
```

## Adding Pieces

Datasets live in `rehearse/eval/datasets/` and only load examples. Evals live
in `rehearse/eval/evals/` and compose one dataset, a scoring plan, compatible
environments, and a rollout timeout. Environments live in
`rehearse/eval/environments/` and run the system under test. Scorers live in
`rehearse/eval/scorers/`.

Register new pieces in the matching package `__init__.py`.

## MME-Emotion Data

The v0 manifest lives at:

```text
evals/datasets/mme-emotion/v0-10clip/manifest.json
```

The manifest is checked in; media files are not. To fetch the v0 local subset:

```bash
# Optional if `hf` is not already on PATH
uv tool install huggingface_hub

python setup/fetch_mme_emotion.py
```

The script downloads [Karl28/MME-Emotion](https://huggingface.co/datasets/Karl28/MME-Emotion)
into `.cache/mme-emotion/`, extracts `ER_Lab.zip`, copies the selected clips into
`evals/datasets/mme-emotion/v0-10clip/clips/`, and rewrites `manifest.json` from
the upstream annotations. Both `.cache/` and copied clip media are gitignored.

If you already downloaded the dataset, reuse it:

```bash
python setup/fetch_mme_emotion.py --cache-dir /path/to/mme-emotion --skip-download
```

You can also set `MME_EMOTION_MANIFEST_PATH` to another manifest with valid
paths.

## Running Tests

```bash
uv run pytest tests/eval/
```

Tests cover protocol conformance, runner end-to-end, subprocess isolation, the
MME-Emotion dataset/eval shape, and the deterministic recognition scorer.

## Per-Dimension Scorers (Spec 1)

Spec 1 of the [eval roadmap](../../docs/specs/v2026-05-06-eval-system-roadmap.md)
introduced two new scorers and four `RubricScore` schema fields. The audio
companions (`AffectPerceptionJudgeScorer`, `DeliveryJudgeScorer`) arrive
with Spec 2.

| Scorer | Dimension | Modality | Source | Notes |
|---|---|---|---|---|
| `ContentJudgeScorer` | `content_quality` | `text` | DeepEval `G-Eval` via `MetricToScorer` | Default criteria narrowed to *what was said* — content, safety, trajectory direction. Stamps `judge_prompt_version` on every score. |
| `AggregateScorer` | `weighted_reward` | `aggregate` | Pure function over per-dim scores | Per-example `rubric_weights` override defaults; missing dimensions renormalize and tag `partial_modality`. Writes `judge.json` provenance to `artifacts_dir` when set. |

`RubricScore` gained four optional fields, all backwards-compatible with
artifacts written before Spec 1:

```python
class RubricScore:
    ...                                                       # legacy fields unchanged
    modality: Literal["text", "audio", "audio+text", "timing", "meta", "aggregate"] = "text"
    confidence: float | None = None
    judge_prompt_version: str | None = None
    flags: list[str] = []   # e.g. ["audio_missing", "uncalibrated", "partial_modality"]
```

A new `MetaScorer` protocol sits alongside `Scorer` for cross-rollout
metrics (stability lands in Spec 8); no implementations yet.

`TrajectoryJudgeScorer` is still in place during the transition. It will
be retired when Spec 2's audio judges replace its `emotion_responsiveness`
output; until then, `mme-sandbox-rollout` continues to use it.

### Composing the new scorers

```python
from rehearse.eval.scorers import AggregateScorer, ContentJudgeScorer

content = ContentJudgeScorer(prompt_version="content-quality-v1")
aggregator = AggregateScorer(
    weights={"content_quality": 0.35, "affect_perception": 0.35, "delivery_quality": 0.30},
    version="aggregate-v1",
)

# In an Eval.scoring_plan(): emit per-dim scorers via .score(), then call
# aggregator.aggregate(example, rollout, all_dim_scores, run_id) once the
# per-dim rows are collected. The runner will wire this in once all four
# scorers (content + affect + delivery + naturalness) coexist.
```

## DeepEval Adapter Layer

`rehearse/eval/deepeval_adapter/` bridges this harness to
[DeepEval](https://github.com/confident-ai/deepeval). The adapter is the
foundation of Spec 0 in the
[eval system roadmap](../../docs/specs/v2026-05-06-eval-system-roadmap.md).

What goes through the adapter:

| Concern | Framework | Why |
|---|---|---|
| Text content scoring (`G-Eval`, faithfulness, role adherence, …) | DeepEval | Their bread and butter. |
| Pytest test running, conversational metrics | DeepEval | Standard idioms; low friction. |
| Audio judges (affect, delivery) | rehearse `Scorer` | DeepEval has no first-class audio in/out. |
| Timing-derived `NaturalnessScorer` (Spec 4) | rehearse `Scorer` | Pure arithmetic; not an LLM judgment. |
| Meta scorers (`StabilityScorer`) | rehearse `MetaScorer` | DeepEval has no concept of grouped rollouts. |
| Rollout orchestration, sandbox env, executors | Custom | Voice-shaped; no DeepEval analogue. |

### Surface

```python
from rehearse.eval.deepeval_adapter import (
    to_conversational_test_case,   # (BenchmarkExample, RolloutResult) -> ConversationalTestCase
    to_llm_test_case,              # ... -> LLMTestCase (single-turn collapse)
    MetricToScorer,                # DeepEval BaseMetric -> rehearse Scorer
    ScorerToMetric,                # rehearse Scorer -> DeepEval BaseConversationalMetric
)
```

### Pattern A — DeepEval metric flowing through the rehearse runner

Use this when you want a DeepEval-authored metric (`G-Eval`, faithfulness,
etc.) to run alongside our custom scorers in `rehearse-eval run`:

```python
from deepeval.metrics import GEval
from deepeval.test_case import TurnParams
from rehearse.eval.deepeval_adapter import MetricToScorer

content_metric = GEval(
    name="ContentQuality",
    criteria=(
        "Evaluate whether the coach's responses moved the user toward "
        "clearer, calmer, actionable phrasing for their conversation."
    ),
    evaluation_params=[TurnParams.CONTENT, TurnParams.ROLE],
    threshold=0.5,
)

content_scorer = MetricToScorer(content_metric, dimension="content_quality")

# Drop content_scorer into your eval's `scoring_plan()` like any other Scorer.
```

### Pattern B — rehearse Scorer running inside DeepEval's `evaluate()`

Use this when you want to run our custom scorers as part of a DeepEval
test run (pytest-style assertions, comparison reports, etc.):

```python
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from rehearse.eval.deepeval_adapter import ScorerToMetric, to_conversational_test_case
from rehearse.eval.scorers.deterministic import MMERecognitionScorer

test_case = to_conversational_test_case(example, rollout)

metric = ScorerToMetric(MMERecognitionScorer(), threshold=0.5)
metric.bind_rollout(example, rollout)   # required for scorers that need
                                        # rollout artifacts (audio, timing)

result = evaluate(
    test_cases=[test_case],
    metrics=[metric],
    async_config=AsyncConfig(run_async=False),  # adapter manages concurrency
)
score = result.test_results[0].metrics_data[0].score
```

`ScorerToMetric` extends `BaseConversationalMetric` (coaching is multi-turn).
`bind_rollout` stores the originating `(example, rollout)` pair on the
scorer object so it survives DeepEval's metric cloning during evaluation.

### Live G-Eval smoke test

The adapter test suite includes a live-API smoke that hits a real LLM
through `G-Eval`:

```bash
OPENAI_API_KEY=sk-... uv run pytest tests/eval/deepeval/ -m live_api
```

Without `OPENAI_API_KEY`, the live test is skipped and the rest of the
suite still passes.
