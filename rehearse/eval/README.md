# rehearse-eval

Eval harness for the rehearse system. Plugin-shaped: evals, datasets, scorers,
environments, providers, and executors are small Python `Protocol`-style units.

The harness is independent of the runtime. You can run smoke evals with no live
phone path, no media files, and no model API keys.

Design specs:
- [`docs/specs/v2026-04-27-eval-harness.md`](../../docs/specs/v2026-04-27-eval-harness.md)
- [`docs/specs/v2026-04-28-mme-emotion-and-audio-targets.md`](../../docs/specs/v2026-04-28-mme-emotion-and-audio-targets.md)

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

This patch checks in the manifest scaffold only. Before a real run, place the
referenced media files under `evals/datasets/mme-emotion/v0-10clip/clips/` or
set `MME_EMOTION_MANIFEST_PATH` to another manifest with valid paths.

## Running Tests

```bash
uv run pytest tests/eval/
```

Tests cover protocol conformance, runner end-to-end, subprocess isolation, the
MME-Emotion dataset/eval shape, and the deterministic recognition scorer.
