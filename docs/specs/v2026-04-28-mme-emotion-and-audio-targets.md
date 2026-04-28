# rehearse — Spec: MME-Emotion + Audio-Native Model Slots

**Status**: draft (build handoff)
**Owner**: jz
**Depends on**: `docs/specs/v2026-04-27-eval-harness.md` (Phases 1–2 shipped)
**Supersedes**: EQ-Bench adapter and `raw-llm`-as-primary-target

---

## 0. One-line summary

Drop EQ-Bench. Add an audio-native evaluation path: a `multimodal-llm` target that loads audio (or video with audio) and submits it to an `AudioLLMProvider`, with two providers in v0 — Gemini 2.5 Pro (hosted, frontier baseline) and Gemma 4 E4B served via vLLM (open-weights, the fine-tune target). First benchmark on this path is MME-Emotion, scoped to a 10-clip subset for the tight build loop, with two scorers: recognition (deterministic accuracy) and reasoning (LLM judge, Claude Opus).

## 1. What this changes and why

EQ-Bench is text-only and stale (no commits in two years). The product is prosody-aware coaching — the words matter less than the tone. A text benchmark cannot measure that. MME-Emotion was published a month ago, is multimodal, has an active research community, and aligns with the rehearse strategy of building a purpose-built audio LLM rather than wrapping a frontier provider.

The strategic frame: **two model slots evaluated on the same benchmark**.

| Slot | Provider | Role |
|---|---|---|
| `multimodal_hosted` | Gemini 2.5 Pro | Frontier baseline. The number to beat. |
| `multimodal_open` | Gemma 4 E4B via vLLM | The fine-tune target. What rehearse owns. |

Long-term, only the second slot matters — it's where DPO on collected session preference pairs lands. Gemini exists in eval to keep us honest about whether the open-weights candidate is competitive.

## 2. Decisions locked

These were resolved before this spec; they are not open questions.

1. **Drop EQ-Bench entirely** — adapter, sample data, README section. Keep `raw-llm` target itself for occasional text diagnostics; it's just no longer the primary path.
2. **Hosted slot**: Gemini 2.5 Pro (audio-native via `google-genai` SDK).
3. **Open-weights slot**: Gemma 4 E4B (4.5B effective params, Apache 2.0, audio up to 30s, served via vLLM behind an OpenAI-compatible endpoint).
4. **Subset for v0**: 10 clips, hand-curated from `ER_Lab` for length-distribution reasons (lab-recorded, mostly under 30s).
5. **Both scorers in v0**: recognition (deterministic) + reasoning (LLM judge, Claude Opus).
6. **vLLM serving lives outside the eval harness.** The harness reads `VLLM_BASE_URL` + `VLLM_API_KEY` and calls a remote OpenAI-compatible endpoint. Spinning up the vLLM server is a separate ops concern, documented but not implemented in this spec.

## 3. Architecture

```
                          ┌────────────────────────────┐
                          │      rehearse-eval         │
                          │     (CLI, unchanged)       │
                          └─────────────┬──────────────┘
                                        │
                                        ▼
                          ┌────────────────────────────┐
                          │         Runner             │
                          │     (unchanged)            │
                          └─────────────┬──────────────┘
                                        │
              ┌─────────────────────────┼──────────────────────────┐
              ▼                         ▼                          ▼
       MMEEmotionBenchmark   LocalSubprocessExecutor      [Recognition +
       loads .json +         (unchanged)                   Reasoning]
       resolves audio paths           │                    Scorers
              │                       ▼                          │
              │              ┌────────────────────┐               │
              │              │  multimodal-llm    │  NEW          │
              │              │   target           │               │
              │              └─────────┬──────────┘               │
              │                        │                          │
              │           ┌────────────┴───────────┐              │
              │           ▼                        ▼              │
              │   AudioLLMProvider         AudioLLMProvider       │
              │     (Gemini)                 (vLLM/Gemma)         │
              │           │                        │              │
              │           │                        │              │
              ▼           ▼                        ▼              ▼
      example.payload  google-genai      OpenAI-compat client   RubricScore
       with video_path     │              (vLLM endpoint)         rows
                           ▼                        ▼
                  Gemini API (cloud)       vLLM server (GPU)
                                            outside this repo
```

What's new in this spec, file-by-file:

| File | Status | Purpose |
|---|---|---|
| `rehearse/eval/providers/__init__.py` | new | Provider registry |
| `rehearse/eval/providers/base.py` | new | `AudioLLMProvider` protocol |
| `rehearse/eval/providers/gemini.py` | new | Google Gemini via `google-genai` |
| `rehearse/eval/providers/vllm.py` | new | OpenAI-compatible client pointed at a vLLM server |
| `rehearse/eval/targets/multimodal_llm.py` | new | Loads audio/video, calls provider, returns `RolloutResult` |
| `rehearse/eval/benchmarks/mme_emotion.py` | new | MME-Emotion adapter |
| `rehearse/eval/scorers/llm_judge.py` | promoted from stub | Reasoning scorer (Claude Opus) + reusable judge primitive |
| `rehearse/eval/scorers/deterministic.py` | extended | `MMERecognitionScorer` (label exact match) |
| `evals/benchmarks/mme-emotion/v0-10clip/manifest.json` | new (vendored, small) | 10 hand-picked clip ids + ground truth |
| `evals/benchmarks/mme-emotion/v0-10clip/clips/*.mp4` | new (vendored, ~50 MB) | The actual audio/video files for offline runs |
| `scripts/fetch_mme_emotion.py` | new | Pulls full dataset from HF for larger runs (out of v0 scope) |
| `rehearse/eval/benchmarks/eq_bench.py` | **deleted** | Strip EQ-Bench |
| `evals/benchmarks/eq-bench/sample/questions.json` | **deleted** | Strip EQ-Bench data |
| `tests/eval/test_eq_bench_adapter.py` | **deleted** | Strip EQ-Bench tests |

What is unchanged: `protocols.py` (the four core protocols), `runner.py`, `worker.py`, `cli.py`, `executors/`, `targets/echo.py`, `targets/raw_llm.py`, `benchmarks/noop.py`, `types.py`. The provider plugin layer slots cleanly under the existing target abstraction.

## 4. Components

### 4.1 `AudioLLMProvider` protocol — `rehearse/eval/providers/base.py`

```python
class AudioLLMProvider(Protocol):
    name: str                     # "gemini-2.5-pro", "gemma-4-e4b-vllm"
    version: str                  # SDK or model build identifier

    async def complete(
        self,
        audio: AudioInput,        # bytes | path; provider decides how to upload/encode
        prompt: str,              # task instruction
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout_s: int = 60,
    ) -> ProviderResponse: ...
```

```python
@dataclass(frozen=True)
class AudioInput:
    path: Path                    # source file (mp4 or wav)
    duration_s: float             # measured at load time
    extracted_audio_path: Path | None  # set when video → audio extracted

@dataclass
class ProviderResponse:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    audio_duration_s: float | None
    raw_finish_reason: str | None
    provider_metadata: dict[str, Any] = field(default_factory=dict)
```

The protocol is `runtime_checkable` (mirrors `Target` etc.). Provider construction takes a `dict[str, str]` of model slots, same as targets, so the CLI's `--model-slot` flag continues to work.

### 4.2 Gemini provider — `rehearse/eval/providers/gemini.py`

- SDK: `google-genai` (the modern unified SDK, not deprecated `google-generativeai`).
- Auth: `GOOGLE_API_KEY` env var. Validated at provider init.
- Uploads: small files (<20 MB) inline base64; larger via the Files API. Threshold lives in a constant.
- Default model: `gemini-2.5-pro`. Override via `--model-slot multimodal_hosted=gemini-2.5-flash` etc.
- Failure handling: any 4xx/5xx → `ProviderResponse` is not produced; raises `ProviderError` which the target catches and converts to `RolloutResult(status="error")`.

### 4.3 vLLM provider — `rehearse/eval/providers/vllm.py`

- SDK: the `openai` Python client pointed at `VLLM_BASE_URL` (e.g. `http://gpu-host:8000/v1`).
- Auth: `VLLM_API_KEY` (vLLM accepts any string as a bearer token by default; may be `dummy` in dev).
- Audio passing: vLLM exposes audio via the OpenAI multimodal `chat.completions` schema. Audio bytes are base64'd into a content part of type `input_audio`. (Confirm exact field shape in build phase against the deployed vLLM version — see Open Question V1.)
- Default model: `gemma-4-e4b`. Override via `--model-slot multimodal_open=...`.
- Failure handling: same as Gemini.

The vLLM provider does not start, manage, or health-check the vLLM server itself. That is the operator's job. The provider assumes a reachable, model-loaded endpoint at startup and surfaces a clear error if not.

### 4.4 `multimodal-llm` target — `rehearse/eval/targets/multimodal_llm.py`

```python
class MultimodalLLMTarget:
    name = "multimodal-llm"
    version = "v0"

    def __init__(self, model_slots: dict[str, str], provider_name: str | None = None) -> None:
        ...
```

Behavior:

1. Read `example.payload`:
   - `video_path` (or `audio_path`) — required, relative to repo or absolute
   - `prompt` — the task instruction the provider sees
   - `audio_max_s` — optional, hard cap; clip is skipped if exceeded
2. Load the file, measure duration, extract audio track if it's a video and the provider needs raw audio (Gemma path; Gemini takes video directly).
3. Call `provider.complete(audio, prompt, ...)`.
4. Wrap into `RolloutResult` with `payload = {"output": text, "model": ..., "provider": provider.name, "audio_duration_s": ..., ...}`.
5. On clip-too-long, status=`"error"` with a structured `error="audio_exceeds_provider_limit"` so the recognition scorer can distinguish "model got it wrong" from "we never asked the model."

Provider selection at runtime: target reads model_slots `multimodal_hosted` and `multimodal_open`; the active one is chosen by `--provider gemini|vllm` (CLI flag added) or by a payload-level hint. If both are set, exactly one must be selected per run, never both.

### 4.5 MME-Emotion adapter — `rehearse/eval/benchmarks/mme_emotion.py`

- Loads `evals/benchmarks/mme-emotion/v0-10clip/manifest.json`.
- 10 hand-curated clips from `ER_Lab` (lab-recorded, mostly <30s, balanced across 9 emotion classes — at least 8 of 9 represented; 1 class may double up given the 10-clip cap).
- One `BenchmarkExample` per clip:

```python
BenchmarkExample(
    id="mme-er-lab-001",
    benchmark="mme-emotion",
    payload={
        "video_path": "evals/benchmarks/mme-emotion/v0-10clip/clips/Ses05M_script01_1_F034.mp4",
        "audio_max_s": 30,
        "prompt": (
            "You are listening to a short clip of a person speaking. "
            "Classify the emotion they are expressing. Respond in JSON: "
            "{\"label\": <one of: Anger, Sadness, Surprise, Happiness, "
            "Excited, Fear, Frustration, Neutral, Other>, "
            "\"reasoning\": <2-4 sentences citing tone, pacing, and word choice>}"
        ),
        "label_set": [
            "Anger", "Sadness", "Surprise", "Happiness", "Excited",
            "Fear", "Frustration", "Neutral", "Other",
        ],
    },
    expected={"label": "Frustration"},
    metadata={"subset": "ER_Lab", "duration_s": 8.4, "speaker_id": "Ses05M_M_F034"},
)
```

Scoring plan returns both scorers. Rollout timeout: 90 s (allows for slow Gemma cold starts).

The 10-clip selection is hand-curated rather than randomly sampled because:
- We need every emotion class represented.
- Clip duration must be under both providers' limits (30s for Gemma, generous for Gemini).
- We want 1–2 ambiguous-on-purpose clips (e.g. Frustration vs Anger boundary) to make the benchmark non-trivial at 10 examples.

The 10-clip manifest is checked into git. The actual `.mp4` files are vendored (estimate ~50 MB total) — also in git, since the repo currently has no LFS setup and 50 MB is acceptable.

### 4.6 Recognition scorer — `rehearse/eval/scorers/deterministic.py`

```python
class MMERecognitionScorer:
    name = "mme_recognition"
    dimension = "mme_recognition_accuracy"   # benchmark-private string
```

- Parse `payload["output"]` for a JSON object with a `label` field. Reuse the same tolerant `parse_eq_ratings` regex pattern, generalized.
- Compare `predicted_label.lower().strip() == expected.label.lower().strip()`.
- `RubricScore.value`: 1.0 on exact match, 0.0 otherwise.
- Aggregate (mean across examples) → accuracy.
- On rollout error: 0.0 with rationale `"rollout {status}: {error}"`.
- On unparseable output: 0.0 with rationale `"could not parse label from output"`.

### 4.7 Reasoning scorer — `rehearse/eval/scorers/llm_judge.py`

LLM-judge scorer using **Claude Opus 4.7** (matches the rest of the rehearse stack — see SPEC §9).

```python
class MMEReasoningScorer:
    name = "mme_reasoning"
    dimension = "mme_reasoning_score"
```

Judge prompt (verbatim):

> You are scoring a model's reasoning about an emotion classification it made
> from a short audio clip. You will be given:
> - The ground-truth emotion label.
> - The model's predicted label.
> - The model's free-text reasoning.
>
> Rate the reasoning on a scale of 0 to 10:
> - 10 — reasoning cites specific prosodic features (pitch, pacing, intensity,
>   pauses) and word choice consistent with the emotion. Coherent. Grounded
>   in what would be audible in the clip.
> - 5 — reasoning is general but defensible. Names the emotion's typical
>   features without specific citations.
> - 0 — reasoning is generic, contradictory, hallucinates content not
>   present, or amounts to "the speaker sounds X."
>
> The score is independent of whether the predicted label matched ground
> truth. A wrong label can still have well-grounded reasoning.
>
> Respond ONLY with JSON: {"score": <0-10 integer>, "rationale": "<one
> sentence on what determined the score>"}.

- Parse, return one `RubricScore` per example with `value` ∈ [0, 10].
- Aggregate as mean across examples.
- Calibration is not done in v0. Gold-set judge calibration (SPEC §10.4) lands when we have ≥10 human-labeled gold examples — out of scope here.

Cost: Opus on 10 short prompts is ~$0.50 per run. Acceptable.

### 4.8 CLI additions

One new flag on `rehearse-eval run`:

```
--provider {gemini|vllm}   # required when target=multimodal-llm
                           # raises if benchmark doesn't support multimodal-llm
```

`list-providers` subcommand added for symmetry with `list-benchmarks` / `list-targets`.

The existing `--model-slot` flag is reused: `--model-slot multimodal_hosted=gemini-2.5-pro` or `--model-slot multimodal_open=gemma-4-e4b`.

### 4.9 Environment variables

| Var | Purpose | Required when |
|---|---|---|
| `GOOGLE_API_KEY` | Gemini auth | `--provider gemini` |
| `VLLM_BASE_URL` | vLLM endpoint, e.g. `http://gpu-host:8000/v1` | `--provider vllm` |
| `VLLM_API_KEY` | bearer token for the vLLM endpoint (`dummy` in dev) | `--provider vllm` |
| `ANTHROPIC_API_KEY` | Claude Opus for the reasoning scorer | always (with mme-emotion benchmark) |

Validated at provider/scorer construction. Missing → fail-fast at runner startup, not at first rollout.

## 5. Storage & data plumbing

### 5.1 v0: 10 clips checked into git

`evals/benchmarks/mme-emotion/v0-10clip/` contains:
- `manifest.json` — 10 entries with id, label, video filename, duration, subset
- `clips/*.mp4` — actual files, ~50 MB total

Acceptable in git for v0. Not LFS, not externally hosted.

### 5.2 Future: full 6.3 GB via fetch script

`scripts/fetch_mme_emotion.py` (skeleton in this spec, not invoked in v0):

- Downloads from `Karl28/MME-Emotion` on Hugging Face.
- Verifies SHA against pinned commit.
- Writes to `evals/benchmarks/mme-emotion/{commit-sha}/`.
- Updates `MMEEmotionBenchmark.version` to the SHA when invoked.

Not built in v0, but the directory layout (`{commit-sha}/`) anticipates it. The 10-clip subset uses `v0-10clip/` as a logical "tag" alongside future SHA dirs.

### 5.3 Subprocess executor implications

Audio bytes are large; we already pass `video_path` (a string) through the executor's stdin JSON, not bytes. The worker opens the file. No changes to `local_subprocess.py`.

## 6. Strip plan: removing EQ-Bench

In one PR, separately reviewable:

1. Delete `rehearse/eval/benchmarks/eq_bench.py`.
2. Delete `evals/benchmarks/eq-bench/` (sample data + directory).
3. Delete `tests/eval/test_eq_bench_adapter.py`.
4. Remove EQ-Bench from `BENCHMARKS` registry in `rehearse/eval/benchmarks/__init__.py`.
5. Remove EQ-Bench-specific helpers from `rehearse/eval/scorers/deterministic.py` (`EQBenchCorrelationScorer`, `parse_eq_ratings`, `_pearson` — generalize-or-delete).
   - `parse_eq_ratings` → keep, rename to `parse_json_object_with_keys`, use in MMERecognitionScorer.
   - `_pearson` → delete; no current scorer uses it.
   - `EQBenchCorrelationScorer` → delete.
6. Remove the EQ-Bench section from `rehearse/eval/README.md`.
7. Remove EQ-Bench mention from the root `README.md`.

This PR can ship before any new code lands. The harness still has `noop` + `echo` for skeleton smoke tests.

## 7. Phasing

Each phase ends with green tests + a working manual demo. Phases are reviewable PRs.

**Phase M1 — Strip EQ-Bench, prep ground.**
- Execute §6 strip plan.
- Update `rehearse/eval/README.md` with placeholder for upcoming MME-Emotion section.
- Tests: `noop` smoke run, registry no longer lists `eq-bench`.

**Phase M2 — Provider plugin layer + Gemini provider.**
- `providers/base.py` (`AudioLLMProvider`, `AudioInput`, `ProviderResponse`, `ProviderError`).
- `providers/gemini.py`.
- `providers/__init__.py` registry + `list-providers` CLI subcommand.
- Tests: protocol conformance, mocked Gemini response shaping, env-var validation, provider registry.
- Demo: `rehearse-eval list-providers` lists `gemini`.

**Phase M3 — `multimodal-llm` target + 10-clip MME-Emotion + recognition scorer.**
- `targets/multimodal_llm.py`.
- `benchmarks/mme_emotion.py`.
- `scorers/deterministic.py::MMERecognitionScorer`.
- Hand-curate 10 clips into `evals/benchmarks/mme-emotion/v0-10clip/` (this is partly research work; see Open Question D1).
- Tests: target + adapter + scorer in isolation; runner end-to-end against a stubbed provider that returns scripted JSON.
- Demo: `rehearse-eval run --benchmark mme-emotion --target multimodal-llm --provider gemini --limit 10` produces real recognition scores against Gemini.

**Phase M4 — Reasoning scorer (Claude Opus judge).**
- Promote `scorers/llm_judge.py` from stub.
- `MMEReasoningScorer`.
- Reusable `LLMJudge` primitive for future judges.
- Tests: judge prompt fixture, parse contract, fallback on unparseable judge output.
- Demo: full `summary.md` shows both `mme_recognition_accuracy` and `mme_reasoning_score`.

**Phase M5 — vLLM provider.**
- `providers/vllm.py`.
- Tests: protocol conformance, mocked OpenAI client.
- Demo: with a reachable `VLLM_BASE_URL`, `rehearse-eval run --provider vllm` produces real Gemma scores. **Gated on a live vLLM endpoint outside this repo.**

**Phase M6 — Side-by-side comparison report.**
- `rehearse-eval diff <gemini_run_id> <vllm_run_id>` (this is a generalization of the deferred `diff` command from the eval-harness spec).
- Per-dimension delta table.
- Demo: comparing a Gemini run vs a Gemma run on the same 10 clips, in one command.

Phases M1–M4 are pure code, runnable today. Phase M5 needs ops setup. Phase M6 closes the loop on "purpose-built model is competitive with frontier."

## 8. vLLM operational notes (out-of-band)

Out of scope for this spec to implement, but documented so the eval engineer knows what they're targeting.

Recommended bootstrap on a single GPU host (RunPod, Lambda, or local with an A100/H100):

```bash
pip install vllm
vllm serve google/gemma-4-e4b-it \
    --port 8000 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --enable-audio-input  # exact flag may vary by vLLM version
```

Verify reachability:

```bash
curl $VLLM_BASE_URL/models -H "Authorization: Bearer $VLLM_API_KEY"
```

The eval harness assumes this endpoint exists and is OpenAI-compatible. If vLLM's audio support flag differs from above, only the operator command changes — the provider code does not.

A `docs/runbooks/vllm-gemma.md` should be authored alongside Phase M5; that's the place for full deployment notes (GPU sizing, model download, monitoring, restart policy).

## 9. Open questions

| # | Question | Phase needed by |
|---|---|---|
| V1 | Exact JSON shape of audio input in vLLM's OpenAI-compatible chat-completions schema. The OpenAI API uses `input_audio` content parts; vLLM's compatibility may vary by version. The provider needs to match what the deployed vLLM accepts. | M5 |
| V2 | Does Gemma 4 E4B handle our prompt format well at audio classification? May need prompt tuning specifically for Gemma vs Gemini. Could lead to different prompt strings per provider — or, preferably, one prompt that works for both. | M3 / M5 |
| D1 | Subset selection: which 10 clips? Needs research access to the full MME-Emotion data on HuggingFace. Hand-curation by jz; criteria in §4.5. | M3 (blocks demo) |
| D2 | Clip-length policy when a clip exceeds 30s on Gemma but fits on Gemini: skip on Gemma, run on Gemini, score asymmetric? Or trim-to-30s on both for fairness? Recommendation: **skip + log on Gemma**, document in summary. | M5 |
| J1 | Reasoning judge calibration: do we need a small human-labeled gold set (5 reasoning samples) for the reasoning scorer to be trustworthy? The MME-Emotion paper used GPT-4o + 5 human experts; we lean on Opus alone in v0. | M4 |
| C1 | Cost: at full 6,500 clips, Gemini audio + Opus judge cost ~$30–80 per full run. Acceptable budget per major change? | full-set runs (post-v0) |
| L1 | License: Apache 2.0 on Gemma is fine; MME-Emotion is MIT; both safe to vendor and ship. **Verify before ingesting any clip outside the 10-clip subset.** | M3 |

## 10. Out of scope

Explicitly not in this spec:

- Full 6,500-clip runs. v0 stays at 10. Larger runs are a Phase M5+ concern.
- HuggingFace dataset fetch script implementation (skeleton only).
- LFS migration for the 10 clips. v0 ships them in regular git.
- Fine-tuning Gemma 4 on rehearse-collected preference pairs. That's the v1 ML work the *eval result* feeds into; this spec ends at "eval is honest and runnable."
- Replacing the runtime spec's persona compiler / synthesis prompts. Those are independent.
- Adding video understanding beyond audio for emotion. MME-Emotion has video; v0 treats it as an audio benchmark and ignores the visual track. (Both Gemini 2.5 Pro and Gemma 4 can process the video too — added in a later phase if scoring deltas suggest visual signal matters for the task.)

## 11. Why this is the right shape

Three claims worth being explicit about, since they'll feel non-obvious during build:

1. **The provider plugin layer is not over-engineering.** It exists because the eval question we actually want answered is "does our open-weights candidate match frontier?" That requires running the same target + benchmark + scorers against ≥2 providers. Without the plugin, we'd have a `gemini-target` and a `gemma-target` with 90% identical code, and the comparison logic would live in the wrong place.

2. **vLLM as a separate concern is intentional.** Coupling the eval harness to vLLM lifecycle (start/stop/health) makes the harness fragile and the comparison less honest. Operators run vLLM however they like; the harness just makes calls. This is the same discipline as the runtime spec's separation of `ArtifactStore` from concrete backends.

3. **10 clips is enough to be useful in v0.** The number we want from this is *signal* — does Gemma get within 80% of Gemini on recognition? Does its reasoning hold up? Ten balanced clips with both scorers gives that signal in a 5-minute loop. Scaling to 100, then full 6500, is mechanical once the loop is tight.
