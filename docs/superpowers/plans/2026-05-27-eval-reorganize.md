# Eval Reorganization Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate dead code, fix confusing naming, and consolidate the eval harness so the structure matches how you'd explain it to a new contributor.

**Before → After:**
```
rehearse/eval/
  evals/          →  suites/        (removes eval/evals/ naming collision)
  customers/      →  drivers/       (clearer: these drive the caller side)
  providers/      →  judges/        (clearer: these back the LLM judges)
  executors/      →  harness/       (merged into single executor.py)
  report.py       →  harness/
  score_stream.py →  harness/stream.py
  watch.py        →  harness/
  worker.py       →  harness/
  tts_bridge.py   →  environments/tts_bridge.py
  targets/        →  environments/utils/   (already done in restructure branch)
  multimodal_llm  →  environments/media_probe.py
  evals/mme_emotion → benchmarks/mme_emotion.py
```

**What gets deleted (dead code — deprecated voice-agent-sandbox cluster):**
- `environments/voice_agent_sandbox.py` (deprecated; emits warning + delegates)
- `environments/live_rollout_audio.py` (wraps the deprecated env)
- `sandboxes.py`, `sandbox_agents.py`, `sandbox_connection.py`
- `synthetic_user.py`, `prosody_scripts.py` (docstring stubs, zero code)
- `evals/voice_agent_smoke.py`, `evals/coach_dialogue_smoke.py`, `evals/mme_sandbox_rollout.py`
- `datasets/voice_agent_smoke.py`, `datasets/coach_dialogue_smoke.py`, `datasets/mme_rollout_seeds.py`
- `benchmarks/` (pure shim wrappers — `class MMEEmotionBenchmark(MMEEmotionEval): pass`)
- Tests for all of the above

**Verification:** `rehearse-eval run --eval noop --environment echo --limit 1` and `rehearse-eval run --eval voice-judges-smoke --limit 1`

---

### Task 1: Delete dead code

- [ ] `git rm` the voice-agent-sandbox cluster and its stub files
- [ ] `git rm` the benchmarks/ shim directory
- [ ] `git rm` tests that covered the deleted code
- [ ] Remove deleted entries from `environments/__init__.py` and `evals/__init__.py` and `datasets/__init__.py`

Files to delete:
```
rehearse/eval/environments/voice_agent_sandbox.py
rehearse/eval/environments/live_rollout_audio.py
rehearse/eval/sandboxes.py
rehearse/eval/sandbox_agents.py
rehearse/eval/sandbox_connection.py
rehearse/eval/synthetic_user.py
rehearse/eval/prosody_scripts.py
rehearse/eval/evals/voice_agent_smoke.py
rehearse/eval/evals/coach_dialogue_smoke.py
rehearse/eval/evals/mme_sandbox_rollout.py
rehearse/eval/datasets/voice_agent_smoke.py
rehearse/eval/datasets/coach_dialogue_smoke.py
rehearse/eval/datasets/mme_rollout_seeds.py
rehearse/eval/benchmarks/__init__.py
rehearse/eval/benchmarks/mme_emotion.py
rehearse/eval/benchmarks/noop.py
tests/eval/test_coach_dialogue_smoke.py
tests/eval/test_sandboxes.py
tests/eval/test_voice_agent_sandbox_environment.py
tests/eval/test_mme_sandbox_rollout.py
```

Verify: `uv run pytest tests/ --ignore=tests/integration -q`

Commit: `refactor(eval): delete deprecated voice-agent-sandbox cluster and shims`

---

### Task 2: Create harness/ package

Move infrastructure files out of the flat top level.

- [ ] Create `rehearse/eval/harness/__init__.py`
- [ ] `git mv rehearse/eval/report.py rehearse/eval/harness/report.py`
- [ ] `git mv rehearse/eval/score_stream.py rehearse/eval/harness/stream.py`  (rename)
- [ ] `git mv rehearse/eval/watch.py rehearse/eval/harness/watch.py`
- [ ] `git mv rehearse/eval/worker.py rehearse/eval/harness/worker.py`
- [ ] Merge `executors/in_process.py` + `executors/local_subprocess.py` → `harness/executor.py`; delete `executors/`
- [ ] Update `score_stream` → `stream` references in `ScoreStreamWriter` class (it self-references the filename in sentinel logic)
- [ ] Update imports in `runner.py`, `cli.py`, `watch.py` (now `harness.stream`, `harness.report`, etc.)
- [ ] Update imports in `tests/test_eval_streaming.py` and `tests/eval/test_local_subprocess.py`

Verify: `uv run pytest tests/ --ignore=tests/integration -q`

Commit: `refactor(eval): consolidate harness infrastructure into harness/ package`

---

### Task 3: Rename evals/ → suites/

- [ ] `git mv rehearse/eval/evals rehearse/eval/suites`
- [ ] Update all imports: `rehearse.eval.evals` → `rehearse.eval.suites`
  - `runner.py`, `cli.py`, `harness/worker.py`
  - `tests/eval/test_runner.py`, `tests/eval/test_protocols.py`, `tests/eval/test_mme_emotion_eval.py`
- [ ] Update `suites/__init__.py` docstring

Verify: `uv run pytest tests/ --ignore=tests/integration -q`

Commit: `refactor(eval): rename evals/ → suites/ to remove eval.evals naming collision`

---

### Task 4: Rename customers/ → drivers/, providers/ → judges/

- [ ] `git mv rehearse/eval/customers rehearse/eval/drivers`
- [ ] `git mv rehearse/eval/providers rehearse/eval/judges`
- [ ] Update all imports: `rehearse.eval.customers` → `rehearse.eval.drivers`
  - `environments/runtime_sandbox.py`, `environments/live_audio_sandbox.py`, `environments/audio_fixture.py`
  - `tests/test_audio_customer_driver.py`, `tests/test_llm_customer_phase_aware.py`, `tests/test_runtime_sandbox_audio.py`, `tests/test_runtime_sandbox_rollout.py`
- [ ] Update all imports: `rehearse.eval.providers` → `rehearse.eval.judges`
  - `environments/multimodal_llm.py` (or media_probe.py after Task 5), `cli.py`
- [ ] Update `__init__.py` docstrings in both

Verify: `uv run pytest tests/ --ignore=tests/integration -q`

Commit: `refactor(eval): rename customers/ → drivers/, providers/ → judges/`

---

### Task 5: Move echo/text_probe → environments/utils/, rename multimodal_llm → media_probe, move tts_bridge

- [ ] Create `rehearse/eval/environments/utils/__init__.py`
- [ ] Create `rehearse/eval/environments/utils/echo.py` (from `targets/echo.py`)
- [ ] Create `rehearse/eval/environments/utils/text_probe.py` (from `targets/raw_llm.py`, class renamed)
- [ ] `git rm rehearse/eval/targets/`
- [ ] `git mv rehearse/eval/environments/multimodal_llm.py rehearse/eval/environments/media_probe.py`
- [ ] `git mv rehearse/eval/tts_bridge.py rehearse/eval/environments/tts_bridge.py`
- [ ] Update imports of `tts_bridge` in `drivers/audio_customer.py`, `drivers/eval_caller.py`, `environments/runtime_sandbox.py`, `environments/audio_fixture.py`, `environments/live_audio_sandbox.py`
- [ ] Update `environments/__init__.py`: new import paths + rename `MultimodalLLMEnvironment` → `MediaProbeEnvironment`
- [ ] Update `benchmarks/mme_emotion.py` environment reference (still uses `"multimodal-llm"` string name — no change needed there)

Verify: `uv run pytest tests/ --ignore=tests/integration -q`

Commit: `refactor(eval): reorganize environments — utils/, media_probe, tts_bridge colocation`

---

### Task 6: Move mme_emotion eval → benchmarks/

- [ ] Create `rehearse/eval/benchmarks/__init__.py` (fresh, not the old shim)
- [ ] `git mv rehearse/eval/suites/mme_emotion.py rehearse/eval/benchmarks/mme_emotion.py`
- [ ] Update `benchmarks/__init__.py` to register `mme-emotion` eval
- [ ] Update `suites/__init__.py` to remove `mme-emotion` entry
- [ ] Update `runner.py` / `cli.py` to also query `benchmarks` registry (or merge registries)
- [ ] Update `tests/eval/test_mme_emotion_eval.py` import path

Verify: `uv run pytest tests/ --ignore=tests/integration -q`

Commit: `refactor(eval): move mme_emotion eval to benchmarks/ (external open-source benchmark)`

---

### Task 7: Verify evals run end to end

- [ ] `uv run rehearse-eval run --eval noop --environment echo --limit 1`
- [ ] `uv run rehearse-eval run --eval voice-judges-smoke --limit 1`
- [ ] `uv run rehearse-eval list-evals` (confirm all expected evals appear)
- [ ] `uv run rehearse-eval list-environments` (confirm all expected environments appear)

---

### Task 8: Push and open PR

- [ ] `git push -u origin eval-reorganize`
- [ ] `gh pr create --base main`
