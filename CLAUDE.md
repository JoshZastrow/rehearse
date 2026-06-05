# CLAUDE.md

## GBrain Configuration (configured by /setup-gbrain)
- Mode: local-stdio
- Engine: pglite
- Config file: ~/.gbrain/config.json (mode 0600)
- Setup date: 2026-05-07
- MCP registered: yes (user scope)
- Artifacts repo: https://github.com/JoshZastrow/gstack-artifacts-joshuazastrow
- Artifacts sync: full
- Current repo policy: read-write

## GBrain Search Guidance (configured by /sync-gbrain)
<!-- gstack-gbrain-search-guidance:start -->

GBrain is set up and synced on this machine. The agent should prefer gbrain
over Grep when the question is semantic or when you don't know the exact
identifier yet. Two indexed corpora available via the `gbrain` CLI:
- This repo's code (registered as `gstack-code-rehearse` source).
- `~/.gstack/` curated memory (registered as `gstack-artifacts-joshuazastrow`
  federated source).

Prefer gbrain when:
- "Where is X handled?" / semantic intent, no exact string yet:
    `gbrain search "<terms>"` or `gbrain query "<question>"`
- "Where is symbol Y defined?" / symbol-based code questions:
    `gbrain code-def <symbol>` or `gbrain code-refs <symbol>`
- "What calls Y?" / "What does Y depend on?":
    `gbrain code-callers <symbol>` / `gbrain code-callees <symbol>`
- "What did we decide last time?" / past plans, retros, learnings:
    `gbrain search "<terms>" --source gstack-artifacts-joshuazastrow`

Grep is still right for known exact strings, regex, multiline patterns, and
file globs. The brain auto-syncs incrementally on every gstack skill start.
Run `/sync-gbrain` to force-refresh, `/sync-gbrain --full` for full reindex.

<!-- gstack-gbrain-search-guidance:end -->

---

# Technical Vision: Continual Learning ML System — Strategic Report

> **Vision:** Data + train time + compute + model size + numeric stability + diversity of data = scale and revenue growth. Runtime learning on experiences.
>
> *Research basis: 31 sources fetched, 132 claims extracted, 25 adversarially verified (3-vote), 6 confirmed. Sources span ICLR 2025, TMLR 2024, and arXiv 2024–2025.*

---

## Core Area 1: Continual Learning Architecture (Plasticity-Stability Balance)

**The problem:** Naive sequential fine-tuning causes catastrophic forgetting, and the severity *scales with model size*. BLOOMZ reading comprehension forgetting increases from 18% at 1.1B parameters to 27% at 7.1B; domain knowledge forgetting doubles over the same range. Bigger models — the very ones worth deploying — forget the most.

**The solution space:**

| Method | Best For | Key Tradeoff |
|---|---|---|
| **I-LoRA** (dual-memory EMA) | Sequential NLP tasks at 7B+ | Requires two adapter copies + replay buffer; backward transfer -0.6% vs -7.0% for plain ER |
| **CURLoRA** (CUR decomposition) | Sequential task fine-tuning, memory constrained | 384× fewer trainable params than LoRA-16; C/R matrices still occupy total memory |
| **Full fine-tuning** | Large domain shifts (code, math, hard CPT) | Best raw performance; full gradient memory, expensive to checkpoint per-task |
| **EWC** | Theoretically grounded; avoid in practice | Fisher matrix per task; degrades under high lambda; did not survive adversarial verification at scale |

**3 Operational Heuristics:**

1. **Default to I-LoRA at 7B+ for sequential NLP tasks.** Forgetting is worst at this scale, and the adapter EMA is the cheapest available mitigation. The dual-memory update `θˡ = λ·θˡ_prev + (1-λ)·θʷ` adds negligible compute overhead.

2. **Size replay buffers relative to per-task data volume, not absolute tokens.** On imbalanced corpora, equal-weight sampling from a fixed-token buffer over-represents small tasks and under-represents large ones — which inverts the forgetting problem. Buffer size should track the relative weight of each task in the training mixture.

3. **Track backward transfer (BWT) per task transition, not just aggregate accuracy.** A single catastrophic transition (e.g., T3→T5 in financial domain sequences) can destroy a model that looks fine on aggregate metrics. Monitor BWT per edge in the task graph at every training run.

---

## Core Area 2: Compute & Training Efficiency (Regime Selection)

**The problem:** The choice between full fine-tuning and PEFT (LoRA, CURLoRA) is the single highest-leverage decision in a training cycle. Making it wrong wastes compute or caps model quality. LoRA at its best rank (256) on code continued pretraining achieves HumanEval=0.224 while full fine-tuning at the same compute budget achieves 0.263 — a meaningful quality gap that compounds with scale.

**The decision gate:**

```
Is the task a large domain shift (code, math, hard CPT)?
  → YES: Full fine-tuning. LoRA's rank ceiling cannot represent
         the necessary weight changes regardless of rank chosen.
  → NO: Is sequential task adaptation the regime, with forgetting
        as the primary risk?
    → YES: CURLoRA (384× fewer trainable params, better BWT).
    → NO: LoRA with rank sweep before committing.
```

**Key numbers to internalize:**
- Full FT (code CPT, 20B tokens): HumanEval 0.263 vs LoRA-256's 0.224
- CURLoRA trainable params: 24,576 vs LoRA-16's 9,437,184 on Mistral (384× reduction)
- CURLoRA sequential task MRPC accuracy after 2 subsequent fine-tunes: 0.66 vs LoRA's 0.32

**3 Operational Heuristics:**

1. **Use full fine-tuning when domain shift is large and compute is available.** LoRA's rank ceiling is a hard architectural constraint, not a tuning parameter. A rank sweep will reveal saturation before parity with full FT — that saturation is the signal to switch regimes, not increase rank further.

2. **Use CURLoRA over standard LoRA in sequential task fine-tuning.** The 384× trainable parameter reduction translates directly to faster, cheaper training cycles and better prior-task preservation. The total memory footprint difference is smaller, but the gradient-tracked reduction is real.

3. **Never commit to a PEFT method without a rank sweep.** Rank 256 LoRA is 16× more expensive than rank 16, but still loses to full FT on hard CPT. If rank saturation appears before the full FT baseline, the method is wrong for the task — no further tuning will close the gap.

---

## Core Area 3: Runtime Feedback Loop (SIA-Based Self-Improvement)

**The problem:** A model trained on a static dataset degrades relative to the live distribution as the policy improves. The deeper problem is that two improvement levers — scaffold iteration (prompt engineering, phase logic, tool dispatch) and weight updates (LoRA fine-tuning) — have historically been applied in isolation. SIA (arXiv:2605.27276) demonstrates that combining both levers in a single closed loop outperforms either alone on every tested benchmark: +25.1% on LawBench, +12.4% GPU kernel speedup, +20.4% on scRNA-seq denoising.

**The architecture of the feedback loop:**

```
[RuntimeHost] → session artifacts (transcript.jsonl, prosody.jsonl, telemetry.jsonl)
      ↓
[Rubric Scorers] → per-session RubricScore (7 dimensions: rwrd, cont, afct, dlvr, nint, slnc, spch)
      ↓
[Feedback-Agent] → reads full trajectory + scores + prior improvement history
      │
      ├─→ Plateau NOT detected → HARNESS UPDATE
      │     Rewrite: system prompts, phase transitions, persona compiler, answer extraction
      │     Output: new prompt config + improvement.md
      │     Loop back to RuntimeHost
      │
      └─→ Plateau detected → WEIGHT UPDATE
            Select RL algorithm based on reward structure (see below)
            Dispatch: LoRA training job on Modal
            Output: LoRA adapter checkpoint
            Loop back to RuntimeHost with adapted model
```

**Algorithm selection (when weight update is triggered):**

| Reward Structure | Algorithm | When to use in Rehearse |
|---|---|---|
| Dense, clean scalar | PPO + GAE (value head) | Content scoring (rwrd, cont) — binary correct/incorrect per turn |
| Sparse, outcome-heavy | Entropic advantage weighting | Delivery/affect scoring — most sessions near-zero signal, few are exceptional |
| Coupled hyperparameters | GRPO | Persona parameters that interact (warmth × directness × pacing) |

**The sequencing rule (from SIA empirics):** Always start with harness iteration. On LawBench, harness iteration alone drove +36.5pp gain before weight updates were needed. Weight updates delivered +20.1pp *on top of* a converged scaffold. Switching to weight updates before the scaffold has converged wastes compute and trains on a noisy signal from a suboptimal execution policy.

**Plateau detection heuristic:** Switch to weight updates when improvement across the last K eval runs is within measurement noise on all rubric dimensions where the scaffold had been showing gains. K=3 is sufficient for Rehearse's eval cadence.

**3 Operational Heuristics:**

1. **The Feedback-Agent reads trajectories, not metrics.** Aggregate rubric scores tell you *that* something broke. The full session transcript + prosody log tells you *where*: which turn the persona broke character, which question caused the intake to stall, which response had prosodic mismatch despite correct words. Wire the Feedback-Agent to `transcript.jsonl` + `prosody.jsonl` as primary inputs, not just `results.jsonl`. This is what makes SIA's harness updates more targeted than prompt engineering from aggregate evals.

2. **Instrument session artifacts from day one — the same schema in prod and eval.** The improvement loop only starts when the Feedback-Agent can read a trajectory that looks identical whether it came from a live session or a sandbox rollout. Rehearse's `evals/runs/{run_id}/sessions/{id}/` and `sessions/{id}/` already share the same artifact schema. Do not diverge these. Every schema divergence between prod and eval artifacts is a dead week when the Feedback-Agent produces wrong improvements.

3. **Run internal eval on Rehearse's actual rubric before committing to an online training regime.** AlpacaEval-2-style benchmarks reward style and length, not coaching quality. Rehearse's 7-dimension rubric is the ground truth. Validate that each harness update moves the rubric in the expected direction before triggering weight updates — a rubric-validated scaffold is a stable foundation; an unvalidated one makes LoRA training noisier, not better.

---

## How the Three Areas Compound

The data flywheel is the binding mechanism:

- **Experiences collected at runtime** (Core 3) → feed replay buffers and reward model updates (Core 1)
- **Replay buffer composition** determines whether adapter or full fine-tuning is warranted for the next cycle (Core 2)
- **Fine-tuning regime selection** (Core 2) determines how much forgetting occurs, which sets the baseline the online feedback loop must overcome (Core 3)

Revenue growth attaches at Core 3 — the feedback loop is where user behavior directly informs model quality. Compute and architecture decisions (Cores 1 and 2) determine how efficiently that signal is absorbed.

---

## Instructions for Parallel Agent Execution

This section provides a task brief for a large-scale parallel coding and computer use agent tasked with implementing the above strategy.

### Agent Mission

Implement the three-core-area technical strategy as a working ML training system. The system must support sequential fine-tuning with forgetting mitigation (I-LoRA or CURLoRA), regime-aware training (full FT vs PEFT decision gate), and a runtime feedback loop that logs completions and drives online preference learning.

### Parallel Workstreams

Agents should be dispatched in parallel across three independent tracks. Each track corresponds to one core area and has a defined interface contract so tracks can be integrated without sequential dependencies.

**Track A — Continual Learning Architecture**
- Implement I-LoRA dual-memory EMA adapter wrapper for HuggingFace PEFT
- Implement CURLoRA adapter (CUR matrix decomposition, train U only)
- Implement replay buffer with per-task proportional sampling
- Expose: `adapter = build_adapter(model, method="ilora"|"curlora", rank=16)`
- Expose: `buffer = ReplayBuffer(tasks, sample_weight="proportional")`
- Metric gate: backward transfer (BWT) per task transition must be logged

**Track B — Compute & Training Efficiency (Regime Selection)**
- Implement the training regime decision gate (full FT vs CURLoRA vs LoRA)
- Implement rank sweep utility: `rank_sweep(model, task, ranks=[16,64,256])`
- Integrate activation checkpointing and mixed precision (bf16) for all regimes
- Expose: `regime = select_regime(task_type, compute_budget, forgetting_risk)`
- Metric gate: track HumanEval / GSM8K / task-specific eval per checkpoint

**Track C — Feedback-Agent (SIA Pattern)**
- Implement `FeedbackAgent` that reads: session transcript, prosody log, rubric scores, and prior improvement.md files from `evals/runs/`
- Implement plateau detection: K=3 consecutive runs below improvement threshold across all improving rubric dimensions → trigger weight update
- Harness update path: Feedback-Agent emits updated system prompts + phase config + improvement.md; no Modal dispatch
- Weight update path: Feedback-Agent selects RL algorithm (PPO/GRPO/entropic) based on reward structure of the stalled dimension, dispatches LoRA training job to Modal
- Expose: `agent = FeedbackAgent(trajectory_dir, rubric_scores, improvement_history)`
         `action = agent.select_action()  # returns HarnessUpdate | WeightUpdate`
- Metric gate: track rubric delta per harness iteration; confirm plateau detection fires correctly before first Modal dispatch; track LoRA training wall-clock vs rubric gain

### Integration Contract

After parallel tracks complete, a single integration agent wires them together:
1. Track B's `regime` selection determines whether Track A uses full FT or adapter
2. Track A's `buffer` feeds into Track C's online training loop as replay pairs
3. Track C's logged completions return to Track A's buffer for the next cycle

### Verification Checkpoints

Before declaring any track complete, the agent must:
- Run the track's metric gate and record the result in `docs/metrics/`
- Confirm no regressions in adjacent tracks (integration contract holds)
- Run a backward transfer measurement across at least 3 sequential tasks

### What to Avoid

- Do not use EWC as the primary forgetting mitigation — it did not survive adversarial verification at 7B+ scale in sequential settings
- Do not use offline-only DPO for the feedback loop — the static dataset risk is the primary quality ceiling
- Do not commit to a LoRA rank without a sweep — rank saturation before full FT parity is the signal to switch regimes, not tune further

---

*Sources: arXiv:2308.08747, arXiv:2402.18865, arXiv:2405.09673v2, arXiv:2408.14572, arXiv:2405.07863v3 (ICLR 2025). Research conducted 2026-05-31 via 6-angle fan-out, 31 sources, 25 adversarially verified claims (3-vote), 6 confirmed.*
