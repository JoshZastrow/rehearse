# CS336 → Rehearse: A Prioritized Learning Curriculum

**Goal: maximize utility.** What to study, in what order, to most directly improve your ability to build Rehearse — the continual-learning voice agent whose **eval harness is the ground-truth research artifact** and whose **runtime feedback loop (eval → Feedback-Agent → harness update | LoRA weight update on Moshi 7B) is where revenue attaches.**

This crawls all 14 CS336 lecture transcripts in `~/Desktop/youtube/*-segmented.md`, scores each against a Rehearse-specific rubric, and orders them by impact. Each lecture's *most useful* idea-blocks are named exactly so you can pull the verbatim passage:

```python
from yt2md.scripts.retrieval import read_table_of_contents, get_transcript_section
get_transcript_section("<exact section header below>")
```

Or invoke the distilled skills (`/llm-evaluation`, `/llm-mid-post-training`, `/llm-post-training-rlvr`, `/llm-parallelism`, `/llm-inference`, `/llm-data-curation`, `/llm-scaling-laws`).

---

## The Rubric

Each lecture scored 1–5 on five weighted dimensions. **Priority = Σ(score × weight)**, max 55.

| Dimension | Weight | 5 = | 1 = |
|---|---|---|---|
| **Stack Fit** | ×3 | Maps directly onto a Rehearse subsystem (`eval/`, `train/`, Feedback-Agent, serving) | No code touches it |
| **Thesis Impact** | ×3 | Advances the continual-learning + feedback-loop thesis where revenue attaches | Orthogonal to the thesis |
| **Gap Coverage** | ×2 | Fills knowledge *not already specified* in `CLAUDE.md`'s research report | Already fully covered |
| **Time-to-Apply** | ×2 | Actionable in days, low effort | Long-horizon / speculative |
| **Foundational** | ×1 | Prerequisite for other high-value modules | Leaf topic |

**Why these weights:** Stack Fit and Thesis Impact dominate because utility = "helps me build *this* system." Gap Coverage and Time-to-Apply break ties toward things you don't already know and can use now. Foundational is a light tiebreaker so prerequisites surface first within a tier.

---

## Scored Ranking

| # | Lecture | Stack | Thesis | Gap | Time | Found | **Priority** | Tier |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 | **L12 Evaluation** | 5 | 5 | 4 | 5 | 4 | **52** | 🔴 1 |
| 2 | **L15 Mid/Post-Training** | 5 | 5 | 4 | 4 | 4 | **50** | 🔴 1 |
| 3 | **L16 Post-Training — RLVR** | 5 | 5 | 4 | 3 | 3 | **47** | 🔴 1 |
| 4 | **L7+L8 Parallelism** | 5 | 3 | 3 | 4 | 3 | **41** | 🟠 2 |
| 5 | **L10 Inference** | 4 | 3 | 4 | 3 | 3 | **38** | 🟠 2 |
| 6 | **L13+L14 Data** | 3 | 4 | 3 | 3 | 2 | **35** | 🟡 3 |
| 7 | **L9+L11 Scaling Laws** | 3 | 2 | 3 | 2 | 3 | **28** | 🟡 3 |
| 8 | **Guest: Dan Fu (serving)** | 3 | 2 | 3 | 2 | 2 | **27** | 🟢 4 |
| 9 | **L17 Multimodality** | 2 | 2 | 2 | 2 | 2 | **22** | 🟢 4 |
| 10 | **L6 Kernels / Triton** | 2 | 1 | 3 | 1 | 2 | **19** | 🟢 4 |
| — | L5 GPUs/TPUs | — | — | — | — | — | *(ref)* | prereq → `gpu-compute-planning` |

---

## Curriculum (study in this order)

### 🔴 TIER 1 — The feedback loop and its ground truth (learn first)

These three *are* Rehearse's research thesis. The eval harness measures, post-training shapes, RLVR/RLHF updates weights. Master them before anything else.

#### Module 1 — L12 Evaluation  *(your eval harness is the product)*
**Why #1:** Rehearse's core artifact is a 7-dimension rubric harness with audio-native LLM judges. Everything in the feedback loop is only as trustworthy as this measurement. This lecture is the textbook for it.

Read first, in order:
1. `Evaluation turns abstract constructs into concrete prompt-and-score procedures that shape the model` — the framing for why your rubric *defines* the agent.
2. `Checklists and rubrics improve the reliability of automatic judging` — directly upgrades your `eval/scorers/` + rubric design (`cont`, `dlvr`, `afct`).
3. `Who are the judges? Biases like sycophancy and length skew preferences` — your Gemini/Claude judges have these biases; affects `rwrd`/`cont` calibration.
4. `AlpacaEval's win-rate-against-baseline and reducing judge bias with ensembles` — ensemble judges to de-bias; `CLAUDE.md` Core-3 Heuristic 3 already warns AlpacaEval-style style/length bias — this is the mechanism.
5. `Agentic benchmarks can be gamed, so always read the actual outputs` — mirrors `CLAUDE.md` "Feedback-Agent reads trajectories, not metrics."
6. `Contamination and decontamination: train-test overlap, cutoff dates, and private codebases` — keep eval fixtures out of LoRA training data.
7. `Match the benchmark to your goal; the nanoGPT speedrun evaluates an algorithm` — pick the right harness per improvement lever.

**Apply:** Audit each of the 7 rubric dims for judge bias (length/sycophancy); add a checklist/rubric spec per dim in `eval/scorers/`; add an ensemble or rubric-anchored judge for the noisiest dims (`afct`, `dlvr`). Add a decontamination gate so eval sessions never leak into replay/training.

#### Module 2 — L15 Mid/Post-Training  *(SFT, reward models, DPO, the traps)*
**Why #2:** Your weight-update path SFT/LoRA-tunes Moshi 7B and is RLHF-shaped. This is the practical playbook — *and* its warnings are exactly your failure modes.

Read first:
1. `RLHF mechanics: sample outputs, train a reward model, and rate for helpful, truthful, and harmless` — your "reward model" is the rubric scorer; same shape.
2. `Overoptimization and mode collapse are the big RLHF pitfalls, setting up the next lecture` — **the** risk for a self-improving loop; watch entropy/diversity on the persona.
3. `DPO, the fun bit...` → `The DPO derivation removes the reward model...` → `DPO is good enough in practice, with many later variants, though the results are fragile` — DPO vs PPO/GRPO for your loop; "fragile/setup-dependent" is critical for plateau decisions.
4. `Rejection sampling and reward-model-selected SFT as simpler alternatives` — cheapest first weight-update before RL; great for early Rehearse.
5. `Teaching the model unknown knowledge forces it to hallucinate, so RL and calibration matter` — don't SFT Moshi on coaching "facts" it can't know; keep targets within capability.
6. `Mid-training blurs pre-training and post-training by mixing high-quality and chat data near the end` + `Model-based annotation ... can beat costly human annotation` — your synthetic/scored sessions as training data.

**Apply:** Make rejection-sampling SFT the *first* weight-update lever in the Feedback-Agent (before PPO/GRPO). Add an entropy/mode-collapse monitor on persona outputs to the eval run. Constrain LoRA SFT targets to behavior/format, not knowledge.

#### Module 3 — L16 Post-Training — RLVR  *(the RL lever, done right)*
**Why #3:** `CLAUDE.md` Core-3 already prescribes PPO/GRPO/entropic by reward structure — this lecture is the *mechanics* behind that table, plus the systems reality of a rollout↔train loop you'll have to build.

Read first:
1. `GRPO removes the value-function network for a much simpler algorithm` + `GRPO normalizes rewards into a per-group z-score and adds a KL term to the reference` — the algorithm you'll most likely implement; no value head to babysit.
2. `Verifiable rewards let you pour in compute without a hackable reward model` — your rubric is a *learned* reward → expect gaming; design verifiable sub-signals where possible (`nint`, `slnc`, `spch` are measurable).
3. `Reward hacking: agents exploit things like git history unless you gate them` — the persona/loop will exploit any rubric loophole; gate it.
4. `On-policy RL is mathematically nice, but inference infrastructure and weight transfer make the systems hard` — the rollout↔train↔weight-transfer loop is the hard part; matches your Modal dispatch design.
5. `Difficulty matters: problems that are too easy or too hard give no learning signal` + `RL-flavored dataset curation drops problems that are far too hard` — curate session scenarios at the edge of the agent's ability.
6. `PPO implementations hide many fragile details, especially the value model and advantage estimation` — why GRPO/rejection-sampling beat PPO for a small team.

**Apply:** Prototype GRPO (group rollouts per scenario → z-scored rubric reward → KL to reference) as the plateau-triggered weight update. Split rubric dims into *verifiable* (deterministic: `nint`, `slnc`, `spch`) vs *learned* (judge: `afct`, `cont`) and weight verifiable ones to resist reward hacking. Add a reward-hacking guard to the loop.

---

### 🟠 TIER 2 — The training & serving substrate

#### Module 4 — L7+L8 Parallelism  *(how Moshi 7B fits and trains)*
**Why:** `train/` is FSDP + LoRA on Moshi 7B with bf16. This is the *why* behind that choice and how to scale it.

Read first:
1. (L8) `ZeRO stage three (FSDP) shards parameters too, gathering and freeing them on demand` + `In practice FSDP lets an A100 jump from not fitting a 7B model to fitting roughly 50B-parameter models` — exactly your regime.
2. (L8) `Memory accounting reveals the situation is terrible: roughly five copies of weights at sixteen bytes per parameter` + `Adam optimizer state ... is the dominant memory cost` — why LoRA + bf16; budget your Modal GPU.
3. (L8) `The actual prescription is simple: cut the model up by any means until it fits` + `Use tensor or expert parallel on fast interconnects, then pipeline or FSDP to fit, then data parallel the rest` — the decision order if you outgrow one GPU.
4. (L7) `The key DDP step all-reduces and averages gradients so all ranks match before updating` — the one collective you must get right.

**Apply:** Document the memory budget for Moshi-7B LoRA (5×-copies math) in `train/`; confirm bf16 + activation checkpointing; keep FSDP as the only parallelism until a single GPU genuinely can't hold the LoRA job.

#### Module 5 — L10 Inference  *(real-time voice = latency is a capability)*
**Why:** Naturalness across a long call is a stated capability goal; the interactive Moshi/Mimi backend is latency-bound. This is the lever set.

Read first:
1. `Latency versus throughput: time-to-first-token, per-token speed, and their tension` — TTFT is what the caller feels; frame serving around it.
2. `Prefill is compute-bound, but generation's intensity does not scale with batch, so it bottlenecks` + `The KV cache reuses computed key/value pairs across generation steps` — the core mental model for streaming generation.
3. `Reducing the KV cache with grouped-query attention without losing accuracy` + `Multi-head latent attention compresses keys and values` — long multi-session context cheaply.
4. `Speculative decoding: a cheap draft model proposes tokens the target verifies losslessly` — cut per-token latency for snappier turns.
5. `Continuous batching ...` + `Paged attention manages the KV cache like OS virtual memory, enabling prefix sharing` — when you self-host serving for many concurrent calls.

**Apply:** Instrument TTFT + per-token latency in the interactive backend; evaluate speculative decoding for turn latency; plan KV-cache strategy for long/multi-session context before self-hosting.

---

### 🟡 TIER 3 — The data flywheel & efficiency tuning

#### Module 6 — L13 + L14 Data  *(replay, mixing, synthetic sessions)*
**Why:** The data flywheel binds the three cores. Your "data" is scored session trajectories, but the *mechanics* (mixing weights = epochs, dedup, synthetic generation) transfer directly to replay-buffer composition (Core 1).

Read first:
1. (L14) `Data mixing: weighting sources and the number of epochs each source implies` + `Optimizing mixture weights by regressing weights to loss while avoiding overfitting` — directly informs replay-buffer per-task weighting (`CLAUDE.md` Core-1 Heuristic 2).
2. (L14) `Synthetic data: defining environments and using stronger models as teachers` + `Synthetic code tasks generated from repositories, and SWE-style pipelines` — generate synthetic caller scenarios from Opus to grow the buffer.
3. (L14) `Fuzzy deduplication via Jaccard similarity and MinHash` + `Locality-sensitive hashing with bands and rows gives an S-shaped match curve` — dedup near-identical sessions so the buffer stays diverse.
4. (L13) `A well-filtered quality subset beats larger raw datasets` — a small, high-rubric set of trajectories beats a big noisy one.

**Apply:** Set replay-buffer sampling weights proportional to task volume (not absolute tokens); add MinHash dedup over session transcripts before they enter the buffer; build a synthetic-caller generator for under-represented scenarios.

#### Module 7 — L9 + L11 Scaling Laws  *(regime selection & HP transfer)*
**Why:** `CLAUDE.md` Core-2 is a full-FT-vs-LoRA decision gate. Scaling laws are the *evidence-fitting* discipline behind "rank sweep before committing" and hyperparameter transfer.

Read first:
1. (L9) `Upstream loss versus downstream task performance is often weakly correlated` — don't trust LoRA val-loss as a proxy for rubric gains; gate on the rubric.
2. (L9) `Compute-optimal allocation trades parameters against data` + `Overtraining is the rational choice once you account for inference` — you serve one model heavily → overtrain/distill toward smaller serving cost.
3. (L11) `MuP ... enforce the two invariants ... so a tuned small model transfers to a large one` + `The WSD trapezoid schedule lets you reuse and extend runs by decaying` — cheap HP transfer + resumable training for the loop's repeated small fine-tunes.

**Apply:** Add a LoRA rank sweep utility gated on *rubric* deltas (not loss); use a WSD schedule so each feedback-loop fine-tune can extend the last instead of restarting.

---

### 🟢 TIER 4 — As-needed references

- **Guest: Dan Fu (serving systems)** — read `Prefill and decode are two very different machine-learning computations` and `Simple request routing by length yields 40 percent faster serving` when you self-host Moshi serving. Loop-transformer content is research-tangential.
- **L17 Multimodality** — the **encoder-adapter-LLM** pattern (`LLaVA's adapter aligns image vectors into the language model`) is the structural analogy for wiring Mimi/audio into an LLM; vision specifics are not directly applicable.
- **L6 Kernels / Triton** — only if you chase a custom-kernel speedup (SIA's +12.4% GPU-kernel result is the temptation). Start at `Fuse to kill HBM round-trips` / `torch.compile`. Most teams should not write kernels yet.
- **L5 GPUs/TPUs** — hardware mental model; already covered by the `gpu-compute-planning` skill.

---

## If you only learn 5 things this week

1. **Rubric & judge design + de-biasing** — L12 (M1 reads 1–5). Your measurement is the product.
2. **Mode collapse / overoptimization guardrails** — L15 (`Overoptimization and mode collapse...`). The #1 self-improving-loop failure.
3. **GRPO mechanics + reward-hacking gates** — L16 (M3 reads 1–3). Your weight-update lever.
4. **FSDP + the 5-copies memory budget** — L8 (M4 reads 1–2). Makes Moshi-7B LoRA actually fit.
5. **Replay-buffer weighting = epochs + dedup** — L14 (M6 reads 1, 3). Keeps the flywheel honest.

## How this maps to Rehearse's three cores
- **Core 1 (Continual Learning):** Module 6 (replay/dedup/mixing), Module 4 (FSDP memory).
- **Core 2 (Compute/Regime):** Module 7 (rank sweep, HP transfer), Module 4.
- **Core 3 (Feedback Loop — where revenue attaches):** Modules 1, 2, 3 (eval → reward → RL). **Spend most of your time here.**

---
*Generated by crawling `~/Desktop/youtube/*-segmented.md` (14 CS336 lectures, 521 indexed idea-blocks) against the Rehearse vision in `CLAUDE.md` + `README.md`. Section titles are retrievable verbatim via `get_transcript_section(...)`; summaries live in `yt2md/docs/transcripts/`.*
