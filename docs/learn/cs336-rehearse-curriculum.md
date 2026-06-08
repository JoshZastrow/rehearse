# CS336 → Rehearse: A Focused Learning Curriculum

> **What this is.** A prioritized study plan that turns Stanford **CS336 — Language Modeling from Scratch (Spring 2026)** into the specific things *you* need to learn to build Rehearse. Built by crawling all 14 indexed lecture transcripts (lectures 5–17 + the Dan Fu guest lecture) and scoring every topic against a rubric tuned to Rehearse's vision (`README.md`, `CLAUDE.md`, `SPEC.md`, `TODO.md`).
>
> **Calibrated to your choices (2026-06-08):** front-load **post-training & the weight-update loop**; primary target = **LoRA fine-tune Moshi 7B**; you're **solid on training, lighter on systems** (so systems content is targeted, not exhaustive); shape = **focused top priorities**.
>
> **Source library.** Transcripts live in `~/Desktop/youtube/…-segmented.md` (searchable, with `table-of-contents.md`); each lecture is also distilled into an action skill (named per module below). Pull a verbatim section with:
> ```python
> from yt2md.scripts.retrieval import read_table_of_contents, get_transcript_section
> get_transcript_section("<exact idea-block header>")
> ```

---

## TL;DR — the priority path

Learn in this order. Each module names the **one action** that most moves Rehearse.

| # | Module | Lecture(s) | Skill | The one action |
|---|--------|-----------|-------|----------------|
| **1** | **SFT → DPO & the preference loop** | L15 | `llm-mid-post-training` | Close the weight-update loop with **rejection-sampling-SFT, then DPO** on harness-ranked trajectories — not PPO. |
| **2** | **Reward design & RLVR / GRPO** | L16 | `llm-post-training-rlvr` | Treat the **reward as the product**; know the **one-page GRPO**; budget for **reward hacking**. |
| **3** | **Trustworthy reward: eval & audio judges** | L12 | `llm-evaluation` | Make judges **bias-resistant & calibrated** (pairwise+checklists+ensemble) *before* any weight update. |
| **4** | **Moshi inside: multimodal arch & staged fine-tuning** | L17 | `llm-alignment-multimodality` | Stage the fine-tune **LLaVA-style** (align adapter → LoRA the LM → extend context); expect mixed-token instability. |
| **5** | **Fine-tuning data & continual learning** | L13–L14 | `llm-data-curation` | Run the **data-mixing math** on your replay buffer; **dedup** synthetic scenarios; generate **multiple variants per scenario**. |
| **6** | **Serving Moshi in real time** | L10 + Dan Fu | `llm-inference` | Serve Moshi through **vLLM/SGLang** (paged attention, continuous batching, prefix sharing); the live call is **decode/bandwidth-bound**. |

**If you only do three things this month:** (1) wire a **rejection-sampling-SFT → DPO** loop on your top-`rwrd` trajectories [M1]; (2) **harden the judges** — pairwise comparison + per-phase checklists + length-debias — and run a **degenerate-baseline attack** on your own rubric [M3]; (3) **stage the Moshi LoRA fine-tune** adapter-first with a strong KL/anti-mode-collapse guard [M4].

**Supporting (targeted, brief):** FSDP + activation-memory OOM levers (L7–L8), WSD schedule + LR/batch priors (L9, L11), the GPU performance mental model (L5–L6). See [§Supporting](#supporting--targeted-skim-as-needed).

---

## The rubric

Each lecture topic was scored **1–5** on five axes, then rolled into a priority tier. The axes:

| Axis | Question |
|---|---|
| **R1 — Direct applicability** | Does it map onto Rehearse's *actual* stack today — the Moshi LoRA/FSDP train stack, the eval harness, the audio judges, the self-improvement loop? |
| **R2 — Impact** | If learned & applied, how much does it move the weight-update loop, Moshi fine-tune quality, real-time latency, or reward trustworthiness? |
| **R3 — Foundational** | Is it a prerequisite for understanding other high-value topics? |
| **R4 — Actionability** | Can you apply it *now*, concretely (vs. purely conceptual)? |
| **R5 — Quick win** | Low effort-to-learn relative to payoff? |

**Weighting (from your calibration).** Priority ≈ `0.30·R1 + 0.30·R2 + 0.15·R4 + 0.15·R3 + 0.10·R5`, with a **front-load boost** for post-training/reward topics and an **applicability boost** for anything that bears on *fine-tuning or serving Moshi specifically*. Because you're "lighter on systems," a systems topic must earn its place on **R3/R4** (it unblocks an action you'll actually take) — otherwise it's demoted to Supporting or Skip.

**Tiers:** **CORE** = deep module below · **SUPPORTING** = targeted/skim · **SKIP-FOR-NOW** = defer until a named trigger.

---

## Priority map (all 14 lectures)

| Lecture | Topic | Skill | Tier | Why (Rehearse lens) |
|---|---|---|---|---|
| **L16** | Post-Training: RLVR | `llm-post-training-rlvr` | **CORE (M2)** | The weight-update lever *is* RL; reward design + reward hacking govern the whole loop. |
| **L15** | Mid/Post-Training (SFT, RLHF, DPO) | `llm-mid-post-training` | **CORE (M1)** | You're post-training a conversational assistant; DPO on preference pairs is the most actionable path. |
| **L12** | Evaluation | `llm-evaluation` | **CORE (M3)** | The eval harness *is* the reward model; judge reliability is the binding constraint on RL. |
| **L17** | Alignment — Multimodality | `llm-alignment-multimodality` | **CORE (M4)** | Moshi *is* an encoder-adapter-LLM over discrete (Mimi) audio tokens — this is its blueprint. |
| **L14** | Data (transform, filter, dedup, mix, synthetic) | `llm-data-curation` | **CORE (M5)** | Synthetic callers, replay-buffer mixing, quality filter, dedup — your four live data workstreams. |
| **L10** | Inference | `llm-inference` | **CORE (M6)** | Real-time voice ≤800 ms + cheap RL rollouts both ride on this. |
| **Guest** | Dan Fu — serving | `llm-inference` | **CORE (M6)** | Practitioner serving: prefix caching, length-routing, megakernels, the PARSE loop-transformer aside. |
| **L8** | Parallelism II (FSDP, memory) | `llm-parallelism` | **SUPPORTING** | Memory accounting + activation recompute = your single biggest OOM lever on the A10G. |
| **L11** | Scaling Laws II (WSD, optimizers, MuP) | `llm-scaling-laws` | **SUPPORTING** | WSD schedule + "small-scale wins bite you" = how to set LoRA HPs and judge tweaks cheaply. |
| **L5** | GPUs/TPUs | `gpu-compute-planning` | **SUPPORTING** | Roofline + memory hierarchy = the mental model under all latency/memory reasoning. |
| **L6** | Kernels, Triton, XLA | `llm-gpu-kernels` | **SUPPORTING** | `torch.compile`/profiling/flash-attn = quick latency wins; hand-writing kernels = skip. |
| **L7** | Parallelism I (collectives, DDP) | `llm-parallelism` | **SKIP-FOR-NOW** | Single-GPU LoRA needs only the all-reduce identity; revisit if you go multi-GPU. |
| **L9** | Scaling Laws I (Chinchilla) | `llm-scaling-laws` | **SKIP-FOR-NOW** | You're not sizing a from-scratch pretrain; keep only "upstream loss ≠ downstream." |
| **L13** | Data sources / licensing | `llm-data-curation` | **SKIP-FOR-NOW** | You own your data; keep only the model-based-filtering recipe (folded into M5). |

---

# The 6 core modules

Each module: **why it matters → what to learn → what to skip → apply to Rehearse → source pointers → time.**

---

## Module 1 — SFT → DPO & the preference loop  *(front-load)*
**Lecture 15 · skill `llm-mid-post-training` · ~5–6 h**

### Why it matters
This is the most *actionable* weight-update path for Rehearse and the gentlest on-ramp to the loop. Your judges already produce ranked trajectories; DPO needs exactly that (ranked pairs) and runs comfortably as LoRA + bf16 on an A10G — **no reward-model server, no rollout loop**. It also carries two warnings that are uniquely dangerous for a *prosody-first* agent.

### What to learn (MUST)
- **The governing principle: the leverage is the *data*, not the algorithm.** SFT is mechanically just gradient descent on different data; frontier labs guard *data recipes*, not PPO. Pour effort into curating scored trajectories and judge prompts, not into agonizing over PPO-vs-GRPO.
- **SFT teaches *format + knowledge* simultaneously → teaching *unknown* knowledge forces hallucination.** If you SFT Moshi on persona facts/backstory it can't already produce, it learns to confabulate. Keep SFT targets to behaviors/prosody the base model can already sample; push novel calibration to the preference stage.
- **SFT at its best *extracts* latent ability with little data** — hundreds of top examples, not thousands. Start small, curate the best.
- **DPO derivation & intuition.** The KL-regularized reward objective has a closed-form optimal policy; invert it to remove the reward model. Loss = raise log-prob of the winner, lower the loser, with **step size ∝ how wrong the model currently is**. Mechanically "SFT on the good + negative-SFT on the bad."
- **Rejection-sampling-SFT is the even-simpler first step**: have the harness pick the top-scored counterparty turns and plain-SFT on them (no negative gradients). The Llama recipe is an **outer loop**: SFT → DPO → generate → rejection-sample → repeat — a near-perfect template for your self-improvement loop.
- **Model-based annotation is validated** (Zephyr/UltraFeedback beat costly human labels). Your Gemini/Claude audio judges *are* this — the correct, frontier-matching choice for a prototype.
- **Two failure modes to design against:**
  - **Length/style bias** — LLM judges reward longer/more "polished" output. For a voice agent this means rewarding verbose TTS over authentic prosody, corrupting `dlvr`/`spch`/`nint`. **Decorrelate style from substance in the rubric.**
  - **Overoptimization → mode collapse + miscalibration** — aggressive RLHF/LoRA flattens the counterparty to one emotional register, destroying Practice-phase realism and the variance `afct`/`dlvr`/`spch` measure. **Keep a strong KL-to-reference, monitor prosodic/output entropy, prefer light LoRA.**

### What to skip
Mid-training mechanism (decay-phase data mixing — you don't control Moshi's pretrain), the Flan→Alpaca history, deep PPO/TRPO internals (→ M2), tool-call SFT, safety/refusal tuning beyond a tiny "stay-in-character" set. Keep only their *methodology* (ablation-driven sweeps; "policy-gradient = weighted SFT").

### Apply to Rehearse
- **Build the outer loop:** `harness scores trajectories → mine PreferencePair rows → DPO LoRA on Moshi → regenerate → repeat`. You already have `PreferencePair`/`TrainingExample` types (`SPEC §7`) and `dev-lab/recipes/preference`.
- **First experiment, this week:** rejection-sampling-SFT on the top-`rwrd` coach turns from `eval/runs/`. Cheaper and more stable than DPO; establishes the plumbing.
- **Guardrail metric:** log prosodic/output entropy per checkpoint; if it drops, you're collapsing — back off LoRA aggressiveness or raise the KL term.
- **Audit:** before the next round, regress judge score against turn duration/token count — if correlated, your reward rewards verbosity, not coaching.

### Source pointers
Skill `llm-mid-post-training`. Transcript: `…lecture-15-midpost-tra-segmented.md`. Return-to headers: *"The DPO derivation removes the reward model using the implied closed-form optimal policy"*; *"Length effects: you can push response length and still do well on benchmarks"*; *"Overoptimization and mode collapse are the big RLHF pitfalls."*

---

## Module 2 — Reward design & RLVR / GRPO  *(front-load)*
**Lecture 16 · skill `llm-post-training-rlvr` · ~6–8 h**

### Why it matters
Your strategy doc (`CLAUDE.md` Core 3) explicitly selects among **PPO / GRPO / entropic-advantage** for the weight-update lever. This lecture is where you learn those cold — and, more importantly, learn that **"RL is all about the reward"**: the entire loop is gated by how hackable your reward (the eval harness) is. *Verifiable rewards are what let you pour compute in safely* — which is exactly why M3 (judge robustness) is the precondition for M2.

### What to learn (MUST)
- **REINFORCE = weighted SFT with signed weights.** The load-bearing primitive under PPO, GRPO, and entropic weighting. Demystifies what a weight-update job actually does to the Moshi policy and why on-policy sampling cost is unavoidable.
- **GRPO is your default.** It deletes PPO's worst part (the value network): advantage = **per-group z-score** (sample G rollouts per prompt, subtract group mean, divide by group std) + a **KL-to-reference** term. Online, the clip ratio is 1 so it collapses to `min(advantage) − KL`. One page to implement — mind the `+1e-4` on std (avoids NaN when all rewards equal) and the `stop_grad` placement. **RFT** (train only on the model's own correct/high-reward answers) is the cheap baseline to beat.
- **GRPO isn't a clean policy gradient.** The std-division and length-normalization are "correction factors" with real behavioral consequences — any normalization you bolt onto `rwrd`/`dlvr`/`spch` silently reshapes what Moshi optimizes.
- **Length normalization → rambling.** Dividing loss by output length lets a model that knows it's wrong ramble infinitely (the −1 penalty ÷ ∞ → 0). Maps directly to **`spch`/`dlvr`/`slnc`**: without length control, your counterparty learns to "blab" when losing the argument, inflating call time. Kimi's length-*compression* recipe (don't over-shorten the failing case) is the lever for trading turn length against quality.
- **Difficulty filtering = train at the edge of capability.** Best-of-K filtering (drop scenarios the model always wins or always loses) keeps signal high. This is the same "is there still signal here?" question as your **K=3 plateau detector** — filter scored trajectories to scenarios where Moshi is *on the edge*.
- **Reward hacking is the #1 risk.** Agents exploit git history; a Moshi counterparty optimized against audio judges will find acoustic/textual exploits of `afct`/`slnc`/`rwrd` (the prosody analog). *"RLVR is only as robust as your reward."* Adversarially probe + gate exploits before each weight update.
- **On-policy RL systems pain.** Weight transfer trainer→inference server every iteration; the long-rollout straggler (one 6-min call blocks the batch) maps onto variable-length *voice sessions*. On-policy is mathematically nice but tempts you to reuse rollouts → off-policy → instability.
- **Does RL beat just training on correct answers?** Expert iteration (SFT on wins) is the stable first move and may suffice — but to squeeze the last gains you eventually need real RL. Justifies the weight-update lever existing at all.

### What to skip
Qwen3 thinking-mode fusion + expert distillation, Kimi's full DPO-style derivation, deep PPO "37 details" (know *that* it's finicky and value-net-heavy → reserve PPO for dense scalar `cont`). DPO-vs-PPO debates beyond "DPO is the wrong tool for non-pairwise rewards."

### Apply to Rehearse
- **Map the algorithm table to reward structure** (your `CLAUDE.md`): dense scalar (`cont`) → PPO+GAE *only if* justified; sparse/outcome (`afct`/`dlvr`) → entropic advantage; coupled persona knobs (warmth×directness×pacing) → **GRPO** over G sampled counterparty turns of the same intake/practice prompt.
- **Build the one-page GRPO loop** in `train/` as the concrete weight-update job — but only after M3 hardens the reward.
- **Add a length/latency reward** on Moshi turns for `spch`/`dlvr`, and *don't* over-compress so `slnc` (deliberate silence) survives.
- **Split the reward by verifiability** (the RLVR thesis applied to your rubric): lean the `rwrd` composite on the **deterministic/verifiable** dims — `nint`, `slnc`, `spch`, measurable straight from audio via the `librosa` scorers in your `TODO.md` — and treat the **learned/judge** dims (`afct`, `cont`) as softer, hack-prone signals. Verifiable sub-rewards are what let you "pour in compute" safely.
- **Reward-hacking drill:** keep a held-out red-team set; before each weight update, check the policy hasn't found an acoustic exploit of the judges.

### Source pointers
Skill `llm-post-training-rlvr`. Transcript: `…lecture-16-post-traini-segmented.md`. Return-to: *"You will implement a one-page GRPO… with a correct-answers baseline"*; *"Verifiable answer-checking is harder than it looks"* → **"RLVR is only as robust as your reward"**; *"On-policy RL is mathematically nice, but inference infrastructure and weight transfer make the systems hard."*

---

## Module 3 — Trustworthy reward: evaluation & audio judges
**Lecture 12 · skill `llm-evaluation` · ~5–6 h**

### Why it matters
Your README calls the eval harness **the core research artifact**, and `TODO.md` is almost entirely judge-reliability work (`affect_perception` swinging 0.0–0.65, `speech_rate` stuck at 0, the ρ≥0.7 calibration gate). In RL terms, **the harness is the reward model** — Modules 1–2 are only as good as this. Do M3 *before* you trust any weight update.

### What to learn (MUST)
- **Judge biases are real and inherited by LLM judges:** sycophancy, **length/verbosity skew**, **position bias**. Your domain is *adversarial* (the coach must sometimes be uncomfortable), so a sycophantic judge will mis-score `afct`/`dlvr`.
- **Pairwise comparison + ELO/Bradley-Terry beats absolute 1–5.** "A held space better than B" is far higher-signal than `slnc=4`. The connected-comparison-graph property lets you rank many model/prompt variants cheaply — exactly what the self-improvement loop needs.
- **Checklists/rubrics make automatic judging reliable (WildBench).** Decompose each of your 7 dims into explicit binary checklist items, and make them **per-phase** (silence/affect weigh heavily right after disclosure in Practice, less in Intake). *This is the single highest-leverage reliability win.*
- **Length-debias by regression** (the AlpacaEval fix): regress turn-duration/token-count out of judge scores so Moshi can't game `cont`/`rwrd` by talking more — critical because in voice, more talk usually *hurts* (`nint`↑, `slnc`↓).
- **Ensemble judges** (Gemini + Claude + a self-hosted judge) and treat agreement as your trustworthy label; report **inter-judge agreement** as a variance diagnostic (knowing it won't catch *shared* bias).
- **Validate any new scorer by correlation with human gold** — re-validated *per checkpoint* (agreement on a weak Moshi may break once it improves). This operationalizes your SPEC ρ≥0.7 / TODO ρ≥0.6 gates.
- **Always read the actual outputs.** Run a **degenerate baseline** (silent coach; canned-empathy coach) and confirm it scores *low* — if "polite silence" games `slnc`/`afct`, your rubric is exploitable. Build a trace-inspection step over judge rationales.
- **Decontamination + private fixtures.** Exclude any audio used to fine-tune Moshi from the eval set; keep a **private fixture slice never sent to third-party judges** (score it only with a self-hosted judge — ties to your self-host roadmap). Source scenarios from **expert-authored real hard conversations** (ecological validity).

### What to skip
MMLU/GPQA/HLE benchmark mechanics, perplexity leaderboards, ARC-AGI, jailbreak/GCG internals. Keep only the *meta-method* (multi-round expert validation, contamination probes), not the benchmarks themselves.

### Apply to Rehearse
- **Convert the 7-dim rubric into per-phase binary checklists** in `eval/judge.py`; expect a step-change in inter-judge agreement (directly attacks the `affect_perception` variance bug).
- **Switch the loop's scoring to pairwise + ELO** across model/prompt variants; fit Bradley-Terry per dimension.
- **Add the length/duration regression debias**; audit current scores for length correlation (a near-quick-win).
- **Run the degenerate-baseline attack** on your own harness this week; tighten any dimension a trivial agent can game.
- **Carve out a private, decontaminated fixture set** scored only by a self-hosted judge.

### Source pointers
Skill `llm-evaluation`. Transcript: `…lecture-12-evaluation-segmented.md`. Return-to: *"Checklists and rubrics improve the reliability of automatic judging"* (WildBench); *"Agentic benchmarks can be gamed, so always read the actual outputs"* (TAU-bench "empty response scores ~38%").

---

## Module 4 — Moshi inside: multimodal architecture & staged fine-tuning
**Lecture 17 · skill `llm-alignment-multimodality` · ~4–5 h**

### Why it matters
You chose **Moshi 7B LoRA** as the target. This lecture is image-domain on its surface but is, by analogy, **Moshi's literal blueprint**: Moshi fuses a speech modality into an LLM via an adapter and *generates* speech as discrete **Mimi** codec tokens. Understanding this is how you decide what to freeze, how to stage the fine-tune, and which instabilities to expect.

### What to learn (MUST — read every "image" as "audio")
- **Token abstraction for non-text modalities.** Every modality becomes discrete or continuous tokens before the transformer sees it. **Mimi is the BPE-equivalent for audio** — waveform → discrete codebook tokens; Moshi interleaves them with text in one stream.
- **The encoder → adapter → LLM recipe (LLaVA).** Frozen encoder + a projection adapter into the LLM's embedding space + concatenation with text tokens. *This is exactly Moshi's structure with audio swapped for image.* Know it cold.
- **Staged freeze/unfreeze training (LLaVA/Qwen-VL).** Stage 1: freeze encoder + LM, train **only the adapter** to align audio tokens into the LM space (cheap, stabilizes). Stage 2: LoRA-tune the LM on session audio. Stage 3+: progressively **extend context** (8K→…→256K for long sessions). *This is your fine-tuning playbook.*
- **Discrete-token generation via a learned codebook (VQ-VAE / Chameleon).** Encoder → quantize to nearest codebook entry → decoder reconstructs. **Mimi is exactly this for audio**, which is why Moshi can *speak* by autoregressive next-token prediction. Moshi is in the *generate-a-modality* camp, not the input-only VLM camp.
- **Mixed-entropy instability — pre-plan for it.** Mixing **low-entropy text + high-entropy audio tokens** in one stream is precisely the regime that caused norm blow-up and loss spikes in unified-token models. Expect it during fine-tuning; mitigations are **QK-norm, Z-loss, and per-modality loss weighting**.
- **Token-budget balancing (AnyRes + √-normalized loss).** Long audio = the "video" modality here: chunk into fixed windows, balance the token budget so audio doesn't drown text, and use a **√-normalized per-example loss** so long sessions don't dominate.

### What to skip
CLIP/SigLIP contrastive-loss math, LAION-5B, resize/center-crop, ViT-patch internals, M-RoPE height/width geometry. Capture only their one-line ideas (e.g., "contrastively align two modalities into a shared space"); the specifics are image-only.

### Apply to Rehearse
- **Stage your Moshi LoRA runs LLaVA-style**: adapter/alignment first, then LM LoRA, then long-context extension — wire this into `finetune/wrapped_model.py`'s regime logic.
- **Pre-arm the trainer against mixed-token instability** (QK-norm/Z-loss/loss-weighting) before you hit NaNs on the audio+text stream.
- **Apply the √-normalized loss / token-budget balance** to your ~100 s audio sequences so long sessions don't dominate the gradient (also eases the activation-memory pressure from M-Supporting).
- Read this **alongside the Moshi/Mimi papers** — the lecture gives you the vocabulary to read them fast.

### Source pointers
Skill `llm-alignment-multimodality`. Transcript: `…lecture-17-alignment---multimodality-segmented.md`. Return-to: *"LLaVA's adapter aligns image vectors into the language model"* (the freeze-then-unfreeze staging); *"Generating images with discrete image tokens via a learned codebook"* + *"Unified any-modality token models are elegant but unstable"* (VQ-VAE = Mimi; the entropy-mismatch instability + QK-norm/Z-loss fix).

---

## Module 5 — Fine-tuning data & continual learning
**Lectures 13–14 · skill `llm-data-curation` · ~4–5 h** *(spend ~80% on L14)*

### Why it matters
This maps almost 1:1 onto Rehearse's data workstreams: **synthetic caller drivers**, **replay-buffer composition** (your continual-learning pillar), **quality filtering** of sessions/scenarios, and **dedup**. The data decisions here directly set how much catastrophic forgetting you incur (`CLAUDE.md` Core 1).

### What to learn (MUST)
- **The filtering skeleton:** small *target* (high-quality) set + large *raw* pool → keep the subset most similar to target, fast. Train a **cheap classifier** (fastText / KenLM-perplexity): positives = your best sessions, negatives = noisy/off-topic ones; threshold into the manifest. The **Phi-1 trick** (expensive LLM labels a subset → distill a cheap classifier) grades scenario quality at scale without running an LLM over everything.
- **A well-filtered subset beats a larger raw set**, and **quality-vs-quantity depends on your token budget** — your session data is *scarce*, so over-epoching overfits; set quality threshold **and** epoch count jointly.
- **Data mixing = the replay-buffer problem, stated formally.** Per-task proportional sampling *is* a data mixture. The killer trap: each source is finite, so a naive distribution silently implies an **epoch count** — over-weighting a small recent task = dozens of epochs on it (overfit) while under-sampling others (forgetting). **Compute implied epochs-per-task before committing buffer weights; mix at the batch level.**
- **UniMax epoch-capping + "simulate scarcity."** Hard-cap epochs per task regardless of weight (fights overfit *and* forgetting); when tuning mixture weights on small proxy runs, **downsample to match the real epoch-repetition regime** or you'll overrate scarce high-quality sessions.
- **Dedup with MinHash + LSH.** Synthetic drivers emit near-duplicate scenarios (same template, swapped names — the "replace Canada with USA" failure). MinHash makes collision probability ≈ Jaccard; LSH bands/rows give a tunable **S-curve** at your similarity cutoff. **Dedup synthetic *against* real and against the eval set** (decontamination). Dedup at *scenario* granularity, not utterance (preserves flow).
- **Synthetic data = environments + tasks + strong-teacher responses.** Key findings: **multiple generations per scenario (~16) beats more seed sources**; **bigger teacher ≠ better teacher** (A/B your caller-driver model on eval quality); validate that synthetic dialogues stay in-character (drop "cheaters").

### What to skip
Common Crawl scale, crawling mechanics, the entire copyright/fair-use/NYT-Anthropic legal tour, HTML→text extraction, the C4/Pile/Dolma/RefinedWeb dataset history, code-corpus building. (All of L13 except the model-based-filtering recipe, which is folded in above.)

### Apply to Rehearse
- **Run the data-mixing math on your replay buffer** *now*: compute implied epochs-per-task; add **UniMax-style epoch caps**. This is the most concrete defense for the continual-learning pillar (`I-LoRA`/`CURLoRA` won't save you from a bad mixture).
- **Add a fastText/perplexity quality gate** to `train/pipeline/dataset.py` before manifest build; positives = top-`rwrd` sessions.
- **MinHash-LSH dedup** synthetic scenarios against each other, against real sessions, and against eval fixtures.
- **Synthetic generation:** generate **multiple variants per seed scenario**; A/B caller-driver teacher models on resulting eval quality rather than assuming the biggest is best.

### Source pointers
Skill `llm-data-curation`. Transcripts: `…lecture-13-data-source-segmented.md` (skim), `…lecture-14-data-segmented.md` (focus). Return-to: *"The number of epochs is 50… you can't just naively define a distribution and sample"* (the replay-buffer trap); *"cap the number of epochs / simulate epoching"* (UniMax); *"Synthetic data: defining environments and using stronger models as teachers."*

---

## Module 6 — Serving Moshi in real time
**Lecture 10 + Dan Fu guest · skill `llm-inference` · ~5–6 h**

### Why it matters
Voice is **latency-critical**: your SPEC's load-bearing number is **≤800 ms p50 speech-to-speech**, and Dan Fu literally names "phone voice mode" as the sub-second-TTFT regime. Self-hosting Moshi (`infra/interactive.py`) means you own this. The same material makes RL rollouts (M2) cheap.

### What to learn (MUST)
- **Arithmetic intensity → prefill vs decode.** Prefill (whole prompt) is compute-bound; **decode (one token at a time) is memory-bandwidth-bound** and won't amortize over batch. A single live call (B≈1) is *deeply* memory-bound — the GPU is mostly idle waiting on HBM. **This is the root cause of every Moshi latency number.**
- **KV cache** size = `B × S × KV-heads × H × layers × 2 × 2 bytes`, grows linearly with call length. Use the formula to bound A10G memory and decide max call length / batch.
- **Two configs, one model.** Live call = small batch, optimize **TTFT + per-token latency**; **prefetch the known scenario's prefill at call-connect** for low first-turn TTFT. RL rollouts = max batch + continuous batching, optimize **throughput**.
- **Serve through vLLM / SGLang, not raw aiohttp.** You get **continuous batching** (evict finished / admit new rollouts each step — keeps the GPU full as variable-length calls finish), **paged attention** + **prefix sharing** (cache the scenario/system-prompt prefill *once*, reuse across turns and rollouts), for free. SGLang is specifically best for agentic/multi-turn — Rehearse's exact shape.
- **Quantization is the highest-leverage serve-as-is quick win.** Decode is bandwidth-bound, so shrinking weight bytes (e.g., INT8/FP16 — pick an **A10G/Ampere-supported** format, *not* FP4) directly lowers per-token latency and frees memory for bigger rollout batches. PTQ needs no retraining; verify audio quality with your existing judges.
- **Request routing by length (~40% faster, two lines).** Don't let a cold call-start prefill (scenario load) share GPUs with warm in-call decode turns — route by request shape.
- **Production debugging priors (Dan Fu war stories).** Garbled/looping audio tokens → suspect a NaN/off-by-one kernel or KV-cache corruption *before* blaming the model or quantization; add a hard **max-token / max-call-duration guard** against doom loops.

### What to learn (SKIM — know it exists, don't build yet)
GQA/MLA/sliding-window/cross-layer KV sharing (model *properties* to check in Moshi's config, not things you implement); **speculative decoding** (lossless, high-impact, but needs a Moshi-compatible draft model over Mimi tokens — blocked today); **megakernels** (the batch-1–16 decode sweet spot is exactly live voice, but ~1 eng-year/model — *watch* for a community Moshi kernel); **PARSE / loop transformers** (architecturally interesting for a future param-efficient speech model; intellectually relevant, practically deferred).

### What to skip
Multi-GPU sharding, flimsy-interconnect/fault-tolerance, disaggregated specialized hardware (Groq/Cerebras), KV-disk-offloading depth. Moshi 7B fits one A10G.

### Apply to Rehearse
- **Migrate Moshi serving from raw aiohttp toward vLLM/SGLang** — single biggest structural latency+throughput upgrade; instruments TTFT/per-token cleanly for your `LatencyBreakdown`.
- **Profile decode vs prefill** with the PyTorch profiler (see Supporting) to confirm where the 800 ms budget actually goes before optimizing.
- **Try PTQ quantization** of Moshi; gate on audio-judge quality.
- **Prefetch the scenario prefill at call-connect** (you *know* the scenario before the call answers — trivial warm-start).
- **Add a hard generation cap** per turn/call.

### Source pointers
Skill `llm-inference`. Transcripts: `…lecture-10-inference-segmented.md`, `…guest-lecture-dan-fu-segmented.md`. Return-to: *"Prefill is compute-bound, but generation's intensity does not scale with batch, so it bottlenecks"* (the table behind every latency number); *"Simple request routing by length yields 40 percent faster serving."*

---

# Supporting / targeted (skim as needed)

You're "lighter on systems," so these are the **keeper concepts only** — enough to unblock the LoRA fine-tune and reason about latency. Skip the rest.

### S1 — Systems to fit the LoRA run (L8 + parts of L5/L6) · `llm-parallelism` · ~2–3 h
The OOM levers for your A10G, in priority order:
- **Activation memory ≈ `34·S·B·H` (+ attention term), NOT parameters, is what OOMs you** — and with ~100 s audio, **S is huge**. This formula is the lever for every batch-size / sequence-length decision; compute it for your Moshi config to predict fit.
- **Activation/flash-attention recomputation** drops the quadratic attention term cheaply — *the* single-GPU OOM fix to reach for before anything else. "More recompute → frees memory → bigger batch → better utilization."
- **Memory accounting:** ~5 weight copies @ ~16 B/param, **Adam optimizer state dominates** — which is *why LoRA wins* (you pay Adam state only on the tiny adapter). Validates `mixed_precision.py` (the bf16 master weight is a real, necessary line item).
- **FSDP/ZeRO** (only matters at world_size ≥ 2): stages 1–2 are "free" (`all-reduce ≡ reduce-scatter + all-gather`); stage 3 hides one all-gather under prefetch. **Trigger to learn deeply:** if you ever **full-FT Moshi 7B** (won't fit one A10G), "many 7B models train purely with FSDP" — a few GPUs, no tensor/pipeline parallel needed.
- **SKIP:** tensor/pipeline/expert/sequence parallel, NVLink/InfiniBand, collectives beyond the all-reduce identity, the per-model case studies. **One watch-item:** *context parallel / ring attention* if you ever extend Moshi's audio context on multi-GPU.

### S2 — Hyperparameters & cheap experiments (L11 + keepers from L9) · `llm-scaling-laws` · ~2–3 h
- **WSD (warmup-stable-decay) schedule = your default LoRA schedule.** Constant warmup → long stable phase → ~10–20% decay to ~10% peak. The payoff: **branch from a stable checkpoint and re-decay to extend a run or add data — instead of restarting.** Real per-iteration compute savings for your many short experiments; *never skip the final anneal.*
- **"Small-scale wins can bite you at scale."** Muon's edge shrank with scale; Cautious-Adam ran a beautiful law then blew up. Before trusting any optimizer/LR/weight-decay tweak from a Moshi smoke test: check ≥2 compute scales, and **make sure your baseline (Adam) is tuned** (an untuned baseline fakes gains).
- **LR/batch priors (DeepSeek/StepFun):** LR ↓ as model/params grow, batch ≈ √(data); grid **LR** (most sensitive) coarsely on a convex landscape, then univariate-sweep weight decay. **Treat all formulas as priors, not gospel — "scaling is still vibes"** — re-center on your own runs.
- **Upstream loss ≠ downstream performance** (the most Rehearse-relevant L9 point): never rank Moshi fine-tunes by val-loss/perplexity alone — wire a cheap *task-level* eval into the loop (you have one: the harness).
- **SKIP:** Chinchilla/compute-optimal allocation, tokens-per-param, isoFLOPs-for-sizing, MoE-sparsity scaling — you're not sizing a pretrain. Full MuP derivation = skim the concept ("reparameterize so the LR optimum doesn't move"), skip the math.

### S3 — GPU performance mental model (L5–L6) · `gpu-compute-planning`, `llm-gpu-kernels` · ~2 h, read-only
The prerequisite lens for Module 6 — concepts, not kernel-writing:
- **Roofline + memory hierarchy:** decode is on the memory-bound diagonal; "more FLOPs" is the wrong lever, reducing **bytes-moved** is the right one.
- **Low precision rules:** downcast matmul inputs, accumulate in FP32, keep softmax/norm/**last layer** high-precision — the guardrail for your bf16 and any future Moshi quantization.
- **Operator fusion / `torch.compile`:** fuses the scattered element-wise ops between matmuls into single kernels — the cheapest latency win on Moshi decode, no kernel code. Try `torch.compile(mode="max-autotune")`.
- **Flash attention** makes long *audio* context cheap on memory (tiling + online softmax + recompute) — ensure Moshi serving uses a flash-attention backend; never reimplement it.
- **Profile before optimizing:** CUDA-event timing (warmup + `synchronize`) + the PyTorch profiler tells you where the 800 ms actually goes.
- **Quick rule:** keep tunable dims (codebook/vocab, padded hidden/seq) **divisible by 32** to dodge wave-quantization cliffs.
- **SKIP (until a profiled perf wall):** hand-writing Triton/CUDA — program IDs, offsets/masks, PTX, occupancy/registers, bank conflicts, DSL choice. The `llm-gpu-kernels` skill is there if that day comes.

---

# Explicit skip / defer (with triggers)

| Topic | Status | Revisit when… |
|---|---|---|
| Compute-optimal sizing (Chinchilla, tokens/param, isoFLOPs) | **Skip** | You ever pretrain from scratch (not on the roadmap). |
| Tensor / pipeline / expert / sequence parallelism | **Skip** | You **full-FT Moshi** or scale multi-node (then: FSDP first, TP≤8 in-node). |
| Context parallel / ring attention | **Defer (watch)** | You extend Moshi's audio context beyond single-GPU memory. |
| Hand-writing Triton/CUDA kernels | **Defer** | A profiled Moshi-serving hot op has no good existing kernel. |
| Speculative decoding | **Defer** | A Mimi-token-compatible draft model becomes available. |
| Megakernels | **Watch** | A community Moshi/Thunderkittens decode kernel appears (batch 1–16 = your sweet spot). |
| PARSE / loop transformers | **Park** | You consider a param-efficient *future* self-hosted speech model. |
| Web crawling / licensing / dataset history (most of L13) | **Skip** | N/A — you own your data. |
| CLIP/SigLIP/ViT image specifics | **Skip** | N/A — keep only the encoder-adapter *analogy* (M4). |
| PPO "37 implementation details" | **Defer** | You need PPO+GAE for a dense `cont` reward and GRPO/DPO proved insufficient. |

---

# Suggested sequence & cadence

A ~6-week arc that front-loads post-training and ships a working weight-update loop, while pulling systems/serving in *just in time*.

**Phase A — Make the reward trustworthy, then close the cheap loop (Weeks 1–2)**
1. **M3 (Eval/judges)** first in practice — per-phase checklists, pairwise+ELO, length-debias, degenerate-baseline attack, private fixtures. *Gate: every dim ρ≥0.7 vs. gold, and no trivial agent games the rubric.* (Directly clears your `TODO.md` scoring bugs.)
2. **M1 (SFT→DPO)** — stand up rejection-sampling-SFT → DPO on top-`rwrd` trajectories. *Gate: one full outer-loop iteration with a measured rubric delta.*

**Phase B — Understand & stage the model (Week 3)**
3. **M4 (Moshi architecture)** — internalize encoder-adapter-LLM + Mimi discrete tokens + staged freeze/unfreeze; arm the trainer against mixed-token instability (QK-norm/Z-loss). Pull in **S1** (activation memory + recompute) and **S3** (bf16 rules) as needed to make the run fit.

**Phase C — The full RL lever (Week 4)**
4. **M2 (RLVR/GRPO)** — implement one-page GRPO; add length/latency reward for `spch`/`dlvr`; run the reward-hacking drill. Use **S2** (WSD schedule + LR/batch priors + tune-your-baseline) to set HPs cheaply.

**Phase D — Data flywheel & serving (Weeks 5–6)**
5. **M5 (Data/continual)** — replay-buffer epoch math + UniMax caps, quality gate, MinHash-LSH dedup, multi-variant synthetic generation.
6. **M6 (Serving)** — migrate Moshi to vLLM/SGLang, profile decode, try PTQ quantization, prefetch scenario prefill. *Gate: ≤800 ms p50 on a live call.*

> **Ordering note:** you asked to front-load post-training, so M1/M2 lead the headline. But in *execution* M3 comes first — *RLVR is only as robust as your reward*, and your SPEC already gates on judge calibration. The two are the same battle from different sides.

---

# Appendix — lecture → skill → transcript index

| Lecture | Skill | Transcript (`~/Desktop/youtube/…-segmented.md`) | Tier |
|---|---|---|---|
| L5 GPUs/TPUs | `gpu-compute-planning` | `…lecture-5-gpus-tpus` | Supporting (S3) |
| L6 Kernels/Triton/XLA | `llm-gpu-kernels` | `…lecture-6-kernels-triton-xla` | Supporting (S3) |
| L7 Parallelism I | `llm-parallelism` | `…lecture-7-parallelism` | Skip-for-now |
| L8 Parallelism II (FSDP) | `llm-parallelism` | `…lecture-8-parallelism` | Supporting (S1) |
| L9 Scaling Laws I | `llm-scaling-laws` | `…lecture-9-scaling-laws` | Skip (keep 1 idea) |
| L10 Inference | `llm-inference` | `…lecture-10-inference` | **Core (M6)** |
| L11 Scaling Laws II | `llm-scaling-laws` | `…lecture-11-scaling-laws` | Supporting (S2) |
| L12 Evaluation | `llm-evaluation` | `…lecture-12-evaluation` | **Core (M3)** |
| L13 Data sources | `llm-data-curation` | `…lecture-13-data-source` | Skip (keep filtering) |
| L14 Data transform/mix/synthetic | `llm-data-curation` | `…lecture-14-data` | **Core (M5)** |
| L15 Mid/Post-Training | `llm-mid-post-training` | `…lecture-15-midpost-tra` | **Core (M1)** |
| L16 Post-Training RLVR | `llm-post-training-rlvr` | `…lecture-16-post-traini` | **Core (M2)** |
| L17 Alignment/Multimodality | `llm-alignment-multimodality` | `…lecture-17-alignment---multimodality` | **Core (M4)** |
| Guest: Dan Fu (serving) | `llm-inference` | `…guest-lecture-dan-fu` | **Core (M6)** |

> Router skill: `llm-systems-design` (maps any sub-problem → the right specialist skill). Project transcript copies (raw + summary): `yt2md/docs/transcripts/`. Cross-video index: `~/Desktop/youtube/table-of-contents.md`.

*Generated 2026-06-08 from the CS336 Spring-2026 transcript library, oriented to Rehearse's `README.md` / `CLAUDE.md` / `SPEC.md` / `TODO.md`.*
