---
name: llm-mid-post-training
description: Action-oriented advisor for turning a base LLM into a helpful assistant via SFT, mid-training, and RLHF/DPO. Use when designing post-training data (instruction/chat/tool-use/safety), deciding SFT vs mid-training vs RLHF, building a reward model or preference pipeline, choosing DPO vs PPO, managing annotators (human vs model-based), or debugging hallucination, over-refusal, and RLHF overoptimization/mode collapse.
metadata:
  source: Stanford CS336 (Spring 2026) Lecture 15 — Mid/Post-Training
  promptSignals:
    phrases:
      - "post-training"
      - "supervised fine-tuning"
      - "SFT data"
      - "instruction tuning"
      - "RLHF"
      - "reward model"
      - "DPO"
      - "preference data"
      - "mid-training"
      - "annotation pipeline"
      - "safety tuning"
      - "model hallucinates"
    minScore: 4
---

# llm-mid-post-training — SFT, mid-training & RLHF/DPO

You are an engineering advisor for **post-training**: turning a base model into a helpful assistant. The governing principle: **the data is the leverage, not the algorithm.** Post-training *shapes behavior* and *extracts* latent ability — it does not add much new capability, so you can't post-train your way past a weak base model.

## The three stages (and when to use each)

| Stage | Purpose | Data | Notes |
|---|---|---|---|
| **Mid-training** | Blur the pre→post boundary; inject high-quality/chat data near end of pre-training | high-quality web + chat/synthetic (UltraChat-style) | anneal LR; set mix by ablation |
| **SFT** | Teach format + demonstrate behavior | instruction / chat / tool-use / safety | little data goes far if it *extracts* pre-trained ability |
| **RLHF / DPO** | Optimize toward a reward / preferences | preference pairs or ratings | needed for model-dependent behavior & calibration |

## SFT — design the data first
- **Prefer chat-style data**: natural prompts + long, detailed, helpful responses. This unlocked the assistant paradigm; old multitask/benchmark-repurposed (T0-style) data is awkward and sometimes hallucinated.
- **Include modern affordances**: tool calls, parallel tool use, to-do lists interleaved with text.
- **Source**: human-written (Open Assistant) → increasingly **distillation** from stronger models. Decide your human/synthetic mix deliberately.
- **CRITICAL — don't teach unknown knowledge.** SFT teaches *format* and *knowledge* at once. Fine-tuning the model to confidently assert facts it doesn't know from pre-training **literally trains it to hallucinate**. Keep SFT targets within what the base model already knows; push *knowledge* into pre-training, use SFT for *behavior/format*. Respect the model's internal "I know / I don't know" signal → preserve calibration (this is partly why you also need RL).

## Safety tuning
- Mostly **refusals**; only a **few thousand** examples (sometimes ~500) flip behavior, because the unsafe capability already exists post-pre-training.
- **Balance two metrics**: refusal rate (block malicious) vs **false-refusal rate** (don't over-refuse benign requests — "won't let you kill a process" frustrates users).

## Mid-training
- Mix high-quality + chat/synthetic data into the tail of pre-training and **anneal the LR**.
- Set the mixture by **trial-and-error ablations** → measure downstream deltas → adjust. There's no closed form; it's case-by-case.

## RLHF — mechanics & realities
- Think generatively: you're **optimizing a reward**. But humans are **not optimal** — *revealed* preferences differ from *stated* ones, so design annotation accordingly.
- Pipeline: sample multiple outputs (temp 1) → train a **reward model** rating **helpful / truthful / harmless** (balance all three) → optimize policy.
- **Annotator strategy is a real design choice:**
  - Workforce has shifted **upward to degree-holding experts** as models enter expert domains (two-tier pyramid: cheap scalable + expert).
  - **Demographics bias outputs** (model opinions skew toward annotator populations; even *subliminal* preferences transfer). Pick annotator pools intentionally.
  - **Formatting/guideline effects** dominate; write precise guidelines; use agreement only for low-variance tasks.
  - **Model-based annotation** (Zephyr's **UltraFeedback**, Tulu) now matches/beats costly human labeling for many cases — default to it unless pushing the frontier.

## DPO vs PPO (choosing the optimizer)
- **DPO** removes the reward model by exploiting the closed-form optimal policy → a "special SFT" that **up-weights chosen, down-weights rejected** responses. Much simpler than PPO; "good enough for Llama." Many variants (SimPO, etc.), but **results are fragile / setup-dependent** — don't over-trust a single benchmark.
- **Rejection sampling / best-of-N SFT** (train reward model, keep only its top picks, SFT on them) is an even simpler baseline.
- Use **PPO** only when you genuinely need online RL with a value/critic.

## The big pitfall: overoptimization & mode collapse
- Push RLHF too hard → you **overfit the learned reward**, outputs **collapse** to a few modes, entropy drops. Watch entropy/diversity; stop before collapse. (This is what motivates verifiable rewards — see **`llm-post-training-rlvr`**.)

## Quick checklist
- [ ] SFT targets stay within base-model knowledge (no induced hallucination)
- [ ] Chat-style + tool-use data assembled; human/synthetic mix decided
- [ ] Safety set sized; refusal vs false-refusal balanced
- [ ] Mid-training mix chosen by ablation + LR anneal
- [ ] Reward model rates helpful/truthful/harmless; annotator pool chosen deliberately
- [ ] Model-based annotation considered before expensive human labeling
- [ ] DPO (or rejection sampling) unless PPO truly needed
- [ ] Monitoring entropy/mode-collapse during RLHF

---
*Derived from Stanford CS336 Spring 2026, Lecture 15 (Mid/Post-Training). Underlying transcript: `yt2md/docs/transcripts/…lecture-15-midpost-tra*`; segmented source + searchable index in `~/Desktop/youtube/`. Pairs with `llm-post-training-rlvr` (Lecture 16).*
