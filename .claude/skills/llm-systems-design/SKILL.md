---
name: llm-systems-design
description: Directory and intelligent router for the CS336-derived LLM engineering skills. Use when designing or building an end-to-end LLM (pre-training → scaling → systems → post-training → alignment → eval → serving) and you want to route to the right specialist skill, sequence the work, or find the underlying lecture transcripts, summaries, and searchable index.
metadata:
  source: Stanford CS336 (Spring 2026) — Lectures 5–18
  promptSignals:
    phrases:
      - "design an LLM system"
      - "build an LLM from scratch"
      - "train a large language model"
      - "LLM systems engineering"
      - "which LLM skill"
      - "end to end LLM"
    minScore: 3
---

# llm-systems-design — router for building an LLM

A directory for the action-based skills distilled from **Stanford CS336: Language Modeling from Scratch (Spring 2026)**. Identify the sub-problem, then **invoke the matching specialist skill**. Each specialist is structured as *mental models + step-by-step procedures*.

## Route by group → skill

### 1. Plan & measure (decide before you spend compute)
| Working on… | Skill |
|---|---|
| Predicting loss, compute-optimal sizing (Chinchilla), tokens/param, hyperparameter transfer (MuP, LR/batch scaling, WSD), comparing archs/optimizers by slope | **`llm-scaling-laws`** (L9, L11) |
| Choosing/building benchmarks, perplexity, MMLU/GPQA/HLE, Arena/AlpacaEval/LLM-judge, SWE-bench, safety eval, decontamination | **`llm-evaluation`** (L12) |
| Sizing GPU memory/compute & provisioning on Modal (VRAM fit, GPU-hours, $) | **`gpu-compute-planning`** (existing) |

### 2. Data (signal per token)
| Working on… | Skill |
|---|---|
| Sourcing, HTML→text extraction, quality filtering, dedup (MinHash/LSH), decontamination, data mixing/epochs, synthetic data, licensing | **`llm-data-curation`** (L13, L14) |

### 3. Train at scale (systems & performance)
| Working on… | Skill |
|---|---|
| Sharding a model that won't fit / training faster — data/ZeRO/FSDP, tensor, pipeline, sequence/context, expert parallelism, 3D/4D, collectives, comms-bound utilization | **`llm-parallelism`** (L7, L8) |
| Making inference fast/cheap — latency vs throughput, KV-cache (GQA/MLA), quantization, speculative decoding, paged attention, continuous batching, prefill/decode disaggregation, megakernels | **`llm-inference`** (L10 + Dan Fu guest) |
| Writing fast custom kernels — fusing element-wise ops, Triton (softmax, tiled matmul, fused activations), occupancy/coalescing/bank-conflicts, torch.compile vs Triton vs PTX, profiling | **`llm-gpu-kernels`** (L6) |
| Sizing GPU memory/compute, the roofline & hardware model (GPUs/TPUs), provisioning | **`gpu-compute-planning`** (existing; covers L5) |

### 4. Post-training & alignment (turn a base model into a product)
| Working on… | Skill |
|---|---|
| SFT/instruction/chat/tool-use/safety data, mid-training, RLHF & DPO, annotation pipelines, hallucination/over-refusal/mode-collapse | **`llm-mid-post-training`** (L15) |
| RL with verifiable rewards — PPO/DPO/GRPO, reward verification, difficulty curation, reasoning ("thinking") models, rollout/training loop, reward hacking | **`llm-post-training-rlvr`** (L16) |
| Adding vision/other modalities — CLIP/SigLIP, ViT, encoder-adapter-LLM (LLaVA/Qwen-VL), resolution/video, image-token generation | **`llm-alignment-multimodality`** (L17) |

## Route by end-to-end build order
When building from scratch, walk the stack in this order, invoking each skill at its step:
1. **`llm-scaling-laws`** — decide model/data size & hyperparameters for your compute budget.
2. **`llm-data-curation`** — assemble and filter the pre-training corpus.
3. **`llm-parallelism`** — make the training run fit and go fast across GPUs.
4. **`llm-evaluation`** — stand up benchmarks (and decontamination) to measure progress.
5. **`llm-mid-post-training`** → **`llm-post-training-rlvr`** — SFT/mid-training, then RLHF/RLVR.
6. **`llm-alignment-multimodality`** — add modalities if needed.
7. **`llm-inference`** — serve it fast and cheap (the repeated cost).

## Route by symptom (quick triage)
- "OOM / model won't fit / training too slow" → **llm-parallelism**
- "serving too slow / costs too much / KV cache huge" → **llm-inference**
- "op is memory-bound / write a Triton kernel / fuse ops" → **llm-gpu-kernels**
- "how big a model / how many tokens / set LR" → **llm-scaling-laws**
- "model hallucinates / won't follow instructions / over-refuses" → **llm-mid-post-training**
- "want a reasoning model / RL on math+code" → **llm-post-training-rlvr**
- "which benchmark / scores look contaminated or gamed" → **llm-evaluation**
- "need it to see images/video" → **llm-alignment-multimodality**
- "build/clean the dataset" → **llm-data-curation**

## Underlying knowledge base
The skills are syntheses; primary sources live in the yt2md transcript library:
- **Project copies** (version-controlled): `yt2md/docs/transcripts/` — raw transcript + summary per lecture.
- **Searchable library**: `~/Desktop/youtube/` — `…-segmented.md` idea blocks, cross-video `table-of-contents.md`, `header-map.json`.
- **Look things up:**
  ```python
  from yt2md.scripts.retrieval import read_table_of_contents, get_transcript_section
  read_table_of_contents()                 # browse every lecture's idea blocks
  get_transcript_section("<exact header>") # pull the verbatim passage for an idea
  ```
  or invoke **`get-youtube-transcript`**.

## Coverage & how these are built
Each lecture is converted to markdown (`yt2md`), segmented into idea blocks, summarized, and its *actionable* engineering guidance distilled into a category skill (lessons consolidated across lectures — e.g. L7+L8 → `llm-parallelism`, L9+L11 → `llm-scaling-laws`, L13+L14 → `llm-data-curation`, L10+guest → `llm-inference`).

Indexed lectures: **5–17** and the **Dan Fu guest** lecture. **L5** (GPUs/TPUs hardware model) is covered by **`gpu-compute-planning`**; **L6** (Kernels/Triton/XLA) by **`llm-gpu-kernels`**. Every CS336 systems/data/post-training lecture in the playlist now maps to a skill.
